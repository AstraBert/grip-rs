#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1"]
# ///
import argparse
import asyncio
import math
import os
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import httpx

QUALITY_PERCENTILES = (5, 10, 25, 50)


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class SearchResponse(TypedDict):
    ids: list[int]
    scores: list[float]
    search_time: float
    total_time: float


class Info(TypedDict):
    vocab_size: int
    num_documents: int
    scale: float


class Sample(NamedTuple):
    latency_ms: float
    search_ms: float
    ok: bool
    ids: list[int]


class Quality(NamedTuple):
    recall: float
    mrr: float
    ndcg: float


@dataclass
class Config:
    url: str
    queries: Path
    n_queries: int
    requests: int
    warmup: int
    levels: list[int]
    options: dict[str, int]
    ground_truth: Path | None


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    errors: int
    qps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    server_search_ms: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Benchmark a running grip-server.")
    parser.add_argument("--url", default="http://localhost:7433")
    parser.add_argument(
        "--queries", type=Path, default=Path("crates/grip-rs/data/queries.bin")
    )
    parser.add_argument(
        "--n-queries", type=int, default=100, help="distinct queries to cycle over"
    )
    parser.add_argument(
        "--requests", type=int, default=2000, help="requests for each level"
    )
    parser.add_argument(
        "--warmup", type=int, default=200, help="requests before measuring"
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--groundtruth",
        type=Path,
        help="TSV of the exact neighbors (query, document, rank, score), to report quality",
    )
    for name in ("top-k", "ef", "patience", "max-query-dims", "rescore-factor"):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args()

    names = ("top_k", "ef", "patience", "max_query_dims", "rescore_factor")
    options = {n: getattr(args, n) for n in names if getattr(args, n) is not None}
    return Config(
        args.url,
        args.queries,
        args.n_queries,
        args.requests,
        args.warmup,
        args.levels,
        options,
        args.groundtruth,
    )


def read_queries(path: Path, limit: int) -> list[SparseVector]:
    """Read the first `limit` vectors of a Seismic binary file."""
    queries: list[SparseVector] = []
    with path.open("rb") as f:
        (total,) = struct.unpack("<I", f.read(4))
        for _ in range(min(total, limit)):
            (length,) = struct.unpack("<I", f.read(4))
            indices = struct.unpack(f"<{length}I", f.read(4 * length))
            values = struct.unpack(f"<{length}f", f.read(4 * length))
            queries.append({"indices": list(indices), "values": list(values)})
    return queries


async def timed_search(
    client: httpx.AsyncClient, vector: SparseVector, options: dict[str, int]
) -> Sample:
    start = time.perf_counter()
    response = await client.post("/search", json={"vector": vector, **options})
    latency_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        return Sample(latency_ms, 0.0, False, [])
    body = SearchResponse(**response.json())
    return Sample(latency_ms, body["search_time"], True, body["ids"])


async def run_level(
    config: Config, queries: list[SparseVector], concurrency: int, n_requests: int
) -> tuple[list[Sample], float]:
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(
        base_url=config.url.rstrip("/"), timeout=60, limits=limits
    ) as client:
        slots = asyncio.Semaphore(concurrency)

        async def one(i: int) -> Sample:
            async with slots:
                return await timed_search(
                    client, queries[i % len(queries)], config.options
                )

        start = time.perf_counter()
        samples = await asyncio.gather(*(one(i) for i in range(n_requests)))
        return list(samples), time.perf_counter() - start


def summarize(concurrency: int, samples: list[Sample], elapsed: float) -> LevelResult:
    good = [s for s in samples if s.ok]
    errors = len(samples) - len(good)
    if len(good) < 2:
        nan = math.nan
        return LevelResult(
            concurrency, len(samples), errors, nan, nan, nan, nan, nan, nan
        )

    latencies = [s.latency_ms for s in good]
    cuts = statistics.quantiles(latencies, n=100, method="inclusive")
    return LevelResult(
        concurrency=concurrency,
        requests=len(samples),
        errors=errors,
        qps=len(good) / elapsed,
        mean_ms=statistics.fmean(latencies),
        p50_ms=cuts[49],
        p95_ms=cuts[94],
        p99_ms=cuts[98],
        server_search_ms=statistics.fmean(s.search_ms for s in good),
    )


def print_table(results: list[LevelResult]) -> None:
    print(
        f"{'conc':>5} {'requests':>9} {'errors':>7} {'QPS':>9} "
        f"{'mean ms':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'search ms':>10}"
    )
    for r in results:
        print(
            f"{r.concurrency:>5} {r.requests:>9} {r.errors:>7} {r.qps:>9.0f} "
            f"{r.mean_ms:>8.2f} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} {r.p99_ms:>8.2f} "
            f"{r.server_search_ms:>10.3f}"
        )
    print(
        "\nms columns are measured by the client. `search ms` is the time inside `search`."
    )


def read_ground_truth(path: Path, n_queries: int) -> list[list[int]]:
    """Exact neighbors of each query, best first. A line is: query, document, rank, score."""
    ranked: dict[int, list[tuple[int, int]]] = {}
    for line in path.read_text().splitlines():
        query, doc, rank, _ = line.split("\t")
        if int(query) < n_queries:
            ranked.setdefault(int(query), []).append((int(rank), int(doc)))
    if len(ranked) < n_queries:
        raise SystemExit(
            f"{path} has ground truth for {len(ranked)} queries, expected {n_queries}"
        )
    return [[doc for _, doc in sorted(ranked[q])] for q in range(n_queries)]


def score_query(returned: list[int], expected: list[int]) -> Quality:
    """Recall, MRR and nDCG, with the exact neighbors as the relevant set."""
    k = len(expected)
    relevant = set(expected)
    hit_ranks = [rank for rank, doc in enumerate(returned[:k]) if doc in relevant]
    recall = len(hit_ranks) / k
    mrr = 1 / (hit_ranks[0] + 1) if hit_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 2) for rank in hit_ranks)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(k))
    return Quality(recall, mrr, dcg / ideal_dcg)


async def measure_quality(
    config: Config, queries: list[SparseVector], truth: list[list[int]]
) -> list[Quality]:
    """Send every query once. The results do not depend on the concurrency."""
    concurrency = min(4, max(config.levels))
    samples, _ = await run_level(config, queries, concurrency, len(queries))
    failed = sum(not sample.ok for sample in samples)
    if failed:
        print(f"WARNING: {failed} queries failed, they count as zero in the quality\n")
    return [
        score_query(sample.ids, expected)
        for sample, expected in zip(samples, truth, strict=True)
    ]


def print_quality(scores: list[Quality], k: int) -> None:
    columns = (
        ("recall", [s.recall for s in scores]),
        ("MRR", [s.mrr for s in scores]),
        ("nDCG", [s.ndcg for s in scores]),
    )
    header = f"{'metric':<8}{'mean':>9}" + "".join(
        f"{f'p{q}':>9}" for q in QUALITY_PERCENTILES
    )
    print(
        f"Quality at {k} over {len(scores)} queries, ground truth is the exact top {k}\n"
    )
    print(header)
    for name, values in columns:
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        percentiles = "".join(f"{cuts[q - 1]:>9.4f}" for q in QUALITY_PERCENTILES)
        print(f"{name:<8}{statistics.fmean(values):>9.4f}{percentiles}")
    print(
        "\np5 is the value that 5% of the queries are below, so a low p5 means a bad tail.\n"
    )


async def read_info(config: Config) -> Info | None:
    async with httpx.AsyncClient(base_url=config.url.rstrip("/"), timeout=10) as client:
        try:
            response = await client.get("/info")
        except httpx.ConnectError:
            return None
        return Info(**response.json())


async def run_bench(config: Config) -> int:
    info = await read_info(config)
    if info is None:
        print("The server does not answer. Is it running?")
        return 1
    print(
        f"server: {info['num_documents']} documents, vocab {info['vocab_size']}"
        f" | client machine: {os.cpu_count()} cores"
        f" | options: {config.options or 'server defaults'}\n"
    )

    queries = read_queries(config.queries, config.n_queries)
    truth = None
    if config.ground_truth is not None:
        truth = read_ground_truth(config.ground_truth, len(queries))
    await run_level(config, queries, min(4, max(config.levels)), config.warmup)
    if truth is not None:
        print_quality(await measure_quality(config, queries, truth), len(truth[0]))

    results: list[LevelResult] = []
    for concurrency in config.levels:
        samples, elapsed = await run_level(
            config, queries, concurrency, config.requests
        )
        results.append(summarize(concurrency, samples, elapsed))
    print_table(results)
    return 1 if any(r.errors for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_bench(parse_args())))
