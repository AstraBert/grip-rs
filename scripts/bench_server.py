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


@dataclass
class Config:
    url: str
    queries: Path
    n_queries: int
    requests: int
    warmup: int
    levels: list[int]
    options: dict[str, int]


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
        return Sample(latency_ms, 0.0, False)
    return Sample(latency_ms, SearchResponse(**response.json())["search_time"], True)


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
    await run_level(config, queries, min(4, max(config.levels)), config.warmup)

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
