#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1"]
# ///
import argparse
import asyncio
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import httpx


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class SearchResult(TypedDict):
    ids: list[int]
    scores: list[float]
    search_time: float
    total_time: float


class Info(TypedDict):
    vocab_size: int
    num_documents: int
    scale: float


@dataclass
class Config:
    url: str
    queries: Path
    n_queries: int
    concurrency: int
    expected: Path | None


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.failures += 1
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"{'PASS' if ok else 'FAIL'}  {name}{suffix}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Check a running grip-server.")
    parser.add_argument("--url", default="http://localhost:7433")
    parser.add_argument(
        "--queries", type=Path, default=Path("crates/grip-rs/data/queries.bin")
    )
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--expected", type=Path, help="TSV from `grip-index search --output`"
    )
    args = parser.parse_args()
    return Config(
        args.url, args.queries, args.n_queries, args.concurrency, args.expected
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


def read_cli_ids(path: Path) -> dict[int, set[int]]:
    """Read the `query<TAB>document<TAB>score` file of the CLI."""
    ids: dict[int, set[int]] = {}
    for line in path.read_text().splitlines():
        query, document, _ = line.split("\t")
        ids.setdefault(int(query), set()).add(int(document))
    return ids


async def search(
    client: httpx.AsyncClient, vector: SparseVector, **options: int
) -> httpx.Response:
    return await client.post("/search", json={"vector": vector, **options})


async def search_ids(
    client: httpx.AsyncClient, vector: SparseVector, **options: int
) -> list[int]:
    response = await search(client, vector, **options)
    response.raise_for_status()
    return SearchResult(**response.json())["ids"]


async def check_basics(client: httpx.AsyncClient, report: Report) -> Info:
    health = await client.get("/health")
    report.check("GET /health answers 200", health.status_code == 200)

    response = await client.get("/info")
    info = Info(**response.json())
    report.check("GET /info answers 200", response.status_code == 200)
    return info


async def check_search(
    client: httpx.AsyncClient, report: Report, query: SparseVector, info: Info
) -> None:
    response = await search(client, query)
    result = SearchResult(**response.json())
    report.check(
        "default search returns 10 ids and 10 scores", len(result["ids"]) == 10
    )
    report.check(
        "scores are in descending order",
        result["scores"] == sorted(result["scores"], reverse=True),
    )
    report.check("ids are unique", len(set(result["ids"])) == len(result["ids"]))
    report.check(
        "ids are inside the collection",
        all(0 <= i < info["num_documents"] for i in result["ids"]),
    )

    top_5 = await search_ids(client, query, top_k=5)
    report.check("top_k=5 returns 5 results", len(top_5) == 5)

    explicit = await search_ids(
        client, query, top_k=10, ef=100, patience=20, max_query_dims=2, rescore_factor=5
    )
    report.check("no options equals the documented defaults", result["ids"] == explicit)

    pairs = list(zip(query["indices"], query["values"], strict=True))
    random.Random(1).shuffle(pairs)
    shuffled: SparseVector = {
        "indices": [i for i, _ in pairs],
        "values": [v for _, v in pairs],
    }
    report.check(
        "a shuffled query gives the same ids",
        result["ids"] == await search_ids(client, shuffled),
    )


async def check_bad_input(
    client: httpx.AsyncClient, report: Report, query: SparseVector
) -> None:
    bad_options: list[tuple[str, dict[str, int]]] = [
        ("top_k = 0", {"top_k": 0}),
        ("ef smaller than top_k", {"top_k": 20, "ef": 10}),
        ("ef above the cap", {"ef": 100_000}),
        ("top_k above the cap", {"top_k": 100_000, "ef": 100_000}),
        ("max_query_dims = 0", {"max_query_dims": 0}),
        ("rescore_factor = 0", {"rescore_factor": 0}),
    ]
    for name, options in bad_options:
        response = await search(client, query, **options)
        report.check(
            f"400 for {name}", response.status_code == 400, str(response.status_code)
        )

    response = await client.post(
        "/search", json={"vector": {"indices": [1, 2, 3], "values": [1.0]}}
    )
    report.check(
        "400 for indices and values of different length", response.status_code == 400
    )

    response = await client.post("/search", json={"nothing": 1})
    report.check("400 for a body without `vector`", response.status_code == 400)


async def check_concurrency(
    client: httpx.AsyncClient,
    report: Report,
    queries: list[SparseVector],
    concurrency: int,
) -> list[list[int]]:
    expected = [await search_ids(client, q) for q in queries]
    jobs = queries * 5
    limit = asyncio.Semaphore(concurrency)

    async def run(vector: SparseVector) -> list[int]:
        async with limit:
            return await search_ids(client, vector)

    got = await asyncio.gather(*(run(v) for v in jobs))
    wrong = sum(g != w for g, w in zip(got, expected * 5, strict=True))
    report.check(
        f"{len(jobs)} parallel searches ({concurrency} at a time) match the sequential ones",
        wrong == 0,
        f"{wrong} differ",
    )
    return expected


def check_cli(report: Report, path: Path, expected: list[list[int]]) -> None:
    cli = read_cli_ids(path)
    total = min(len(expected), len(cli))
    differ = sum(set(expected[i]) != cli[i] for i in range(total))
    report.check(
        f"server ids equal the CLI ids for {total} queries",
        differ == 0,
        f"{differ} differ",
    )


async def check_vocabulary(
    client: httpx.AsyncClient, report: Report, query: SparseVector, vocab_size: int
) -> None:
    # `vocab_size` is the first id that is too big
    for term in (vocab_size, vocab_size + 5):
        response = await search(client, {"indices": [term], "values": [1.0]})
        report.check(
            f"400 for the term id {term} (outside the vocabulary)",
            response.status_code == 400,
            str(response.status_code),
        )

    still_ok = await search(client, query)
    report.check(
        "the server still answers normally at the end", still_ok.status_code == 200
    )


async def run_checks(config: Config) -> int:
    report = Report()
    queries = read_queries(config.queries, config.n_queries)

    async with httpx.AsyncClient(base_url=config.url.rstrip("/"), timeout=30) as client:
        try:
            info = await check_basics(client, report)
        except httpx.ConnectError:
            print("The server does not answer. Is it running?")
            return 1

        await check_search(client, report, queries[0], info)
        await check_bad_input(client, report, queries[0])
        expected = await check_concurrency(client, report, queries, config.concurrency)
        if config.expected is not None:
            check_cli(report, config.expected, expected)
        # last, because a server without this check can crash
        await check_vocabulary(client, report, queries[0], info["vocab_size"])

    print(
        f"\n{report.failures} check(s) failed"
        if report.failures
        else "\nall checks passed"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_checks(parse_args())))
