#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Measure what a cleanup costs the searches, on a grip-server that is already running.

The script does not start the server. It needs a fresh index (no deletes, empty delta).

Steps:
  1. Bring the server just below the cleanup trigger (deletes: 45 percent deleted,
     delta: one document short of a full delta).
  2. Run searches in a loop and record every one. This is the baseline.
  3. Cross the trigger. The server starts a cleanup within its check interval (10 s).
  4. Keep searching (and inserting a few documents) until the cleanup is done.
  5. Keep searching a little longer, then print the three periods side by side.

How the end of a cleanup is detected, without touching the server's folder:
  deletes: the number of documents in /info drops by a lot.
  delta:   an insert is accepted again (until then the server answers 507).
"""

import argparse
import asyncio
import csv
import gc
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path

import httpx

MAX_DELTA = 50_000  # the limit in grip-rs/src/delta.rs
SEARCH_OPTIONS = ("top_k", "ef", "patience", "max_query_dims", "rescore_factor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a cleanup on a running grip-server.")
    p.add_argument("--url", default="http://localhost:7433")
    p.add_argument("--mode", choices=["deletes", "delta"], default="deletes")
    p.add_argument("--queries", type=Path, default=Path("crates/grip-rs/data/queries.bin"))
    p.add_argument(
        "--documents",
        type=Path,
        default=Path("crates/grip-rs/data/documents.bin"),
        help="the vectors to insert (the first ones of the file, so they are copies of indexed documents)",
    )
    p.add_argument("--n-queries", type=int, default=200, help="distinct queries to cycle over")
    p.add_argument("--concurrency", type=int, default=8, help="searches in flight")
    p.add_argument("--insert-rate", type=float, default=20, help="inserts per second during the cleanup. 0 turns them off (deletes mode only)")
    p.add_argument("--baseline-seconds", type=float, default=20)
    p.add_argument("--after-seconds", type=float, default=20)
    p.add_argument("--cleanup-timeout", type=float, default=1800, help="seconds to wait for the cleanup")
    p.add_argument("--threshold", type=float, default=0.5, help="deleted ratio that triggers the server")
    p.add_argument("--pre-percent", type=float, default=45, help="deletes: percent deleted before the baseline")
    p.add_argument("--delete-percent", type=float, default=55, help="deletes: percent deleted at the end")
    p.add_argument("--delta-fill", type=int, default=MAX_DELTA, help="delta: documents that fill the delta")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--csv", type=Path, help="write one row for each second to this file")
    p.add_argument(
        "--keep-gc",
        action="store_true",
        help="keep Python's garbage collector on. By default it is off while the load runs, because a "
        "full collection over the records of 100,000 searches can freeze the whole client for a while.",
    )
    p.add_argument(
        "--shared-client",
        action="store_true",
        help="send the searches, inserts and deletes through one httpx client (one connection pool). "
        "By default each search worker and the inserter have their own client.",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="split the time of each search into connect, send, wait for the server and read "
        "(a little slower, so use it to find a stall, not for the final numbers)",
    )
    for name in SEARCH_OPTIONS:
        p.add_argument(f"--{name.replace('_', '-')}", type=int)
    return p.parse_args()


def read_vectors(path: Path, limit: int) -> list[dict]:
    """The first `limit` vectors of a Seismic binary file."""
    out = []
    with path.open("rb") as f:
        (total,) = struct.unpack("<I", f.read(4))
        for _ in range(min(total, limit)):
            (n,) = struct.unpack("<I", f.read(4))
            idx = struct.unpack(f"<{n}I", f.read(4 * n))
            val = struct.unpack(f"<{n}f", f.read(4 * n))
            out.append({"indices": list(idx), "values": list(val)})
    return out


class Search(NamedTuple):
    start: float  # seconds since t0
    ms: float  # measured by the client
    ok: bool
    search_ms: float  # `search_time` of the response: only the search, after the locks are taken
    total_ms: float  # `total_time` of the response: the whole handler, with the wait for the locks
    phases: dict[str, float] | None = None  # with --trace: where the client time went, in ms


@dataclass
class Run:
    """Everything the tasks share. Times are seconds since `t0`."""

    t0: float = field(default_factory=time.perf_counter)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    trace: bool = False
    t_trigger: float | None = None
    t_done: float | None = None
    searches: list[Search] = field(default_factory=list)
    writes: list[tuple[float, float, int]] = field(default_factory=list)  # (start, ms, status)
    lags: list[tuple[float, float]] = field(default_factory=list)  # (start, ms late)

    def now(self) -> float:
        return time.perf_counter() - self.t0


def split_phases(marks: dict[str, float], t_request: float, t_end: float) -> dict[str, float]:
    """Where the client time of one request went, in ms, from the httpcore trace events.

    wait for a connection = from the call until the request headers start to go out (a free
                            connection from the pool, or a new one). `new` tells which.
    connect               = opening a new TCP connection (0 when one was reused)
    send                  = writing the request
    server                = after the request is sent, until the response headers arrive
    read                  = reading the response body
    """

    def at(name: str) -> float | None:
        return marks.get(name)

    def gap(a: float | None, b: float | None) -> float:
        return (b - a) * 1000 if a is not None and b is not None else 0.0

    sent = at("http11.send_request_body.complete") or at("http11.send_request_headers.complete")
    headers = at("http11.receive_response_headers.complete")
    body = at("http11.receive_response_body.complete")
    connect = gap(at("connection.connect_tcp.started"), at("connection.connect_tcp.complete"))
    return {
        "new": 1.0 if at("connection.connect_tcp.started") is not None else 0.0,
        "connect": connect,
        "pool": gap(t_request, at("http11.send_request_headers.started")) - connect,
        "send": gap(at("http11.send_request_headers.started"), sent),
        "server": gap(sent, headers),
        "read": gap(headers, body),
    }


async def search_loop(client: httpx.AsyncClient, run: Run, queries: list[dict], options: dict, first: int) -> None:
    i = first
    while not run.stop.is_set():
        body = {"vector": queries[i % len(queries)], **options}
        i += 1
        start = run.now()
        search_ms = total_ms = float("nan")
        marks: dict[str, float] = {}
        extensions = {}
        if run.trace:
            t_request = time.perf_counter()

            async def note(event: str, info: dict) -> None:
                marks.setdefault(event, time.perf_counter())

            extensions = {"trace": note}
        try:
            r = await client.post("/search", json=body, extensions=extensions)
            t_response = time.perf_counter()
            ok = r.status_code == 200
            if ok:
                payload = r.json()
                search_ms, total_ms = payload["search_time"], payload["total_time"]
        except httpx.HTTPError:
            ok = False
        phases = split_phases(marks, t_request, t_response) if run.trace and ok else None
        run.searches.append(Search(start, (run.now() - start) * 1000, ok, search_ms, total_ms, phases))


async def lag_loop(run: Run) -> None:
    """Asks the event loop for a wake-up every 10 ms and records how late it comes.

    If the loop (or the CPU under it) is too busy, the client times include that delay."""
    while not run.stop.is_set():
        start = run.now()
        await asyncio.sleep(0.01)
        run.lags.append((start, max(0.0, (run.now() - start - 0.01) * 1000)))


async def insert_loop(
    client: httpx.AsyncClient, run: Run, docs: list[dict], rate: float, mode: str
) -> None:
    """A few writers during the cleanup, to test the writes that arrive while it runs."""
    i = 0
    while not run.stop.is_set():
        start = run.now()
        try:
            r = await client.post("/points", json=docs[i % len(docs)])
            status = r.status_code
        except httpx.HTTPError:
            status = 0
        run.writes.append((start, (run.now() - start) * 1000, status))
        if status == 201:
            i += 1
            # delta mode: the server refused (507) until the swap. The first accepted insert is the end.
            if mode == "delta" and run.t_done is None:
                run.t_done = run.now()
        await asyncio.sleep(1 / rate)


async def get_info(client: httpx.AsyncClient) -> dict:
    return (await client.get("/info")).json()


async def delete_until(
    client: httpx.AsyncClient, order: list[int], pos: int, target: float, run: Run | None, threshold: float
) -> int:
    """Delete ids from `order[pos:]` until the deleted ratio reaches `target`. Returns the new position.

    If `run` is given, the moment the ratio crosses `threshold` is the trigger time."""
    sem = asyncio.Semaphore(16)
    failures = 0

    async def one(i: int) -> None:
        nonlocal failures
        async with sem:
            r = await client.delete(f"/points/{i}")
            if r.status_code not in (202, 204, 404):
                failures += 1

    chunk = 2000
    while pos < len(order):
        info = await get_info(client)
        if run is not None and run.t_trigger is None and info["deleted_ratio"] >= threshold:
            run.t_trigger = run.now()
        if info["deleted_ratio"] >= target:
            break
        part = order[pos : pos + chunk]
        pos += len(part)
        await asyncio.gather(*(one(i) for i in part))
    if failures:
        raise SystemExit(f"{failures} deletes failed with an unexpected status")
    return pos


async def fill_delta(client: httpx.AsyncClient, docs: list[dict], count: int) -> None:
    sem = asyncio.Semaphore(8)

    async def one(i: int) -> None:
        async with sem:
            r = await client.post("/points", json=docs[i % len(docs)])
            if r.status_code != 201:
                raise SystemExit(f"POST /points returned {r.status_code} while filling the delta: {r.text}")

    for start in range(0, count, 2000):
        await asyncio.gather(*(one(i) for i in range(start, min(start + 2000, count))))
        print(f"  filled {min(start + 2000, count)} of {count}", end="\r", flush=True)
    print()


def pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    return sorted_values[min(len(sorted_values) - 1, int(q / 100 * len(sorted_values)))]


def period_row(name: str, rows: list[Search], seconds: float) -> str:
    ms = sorted(r.ms for r in rows if r.ok)
    errors = sum(1 for r in rows if not r.ok)
    qps = len(ms) / seconds if seconds > 0 else float("nan")
    return (
        f"{name:<10}{seconds:>8.1f}{len(rows):>9}{qps:>8.0f}{pct(ms, 50):>9.2f}{pct(ms, 95):>9.2f}"
        f"{pct(ms, 99):>9.2f}{(ms[-1] if ms else float('nan')):>10.1f}{errors:>8}"
    )


async def run_bench(args: argparse.Namespace) -> int:
    if args.mode == "delta" and args.insert_rate <= 0:
        raise SystemExit("--insert-rate must be above 0 in delta mode: an accepted insert marks the end of the cleanup")
    base = args.url.rstrip("/")
    options = {n: getattr(args, n) for n in SEARCH_OPTIONS if getattr(args, n) is not None}
    rng = random.Random(args.seed)

    async with httpx.AsyncClient(base_url=base, timeout=300, limits=httpx.Limits(max_connections=64)) as client:
        try:
            info = await get_info(client)
        except httpx.HTTPError:
            print("The server does not answer. Is it running?")
            return 1
        n_start = info["num_documents"]
        print(
            f"server: {n_start} documents, {info['deleted_ratio']:.3f} deleted | mode: {args.mode} | "
            f"{'one shared client' if args.shared_client else 'one client for each search worker'}"
        )
        if info["deleted_ratio"] > 0:
            print("WARNING: the index already has deletes. Use a fresh index for clean numbers.")

        queries = read_vectors(args.queries, args.n_queries)
        n_docs = args.delta_fill + 1000 if args.mode == "delta" else 5000
        docs = read_vectors(args.documents, n_docs)
        run = Run()
        run.trace = args.trace

        # 1. bring the server just below the trigger
        order = list(range(n_start))
        rng.shuffle(order)
        pos = 0
        if args.mode == "deletes":
            print(f"deleting up to {args.pre_percent:.0f} percent (below the trigger) ...")
            pos = await delete_until(client, order, 0, args.pre_percent / 100, None, args.threshold)
        else:
            print(f"filling the delta with {args.delta_fill - 1} documents (one short of full) ...")
            await fill_delta(client, docs, args.delta_fill - 1)

        # Each search worker and the inserter get a client of their own, with one connection, so a
        # slow request on one of them cannot hold up the others in a shared pool.
        def new_client() -> httpx.AsyncClient:
            if args.shared_client:
                return client
            return httpx.AsyncClient(
                base_url=base, timeout=300, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1)
            )

        search_clients = [new_client() for _ in range(args.concurrency)]
        writer_client = new_client()

        async def close_clients() -> None:
            if not args.shared_client:
                await asyncio.gather(*(c.aclose() for c in [*search_clients, writer_client]))

        if not args.keep_gc:
            gc.collect()
            gc.freeze()  # what exists now is never looked at again
            gc.disable()

        # 2. baseline
        print("warm up ...")
        run.stop.clear()
        run.t0 = time.perf_counter()
        workers = [
            asyncio.create_task(search_loop(search_clients[k], run, queries, options, k * 17))
            for k in range(args.concurrency)
        ]
        workers.append(asyncio.create_task(lag_loop(run)))
        print(f"baseline for {args.baseline_seconds:.0f} s ...")
        await asyncio.sleep(args.baseline_seconds)

        # 3. cross the trigger
        writer = None
        if args.mode == "deletes":
            print(f"deleting up to {args.delete_percent:.0f} percent (crosses the trigger) ...")
            await delete_until(client, order, pos, args.delete_percent / 100, run, args.threshold)
            if run.t_trigger is None:
                print("the deleted ratio never crossed the threshold, nothing to measure")
                run.stop.set()
                await asyncio.gather(*workers)
                await close_clients()
                return 1
            if args.insert_rate > 0:
                writer = asyncio.create_task(insert_loop(writer_client, run, docs, args.insert_rate, args.mode))
            docs_at_trigger = (await get_info(client))["num_documents"]
        else:
            r = await client.post("/points", json=docs[args.delta_fill % len(docs)])
            if r.status_code != 201:
                raise SystemExit(f"the last insert returned {r.status_code}, the delta was not one short of full")
            run.t_trigger = run.now()
            docs_at_trigger = n_start + args.delta_fill
            writer = asyncio.create_task(insert_loop(writer_client, run, docs, args.insert_rate, args.mode))
        print(f"trigger crossed at t = {run.t_trigger:.1f} s, waiting for the cleanup ...")

        # 4. wait for the end
        deadline = time.perf_counter() + args.cleanup_timeout
        while run.t_done is None and time.perf_counter() < deadline:
            await asyncio.sleep(0.25)
            if args.mode == "deletes":
                if (await get_info(client))["num_documents"] < docs_at_trigger * 0.8:
                    run.t_done = run.now()
        if run.t_done is None:
            print(f"the cleanup did not finish in {args.cleanup_timeout:.0f} s")
        else:
            print(f"cleanup finished {run.t_done - run.t_trigger:.1f} s after the trigger")

        # 5. after
        await asyncio.sleep(args.after_seconds)
        run.stop.set()
        end = run.now()
        await asyncio.gather(*workers, *([writer] if writer else []))
        await close_clients()
        gc.enable()

    report(args, run, end)
    return 0 if run.t_done is not None and not any(not s[2] for s in run.searches) else 1


def server_row(name: str, rows: list[Search]) -> str:
    cells = ""
    for field_name, width in (("search_ms", 10), ("total_ms", 11)):
        values = sorted(getattr(r, field_name) for r in rows if r.ok)
        cells += f"{pct(values, 50):>{width}.3f}{pct(values, 99):>9.3f}{(values[-1] if values else float('nan')):>10.1f}"
    return f"{name:<10}{cells}"


def report(args: argparse.Namespace, run: Run, end: float) -> None:
    trigger = run.t_trigger
    done = run.t_done if run.t_done is not None else end
    periods = [
        ("before", [s for s in run.searches if s[0] < trigger], trigger),
        ("during", [s for s in run.searches if trigger <= s[0] < done], done - trigger),
        ("after", [s for s in run.searches if s[0] >= done], end - done),
    ]
    print(f"\nSearch latency as the client sees it ({args.concurrency} searches in flight)\n")
    print(f"{'period':<10}{'seconds':>8}{'queries':>9}{'qps':>8}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>10}{'errors':>8}")
    for name, rows, seconds in periods:
        print(period_row(name, rows, seconds))
    print("\n'during' starts at the trigger, so it includes the wait for the server's next check (up to 10 s).")

    print("\nInside the server, from the `search_time` and `total_time` of each response\n")
    print(f"{'':<10}{'search p50':>10}{'p99':>9}{'max ms':>10}{'total p50':>11}{'p99':>9}{'max ms':>10}")
    for name, rows, _ in periods:
        print(server_row(name, rows))
    print(
        "\nsearch = only the search, after the locks are taken. It shows what the CPU and memory pressure of\n"
        "the cleanup does to the search itself.\n"
        "total = the whole handler, including the wait for the locks. total minus search is the lock wait\n"
        "(and the rebuild of the scratch space after a swap).\n"
        "client minus total is the network and the queue in front of the server."
    )

    print("\nHow late the client's own event loop wakes up (asked every 10 ms)\n")
    print(f"{'':<10}{'p99 ms':>9}{'max ms':>10}")
    for name, lo, hi in (("before", 0, trigger), ("during", trigger, done), ("after", done, end)):
        v = sorted(ms for t, ms in run.lags if lo <= t < hi)
        print(f"{name:<10}{pct(v, 99):>9.2f}{(v[-1] if v else float('nan')):>10.1f}")
    print(
        "\nIf this is large, the client itself is the slow part: its loop is busy or the CPU is taken\n"
        "by the server. Then client minus total says nothing about the server."
    )

    traced = [s for s in run.searches if s.phases]
    if traced:
        print("\nWhere the time went in the slow searches (--trace)\n")
        slow = [s for s in traced if s.ms > 200]
        names = ("pool", "connect", "send", "server", "read")
        biggest = {n: 0 for n in names}
        for s in slow:
            biggest[max(names, key=lambda n: s.phases[n])] += 1
        print(f"{len(slow)} of {len(traced)} searches took more than 200 ms. The longest part was:")
        print("  " + ", ".join(f"{n} {biggest[n]}" for n in names))
        print(f"\n{'client ms':>10}{'new conn':>9}{'pool':>9}{'connect':>9}{'send':>9}{'server':>9}{'read':>9}{'other':>9}")
        for s in sorted(traced, key=lambda s: -s.ms)[:5]:
            p = s.phases
            other = s.ms - sum(p[n] for n in names)
            print(
                f"{s.ms:>10.1f}{'yes' if p['new'] else 'no':>9}{p['pool']:>9.1f}{p['connect']:>9.1f}"
                f"{p['send']:>9.1f}{p['server']:>9.1f}{p['read']:>9.1f}{other:>9.1f}"
            )
        print(
            "\npool = waiting for a connection, connect = opening a new one, send = writing the request,\n"
            "server = from sent until the first byte of the answer, read = the body, other = the client (JSON, loop)."
        )

    print("\nThe five slowest searches\n")
    print(f"{'client ms':>10}{'total ms':>10}{'search ms':>10}  {'s after the trigger':>20}  {'s before the end':>17}")
    for s in sorted(run.searches, key=lambda s: -s.ms)[:5]:
        print(
            f"{s.ms:>10.1f}{s.total_ms:>10.1f}{s.search_ms:>10.3f}  {s.start - trigger:>20.1f}  {done - s.start:>17.1f}"
            f"{'' if s.ok else '  (failed)'}"
        )

    during_writes = [w for w in run.writes if trigger <= w[0] < done]
    if during_writes:
        by_status: dict[int, int] = {}
        for _, _, st in during_writes:
            by_status[st] = by_status.get(st, 0) + 1
        slowest = max(w[1] for w in during_writes)
        print(f"\nInserts during the cleanup: {dict(sorted(by_status.items()))}, slowest {slowest:.1f} ms")
        if args.mode == "delta":
            print("(507 is expected in delta mode until the swap)")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["second", "period", "queries", "client_p50", "client_p99", "client_max",
                 "search_p50", "search_p99", "search_max", "total_p50", "total_p99", "total_max"]
            )
            buckets: dict[int, list[Search]] = {}
            for s in run.searches:
                if s.ok:
                    buckets.setdefault(int(s.start), []).append(s)
            for sec in sorted(buckets):
                rows = buckets[sec]
                label = "before" if sec < trigger else ("during" if sec < done else "after")
                cells: list[str] = []
                for name in ("ms", "search_ms", "total_ms"):
                    v = sorted(getattr(r, name) for r in rows)
                    cells += [f"{pct(v, 50):.3f}", f"{pct(v, 99):.3f}", f"{v[-1]:.3f}"]
                w.writerow([sec, label, len(rows), *cells])
        print(f"\nper-second numbers written to {args.csv}")


if __name__ == "__main__":
    sys.exit(asyncio.run(run_bench(parse_args())))
