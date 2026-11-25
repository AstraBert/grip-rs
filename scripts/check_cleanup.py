#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28,<1",
#   "numpy==2.5.4",
#   "scipy==1.18.1",
# ]
# ///
"""Phase 2 checks for the background cleanup (heal + compaction), against a real grip-server.

The new documents are the ones that follow the indexed documents in the documents file:
if the index was built from the first N documents, documents N, N+1, ... are inserted.
So the id of a document is also its row in the documents file, and it must stay that way
after the cleanup.

The steps:
  1. Insert a few documents.
  2. Delete 45 percent of the documents. This is below the cleanup threshold: nothing happens.
  3. Delete 10 percent more. The threshold (half of the documents) is crossed, and the
     server starts a cleanup at its next tick.
  4. While the cleanup runs, keep inserting and deleting documents, and keep searching.
     These are the "late" updates that the cleanup has to copy into the new version.
  5. When the new version appears (the folder v1), check:
       - no search failed during the cleanup
       - /info agrees with what was acknowledged (documents - deleted = alive)
       - no deleted id is returned, and the recall holds
       - the ids did not change: a deleted document is gone (404), a late insert is there
       - a new insert gets the next id
  6. Restart the server: same /info and same results (the new version, with its logs).

The script never touches the index you pass in: it makes a work directory with a v0 folder
that holds symlinks to the index files. The server writes the new versions next to it.

--index-dir can be the folder written by `build` (a v0 folder inside) or an old flat folder.

Keep these in sync with grip-server: the cleanup threshold (0.5) and the check interval (10 s).

Build the server first: cargo build --release -p grip-server
"""

import argparse
import asyncio
import random
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from check_updates import (
    INDEX_FILES,
    Server,
    delete_ids,
    documents_in_index,
    exact_top_k,
    insert_docs,
    read_seismic,
    recall,
    row_json,
    run_queries,
)

CLEANUP_THRESHOLD = 0.5  # keep in sync with CLEANUP_THRESHOLD in grip-server


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--documents", type=Path, required=True, help="documents file (Seismic binary format)")
    p.add_argument("--queries", type=Path, required=True, help="queries file (Seismic binary format)")
    p.add_argument("--index-dir", type=Path, required=True, help="index built from the first documents of the file")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--server-bin", type=Path, default=Path("target/release/grip-server"))
    p.add_argument("--port", type=int, default=7535)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-query-dims", type=int, default=2)
    p.add_argument("--rescore-factor", type=int, default=5)
    p.add_argument("--insert-percent", type=float, default=2, help="documents inserted before the deletions")
    p.add_argument("--late-inserts", type=int, default=2000, help="most documents inserted during the cleanup")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--startup-timeout", type=float, default=120.0)
    p.add_argument("--cleanup-timeout", type=float, default=180.0)
    return p.parse_args()


class Ledger:
    """What the server has acknowledged: the ids that must be alive, and the next id to insert."""

    def __init__(self, first_alive: int):
        self.alive: set[int] = set(range(first_alive))
        self.next_row = first_alive

    def mask(self, n_rows: int) -> np.ndarray:
        m = np.zeros(n_rows, dtype=bool)
        m[list(self.alive)] = True
        return m


async def writer(
    client: httpx.AsyncClient, docs, ledger: Ledger, max_row: int, stop: asyncio.Event, rng: random.Random
) -> list[str]:
    """Insert one document and delete one random live document, until told to stop."""
    errors: list[str] = []
    while not stop.is_set() and ledger.next_row < max_row and not errors:
        row = ledger.next_row
        r = await client.post("/points", json=row_json(docs, row))
        if r.status_code != 201 or r.json()["external_id"] != row:
            errors.append(f"POST /points (document {row}): {r.status_code} {r.text}")
            break
        ledger.next_row += 1
        ledger.alive.add(row)

        victim = rng.randrange(ledger.next_row)
        while victim not in ledger.alive:
            victim = rng.randrange(ledger.next_row)
        r = await client.delete(f"/points/{victim}")
        if r.status_code != 202:
            errors.append(f"DELETE /points/{victim}: {r.status_code} {r.text}")
            break
        ledger.alive.discard(victim)
        await asyncio.sleep(0.002)
    return errors


async def reader(client: httpx.AsyncClient, queries, args, stop: asyncio.Event) -> tuple[int, list[str], float]:
    """Search all the time. Returns the number of searches, the errors, and the slowest one in ms."""
    n, errors, slowest, i = 0, [], 0.0, 0
    while not stop.is_set():
        start = time.perf_counter()
        r = await client.post(
            "/search",
            json={
                "vector": row_json(queries, i % queries.shape[0]),
                "top_k": args.top_k,
                "ef": args.ef,
                "patience": args.patience,
                "max_query_dims": args.max_query_dims,
                "rescore_factor": args.rescore_factor,
            },
        )
        slowest = max(slowest, (time.perf_counter() - start) * 1000)
        n += 1
        i += 1
        if r.status_code != 200:
            errors.append(f"POST /search: {r.status_code} {r.text}")
            break
        await asyncio.sleep(0.005)
    return n, errors, slowest


async def alive_count(client: httpx.AsyncClient) -> tuple[int, dict]:
    info = (await client.get("/info")).json()
    n = info["num_documents"]
    return n - round(info["deleted_ratio"] * n), info


async def main_async(args: argparse.Namespace, workdir: Path, server: Server, source_dir: Path) -> bool:
    n_main = documents_in_index(source_dir)
    n_init = round(n_main * args.insert_percent / 100)
    n_rows = n_main + n_init + args.late_inserts + 2
    all_docs = read_seismic(args.documents, args.vocab_size, n_rows)
    if all_docs.shape[0] < n_rows:
        raise SystemExit(
            f"the documents file has {all_docs.shape[0]} documents and {n_rows} are needed: the index "
            f"({n_main}), the first inserts ({n_init}), --late-inserts and 2 more. Build the index from "
            f"fewer documents, or lower --late-inserts"
        )
    queries = read_seismic(args.queries, args.vocab_size, args.n_queries)
    docs_t = all_docs.T.tocsc()
    rng = random.Random(args.seed)
    print(f"{n_main} indexed documents, {n_init} inserted first, up to {args.late_inserts} inserted during "
          f"the cleanup, {queries.shape[0]} queries\n")

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {name}{': ' + detail if detail else ''} {'ok' if good else 'FAIL'}")

    server.start()
    async with httpx.AsyncClient(base_url=server.url, timeout=120) as client:
        # ----- steps 1 to 3 -----
        print("Before the cleanup")
        await insert_docs(client, all_docs, range(n_main, n_main + n_init), n_main)
        total = n_main + n_init
        ledger = Ledger(total)
        ledger.next_row = total

        order = list(range(total))
        rng.shuffle(order)
        n_a = round(total * 0.45)
        n_b = round(total * 0.10)
        phase_a, phase_b = order[:n_a], order[n_a : n_a + n_b]

        await delete_ids(client, phase_a)
        ledger.alive -= set(phase_a)
        n_alive, info = await alive_count(client)
        report("45 percent deleted, below the threshold", n_alive == len(ledger.alive) and not (workdir / "v1").exists(),
               f"{n_alive} alive, ratio {info['deleted_ratio']:.3f}")
        results, _ = await run_queries(client, queries, args)
        truth = exact_top_k(queries, docs_t, ledger.mask(n_rows), args.top_k)
        before = recall(results, truth)
        print(f"  recall before the cleanup: {before:.4f}")

        await delete_ids(client, phase_b)
        ledger.alive -= set(phase_b)
        # the same documents are alive as when the cleanup starts: the fair "before"
        results, _ = await run_queries(client, queries, args)
        truth = exact_top_k(queries, docs_t, ledger.mask(n_rows), args.top_k)
        before_b = recall(results, truth)
        print(f"  recall with 55 percent deleted, before the cleanup: {before_b:.4f}")

        # ----- step 4: late updates and searches while the cleanup runs -----
        print("\nDuring the cleanup")
        stop = asyncio.Event()
        max_row = n_main + n_init + args.late_inserts
        w = asyncio.create_task(writer(client, all_docs, ledger, max_row, stop, rng))
        r = asyncio.create_task(reader(client, queries, args, stop))
        deadline = time.time() + args.cleanup_timeout
        while not (workdir / "v1").exists() and time.time() < deadline and not (w.done() and w.result()):
            await asyncio.sleep(0.2)
        appeared = (workdir / "v1").exists()
        # let a few more requests go through the new version
        await asyncio.sleep(1.0)
        stop.set()
        w_errors = await w
        n_searches, r_errors, slowest = await r

        report("the new version v1 appeared", appeared, f"{time.time() - (deadline - args.cleanup_timeout):.1f} s")
        report("no leftover v1.tmp", not (workdir / "v1.tmp").exists())
        late = ledger.next_row - (n_main + n_init)
        report("late inserts and deletes were all acknowledged", not w_errors, f"{late} inserts" if not w_errors else w_errors[0])
        report("no search failed", not r_errors, f"{n_searches} searches, slowest {slowest:.0f} ms" if not r_errors else r_errors[0])
        if not appeared:
            return False

        # ----- step 5: the new version -----
        print("\nAfter the cleanup")
        n_alive, info = await alive_count(client)
        report("/info agrees with the acknowledged updates", n_alive == len(ledger.alive),
               f"{n_alive} alive, expected {len(ledger.alive)}")

        results, _ = await run_queries(client, queries, args)
        leaked = sum(1 for res in results for i in res if i not in ledger.alive)
        truth = exact_top_k(queries, docs_t, ledger.mask(n_rows), args.top_k)
        after = recall(results, truth)
        report("no deleted id in the results", leaked == 0, f"{leaked} found")
        report("the recall holds", after >= before - 0.03, f"{before:.4f} -> {after:.4f}")
        print(f"  recall: 45 percent deleted {before:.4f}, 55 percent deleted {before_b:.4f}, after the cleanup {after:.4f}")

        gone = (await client.delete(f"/points/{phase_a[0]}")).status_code
        report("DELETE of an id removed before the cleanup", gone == 404, f"{gone}, expected 404")
        maybe = (await client.delete(f"/points/{phase_b[0]}")).status_code
        report("DELETE of an id removed around the cleanup", maybe in (204, 404), f"{maybe}, expected 204 or 404")

        victim = rng.choice(sorted(ledger.alive))
        status = (await client.delete(f"/points/{victim}")).status_code
        ledger.alive.discard(victim)
        report("DELETE of a live id", status == 202, f"{status}, expected 202")

        row = ledger.next_row
        resp = await client.post("/points", json=row_json(all_docs, row))
        got = resp.json().get("external_id") if resp.status_code == 201 else None
        report("the next insert gets the next id", got == row, f"{got}, expected {row}")
        ledger.next_row += 1
        ledger.alive.add(row)

        results, _ = await run_queries(client, queries, args)
        mine = (await client.post("/search", json={"vector": row_json(all_docs, row), "top_k": args.top_k,
                                                   "ef": args.ef, "patience": args.patience,
                                                   "max_query_dims": args.max_query_dims,
                                                   "rescore_factor": args.rescore_factor})).json()["ids"]
        exact_mine = exact_top_k(all_docs[row], docs_t, ledger.mask(n_rows), args.top_k)[0]
        report("the new document is found", row not in exact_mine or row in mine)
        info_before = (await client.get("/info")).json()

    # ----- step 6: restart -----
    print("\nAfter a restart")
    server.stop()
    server.start()
    async with httpx.AsyncClient(base_url=server.url, timeout=120) as client:
        info_after = (await client.get("/info")).json()
        after_results, _ = await run_queries(client, queries, args)
        report("/info is the same", info_after == info_before, f"{info_before} -> {info_after}" if info_after != info_before else "")
        report("search results are identical", after_results == results)
        row = ledger.next_row
        resp = await client.post("/points", json=row_json(all_docs, row))
        got = resp.json().get("external_id") if resp.status_code == 201 else None
        report("the next insert continues the numbering", got == row, f"{got}, expected {row}")
    return ok


def main() -> None:
    args = parse_args()
    if not args.server_bin.exists():
        raise SystemExit(f"{args.server_bin} not found. Build it with: cargo build --release -p grip-server")

    source_dir = args.index_dir / "v0" if (args.index_dir / "v0").is_dir() else args.index_dir
    with tempfile.TemporaryDirectory(prefix="grip-cleanup-") as tmp:
        workdir = Path(tmp)
        (workdir / "v0").mkdir()
        for name in INDEX_FILES:
            (workdir / "v0" / name).symlink_to((source_dir / name).resolve())
        server = Server(args.server_bin.resolve(), workdir, args.vocab_size, args.port, args.startup_timeout)
        try:
            ok = asyncio.run(main_async(args, workdir, server, source_dir))
        finally:
            server.stop()

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
