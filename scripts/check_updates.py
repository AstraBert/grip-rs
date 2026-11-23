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
"""Phase 3 checks for inserts, against a real grip-server.

The new documents are the ones that follow the indexed documents in the documents file:
if the index was built from the first N documents, documents N, N+1, ... are inserted.

Check 1: insert documents (0, 1, 2 and 5 percent of the index size, cumulative) with
         POST /points. After each step, compare the search results with an exact search
         over the main and the inserted documents. Reports recall@k, the share of the
         results that come from the delta, and the search time.
         Also verifies that the server hands out the ids n, n+1, ... in order.
Check 2: delete 10 percent of all the documents (main and inserted), check that no deleted
         id comes back and that the recall holds.
Check 3: restart the server, and verify that the inserts and deletions survived:
         same /info, same results, and the next insert continues the numbering.

The script never touches the index you pass in: it makes a work directory with symlinks to
index.bin, vectors.bin and sketches.bin, and the server writes delta.log and delete.log
there. Any delta.log or delete.log in the source directory is ignored.

Build the server first: cargo build --release -p grip-server
"""

import argparse
import asyncio
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from scipy import sparse

INDEX_FILES = ("index.bin", "vectors.bin", "sketches.bin")
DELTA_LOG = "delta.log"
DELETE_LOG = "delete.log"
MAX_DELTA = 50_000  # keep in sync with grip_rs::delta::MAX_DELTA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--documents", type=Path, required=True, help="documents file (Seismic binary format)")
    p.add_argument("--queries", type=Path, required=True, help="queries file (Seismic binary format)")
    p.add_argument("--index-dir", type=Path, required=True, help="index built from the first documents of the file")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--server-bin", type=Path, default=Path("target/release/grip-server"))
    p.add_argument("--port", type=int, default=7534)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-query-dims", type=int, default=2)
    p.add_argument("--rescore-factor", type=int, default=5)
    p.add_argument("--ratios", type=float, nargs="+", default=[0, 1, 2, 5], help="percent of the index size")
    p.add_argument("--delete-percent", type=float, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--startup-timeout", type=float, default=120.0)
    return p.parse_args()


# ---------- data ----------


def read_seismic(path: Path, vocab_size: int, limit: int) -> sparse.csr_matrix:
    """Seismic format: u32 count, then per vector a u32 length, `length` u32 ids, `length` f32 values.

    The file is streamed, and reading stops after `limit` vectors (the file can be huge).
    """
    indptr, idx, val = [0], [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        n = min(n, limit)
        for _ in range(n):
            (length,) = struct.unpack("<I", f.read(4))
            buf = f.read(8 * length)
            idx.append(np.frombuffer(buf, dtype="<u4", count=length))
            val.append(np.frombuffer(buf, dtype="<f4", count=length, offset=4 * length))
            indptr.append(indptr[-1] + length)
    return sparse.csr_matrix(
        (np.concatenate(val), np.concatenate(idx).astype(np.int64), np.array(indptr)),
        shape=(n, vocab_size),
    )


def documents_in_index(index_dir: Path) -> int:
    with open(index_dir / "vectors.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def row_json(m: sparse.csr_matrix, i: int) -> dict:
    a, b = m.indptr[i], m.indptr[i + 1]
    return {"indices": m.indices[a:b].tolist(), "values": m.data[a:b].tolist()}


def nnz_after_validation(m: sparse.csr_matrix, i: int) -> int:
    """The server drops zero values, so the log record has this many terms."""
    a, b = m.indptr[i], m.indptr[i + 1]
    return int(np.count_nonzero(m.data[a:b]))


def exact_top_k(queries: sparse.csr_matrix, docs_t: sparse.csc_matrix, alive: np.ndarray, k: int) -> list[set[int]]:
    """Exact top k over the live documents. Documents with score 0 are not results."""
    n_docs = docs_t.shape[1]
    chunk = max(1, 50_000_000 // n_docs)
    truth: list[set[int]] = []
    for start in range(0, queries.shape[0], chunk):
        scores = (queries[start : start + chunk] @ docs_t).toarray()
        scores[:, ~alive] = 0.0
        kk = min(k, n_docs)
        top = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        for row, ids in enumerate(top):
            truth.append({int(i) for i in ids if scores[row, i] > 0.0})
    return truth


# ---------- server ----------


class Server:
    def __init__(self, binary: Path, workdir: Path, vocab_size: int, port: int, timeout: float):
        self.binary, self.workdir, self.vocab_size = binary, workdir, vocab_size
        self.port, self.timeout = port, timeout
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{port}"

    def start(self) -> None:
        log = open(self.workdir / "server.log", "ab")
        self.proc = subprocess.Popen(
            [
                str(self.binary),
                "--index-dir", str(self.workdir),
                "--vocab-size", str(self.vocab_size),
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ],
            stdout=log,
            stderr=log,
        )
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited early, see {self.workdir / 'server.log'}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                time.sleep(0.2)
        raise RuntimeError("the server did not become healthy in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# ---------- requests ----------


async def insert_docs(client: httpx.AsyncClient, docs: sparse.csr_matrix, rows: range, first_id: int) -> None:
    """One at a time: the id of a document is the order it arrives in, so the order must be known."""
    for n, row in enumerate(rows):
        r = await client.post("/points", json=row_json(docs, row))
        if r.status_code != 201:
            raise RuntimeError(f"POST /points (document {row}) returned {r.status_code}: {r.text}")
        got, want = r.json()["external_id"], first_id + n
        if got != want:
            raise RuntimeError(f"POST /points (document {row}) returned the id {got}, expected {want}")


async def delete_ids(client: httpx.AsyncClient, ids: list[int], concurrency: int = 32) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with sem:
            r = await client.delete(f"/points/{i}")
            if r.status_code != 202:
                raise RuntimeError(f"DELETE /points/{i} returned {r.status_code}: {r.text}")

    await asyncio.gather(*(one(i) for i in ids))


async def run_queries(
    client: httpx.AsyncClient, queries: sparse.csr_matrix, args: argparse.Namespace
) -> tuple[list[list[int]], float]:
    """Returns the ids of each query, and the mean search time in milliseconds."""
    sem = asyncio.Semaphore(8)

    async def one(i: int) -> tuple[list[int], float]:
        async with sem:
            r = await client.post(
                "/search",
                json={
                    "vector": row_json(queries, i),
                    "top_k": args.top_k,
                    "ef": args.ef,
                    "patience": args.patience,
                    "max_query_dims": args.max_query_dims,
                    "rescore_factor": args.rescore_factor,
                },
            )
            r.raise_for_status()
            body = r.json()
            return body["ids"], body["search_time"]

    out = await asyncio.gather(*(one(i) for i in range(queries.shape[0])))
    return [ids for ids, _ in out], sum(t for _, t in out) / len(out)


def recall(results: list[list[int]], truth: list[set[int]]) -> float:
    scores = [len(set(r) & t) / len(t) for r, t in zip(results, truth) if t]
    return sum(scores) / len(scores) if scores else float("nan")


# ---------- the checks ----------


async def main_async(args: argparse.Namespace, workdir: Path, server: Server) -> bool:
    n_main = documents_in_index(args.index_dir)
    ratios = sorted(args.ratios)
    n_insert = min(MAX_DELTA, round(n_main * max(ratios) / 100))

    all_docs = read_seismic(args.documents, args.vocab_size, n_main + n_insert)
    if all_docs.shape[0] < n_main + n_insert:
        raise SystemExit(
            f"the index has {n_main} documents and {n_insert} more are needed to insert, but the documents "
            f"file has only {all_docs.shape[0]}. Build the index from fewer documents (--n-documents), "
            f"or lower --ratios"
        )
    queries = read_seismic(args.queries, args.vocab_size, args.n_queries)
    docs_t = all_docs.T.tocsc()
    print(f"{n_main} indexed documents, up to {n_insert} inserted, {queries.shape[0]} queries, "
          f"top_k={args.top_k}, ef={args.ef}\n")

    ok = True
    inserted = 0
    server.start()
    async with httpx.AsyncClient(base_url=server.url, timeout=60) as client:
        info = (await client.get("/info")).json()
        if info["num_documents"] != n_main or info["deleted_ratio"] != 0:
            raise SystemExit(f"unexpected /info on a fresh index: {info}")

        # ----- check 1: recall while the delta grows -----
        print("Check 1: recall while inserting")
        print(f"{'inserted %':>11} {'inserted':>9} {'recall':>8} {'from delta':>11} {'ms/query':>9} {'/info docs':>11}")
        results: list[list[int]] = []
        for pct in ratios:
            target = min(MAX_DELTA, round(n_main * pct / 100))
            await insert_docs(client, all_docs, range(n_main + inserted, n_main + target), n_main + inserted)
            inserted = target
            total = n_main + inserted

            alive = np.zeros(n_main + n_insert, dtype=bool)
            alive[:total] = True
            results, ms = await run_queries(client, queries, args)
            truth = exact_top_k(queries, docs_t, alive, args.top_k)

            out_of_range = sum(1 for r in results for i in r if i >= total)
            n_returned = sum(len(r) for r in results)
            from_delta = sum(1 for r in results for i in r if i >= n_main) / max(1, n_returned)
            n_docs_info = (await client.get("/info")).json()["num_documents"]
            row_ok = out_of_range == 0 and n_docs_info == total
            ok &= row_ok
            print(
                f"{pct:>11g} {inserted:>9} {recall(results, truth):>8.4f} {from_delta:>11.1%} {ms:>9.3f} {n_docs_info:>11}"
                + ("" if row_ok else "   <-- FAIL")
            )

        # ----- the endpoint itself -----
        print("\nEndpoint checks")
        bad_term = await client.post("/points", json={"indices": [args.vocab_size], "values": [1.0]})
        bad_len = await client.post("/points", json={"indices": [1, 2], "values": [1.0]})
        for name, got, want in (
            ("POST a term outside of the vocabulary", bad_term.status_code, 400),
            ("POST indices and values of different lengths", bad_len.status_code, 400),
        ):
            good = got == want
            ok &= good
            print(f"  {name}: {got} (expected {want}) {'ok' if good else 'FAIL'}")
        total = n_main + inserted
        out_of_range = (await client.delete(f"/points/{total}")).status_code
        good = out_of_range == 404
        ok &= good
        print(f"  DELETE out-of-range id: {out_of_range} (expected 404) {'ok' if good else 'FAIL'}")

        # ----- check 2: deleting from both the graph and the delta -----
        print("\nCheck 2: deletions across the graph and the delta")
        order = list(range(total))
        random.Random(args.seed).shuffle(order)
        to_delete = order[: round(total * args.delete_percent / 100)]
        deleted = set(to_delete)
        n_deleted_delta = sum(1 for i in to_delete if i >= n_main)
        await delete_ids(client, to_delete)

        alive = np.zeros(n_main + n_insert, dtype=bool)
        alive[:total] = True
        alive[list(deleted)] = False
        results, ms = await run_queries(client, queries, args)
        truth = exact_top_k(queries, docs_t, alive, args.top_k)
        leaked = sum(1 for r in results for i in r if i in deleted)
        info_ratio = (await client.get("/info")).json()["deleted_ratio"]
        ratio_ok = abs(info_ratio - len(deleted) / total) < 1e-9
        check_ok = leaked == 0 and ratio_ok
        ok &= check_ok
        print(f"  deleted {len(deleted)} documents ({n_deleted_delta} of them inserted)")
        print(f"  recall {recall(results, truth):.4f}, deleted ids in the results: {leaked}, "
              f"/info ratio {info_ratio:.5f} {'ok' if check_ok else 'FAIL'}")
        again = (await client.delete(f"/points/{to_delete[0]}")).status_code if to_delete else 204
        good = again == 204
        ok &= good
        print(f"  DELETE already-deleted id: {again} (expected 204) {'ok' if good else 'FAIL'}")

        info_before = (await client.get("/info")).json()

    # ----- check 3: persistence -----
    expected_delta = sum(8 + 8 * nnz_after_validation(all_docs, i) for i in range(n_main, n_main + inserted))
    delta_size = (workdir / DELTA_LOG).stat().st_size if (workdir / DELTA_LOG).exists() else 0
    delete_size = (workdir / DELETE_LOG).stat().st_size if (workdir / DELETE_LOG).exists() else 0
    print("\nCheck 3: persistence across a restart")
    for name, got, want in (
        (DELTA_LOG, delta_size, expected_delta),
        (DELETE_LOG, delete_size, 4 * len(deleted)),
    ):
        good = got == want
        ok &= good
        print(f"  {name}: {got} bytes (expected {want}) {'ok' if good else 'FAIL'}")

    server.stop()
    server.start()
    async with httpx.AsyncClient(base_url=server.url, timeout=60) as client:
        info_after = (await client.get("/info")).json()
        after, _ = await run_queries(client, queries, args)
        leaked = sum(1 for r in after for i in r if i in deleted)
        next_id = (await client.post("/points", json=row_json(all_docs, 0)))
        next_ok = next_id.status_code == 201 and next_id.json()["external_id"] == n_main + inserted

    for name, good in (
        ("/info is the same after the restart", info_after == info_before),
        ("no deleted id in the results", leaked == 0),
        ("search results are identical to before", after == results),
        (f"the next insert gets the id {n_main + inserted}", next_ok),
    ):
        ok &= good
        print(f"  {name}: {'ok' if good else 'FAIL'}")
    return ok


def main() -> None:
    args = parse_args()
    if not args.server_bin.exists():
        raise SystemExit(f"{args.server_bin} not found. Build it with: cargo build --release -p grip-server")

    with tempfile.TemporaryDirectory(prefix="grip-updates-") as tmp:
        workdir = Path(tmp)
        for name in INDEX_FILES:
            (workdir / name).symlink_to((args.index_dir / name).resolve())
        server = Server(args.server_bin.resolve(), workdir, args.vocab_size, args.port, args.startup_timeout)
        try:
            ok = asyncio.run(main_async(args, workdir, server))
        finally:
            server.stop()

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
