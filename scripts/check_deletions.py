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
"""Phase 1 checks for deletions, against a real grip-server.

Check 2: delete 0, 5, 10, 20, 30 and 50 percent of the documents (cumulative, at random),
         and after each step compare the search results with an exact search over the
         live documents. Reports recall@k and whether a deleted id ever shows up.
Check 3: restart the server and verify that the deletions survived.

The script never touches the index you pass in: it makes a work directory with symlinks to
index.bin, vectors.bin and sketches.bin, and the server writes delete.log there. Any
delete.log already in the source directory is ignored.

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
LOG_FILE = "delete.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--documents", type=Path, required=True, help="documents file (Seismic binary format)")
    p.add_argument("--queries", type=Path, required=True, help="queries file (Seismic binary format)")
    p.add_argument("--index-dir", type=Path, required=True, help="index built from the documents file")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--server-bin", type=Path, default=Path("target/release/grip-server"))
    p.add_argument("--port", type=int, default=7533)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-query-dims", type=int, default=2)
    p.add_argument("--rescore-factor", type=int, default=5)
    p.add_argument("--ratios", type=float, nargs="+", default=[0, 5, 10, 20, 30, 50], help="percent of documents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--startup-timeout", type=float, default=120.0)
    return p.parse_args()


# ---------- data ----------


def read_seismic(path: Path, vocab_size: int, limit: int | None = None) -> sparse.csr_matrix:
    """Seismic format: u32 count, then per vector a u32 length, `length` u32 ids, `length` f32 values."""
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<I", raw, 0)
    n = n if limit is None else min(n, limit)
    pos, indptr, idx, val = 4, [0], [], []
    for _ in range(n):
        (length,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        idx.append(np.frombuffer(raw, dtype="<u4", count=length, offset=pos))
        pos += 4 * length
        val.append(np.frombuffer(raw, dtype="<f4", count=length, offset=pos))
        pos += 4 * length
        indptr.append(indptr[-1] + length)
    return sparse.csr_matrix(
        (np.concatenate(val), np.concatenate(idx).astype(np.int64), np.array(indptr)),
        shape=(n, vocab_size),
    )


def documents_in_index(index_dir: Path) -> int:
    with open(index_dir / "vectors.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def query_json(queries: sparse.csr_matrix, i: int) -> dict:
    a, b = queries.indptr[i], queries.indptr[i + 1]
    return {"indices": queries.indices[a:b].tolist(), "values": queries.data[a:b].tolist()}


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


async def delete_ids(client: httpx.AsyncClient, ids: list[int], concurrency: int = 32) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with sem:
            r = await client.delete(f"/points/{i}")
            if r.status_code != 202:
                raise RuntimeError(f"DELETE /points/{i} returned {r.status_code}: {r.text}")

    await asyncio.gather(*(one(i) for i in ids))


async def run_queries(client: httpx.AsyncClient, queries: sparse.csr_matrix, args: argparse.Namespace) -> list[list[int]]:
    sem = asyncio.Semaphore(8)

    async def one(i: int) -> list[int]:
        async with sem:
            r = await client.post(
                "/search",
                json={
                    "vector": query_json(queries, i),
                    "top_k": args.top_k,
                    "ef": args.ef,
                    "patience": args.patience,
                    "max_query_dims": args.max_query_dims,
                    "rescore_factor": args.rescore_factor,
                },
            )
            r.raise_for_status()
            return r.json()["ids"]

    return await asyncio.gather(*(one(i) for i in range(queries.shape[0])))


def recall(results: list[list[int]], truth: list[set[int]]) -> float:
    scores = [len(set(r) & t) / len(t) for r, t in zip(results, truth) if t]
    return sum(scores) / len(scores) if scores else float("nan")


# ---------- the checks ----------


async def main_async(args: argparse.Namespace, workdir: Path, server: Server) -> bool:
    n_docs = documents_in_index(args.index_dir)
    docs = read_seismic(args.documents, args.vocab_size, n_docs)
    if docs.shape[0] != n_docs:
        raise SystemExit(f"the index has {n_docs} documents but the documents file has {docs.shape[0]}")
    queries = read_seismic(args.queries, args.vocab_size, args.n_queries)
    docs_t = docs.T.tocsc()
    print(f"{n_docs} documents, {queries.shape[0]} queries, top_k={args.top_k}, ef={args.ef}\n")

    order = list(range(n_docs))
    random.Random(args.seed).shuffle(order)
    ok = True

    server.start()
    async with httpx.AsyncClient(base_url=server.url, timeout=30) as client:
        info = (await client.get("/info")).json()
        if info["num_documents"] != n_docs or info["deleted_ratio"] != 0:
            raise SystemExit(f"unexpected /info on a fresh index: {info}")

        # ----- check 2: recall against the deleted ratio -----
        print("Check 2: recall against the deleted ratio")
        print(f"{'deleted %':>10} {'deleted':>9} {'recall':>8} {'deleted in results':>19} {'/info ratio':>12}")
        deleted_count = 0
        results: list[list[int]] = []
        for pct in sorted(args.ratios):
            target = round(n_docs * pct / 100)
            await delete_ids(client, order[deleted_count:target])
            deleted_count = target
            deleted = set(order[:deleted_count])

            alive = np.ones(n_docs, dtype=bool)
            alive[list(deleted)] = False
            results = await run_queries(client, queries, args)
            truth = exact_top_k(queries, docs_t, alive, args.top_k)

            leaked = sum(1 for r in results for i in r if i in deleted)
            ratio = (await client.get("/info")).json()["deleted_ratio"]
            ratio_ok = abs(ratio - deleted_count / n_docs) < 1e-9
            row_ok = leaked == 0 and ratio_ok
            ok &= row_ok
            print(
                f"{pct:>10g} {deleted_count:>9} {recall(results, truth):>8.4f} {leaked:>19} {ratio:>12.5f}"
                + ("" if row_ok else "   <-- FAIL")
            )

        # ----- the endpoint itself -----
        print("\nEndpoint checks")
        out_of_range = (await client.delete(f"/points/{n_docs}")).status_code
        again = (await client.delete(f"/points/{order[0]}")).status_code if deleted_count else 204
        for name, got, want in (("DELETE out-of-range id", out_of_range, 404), ("DELETE already-deleted id", again, 204)):
            good = got == want
            ok &= good
            print(f"  {name}: {got} (expected {want}) {'ok' if good else 'FAIL'}")

        info_before = (await client.get("/info")).json()

    log_size = (workdir / LOG_FILE).stat().st_size if (workdir / LOG_FILE).exists() else 0
    print(f"\nCheck 3: persistence across a restart\n  delete.log: {log_size} bytes (expected {4 * deleted_count})")
    ok &= log_size == 4 * deleted_count

    server.stop()
    server.start()
    deleted = set(order[:deleted_count])
    async with httpx.AsyncClient(base_url=server.url, timeout=30) as client:
        info_after = (await client.get("/info")).json()
        same_info = info_after == info_before
        after = await run_queries(client, queries, args)
        leaked = sum(1 for r in after for i in r if i in deleted)
        same_results = after == results
        again = (await client.delete(f"/points/{order[0]}")).status_code if deleted_count else 204

    for name, good in (
        ("/info is the same after the restart", same_info),
        ("no deleted id in the results", leaked == 0),
        ("search results are identical to before", same_results),
        ("an already-deleted id is still no-op (204)", again == 204),
    ):
        ok &= good
        print(f"  {name}: {'ok' if good else 'FAIL'}")
    return ok


def main() -> None:
    args = parse_args()
    if not args.server_bin.exists():
        raise SystemExit(f"{args.server_bin} not found. Build it with: cargo build --release -p grip-server")

    with tempfile.TemporaryDirectory(prefix="grip-deletions-") as tmp:
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
