#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==2.4.0",
#   "ijson==3.3.0",
#   "numpy==2.5.4",
#   "pyarrow==26.0.0",
#   "scipy==1.18.1",
# ]
# ///
import argparse
import json
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

import ijson
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from scipy import sparse

MODELS = ("splade", "slideon", "sparseup", "granite")
DEFAULT_STORAGE = Path(__file__).resolve().parents[3] / "qdrant-seismic-exp" / "storage"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "nq"
SPLADE_ARCHIVE = "documents.tar.gz"
SPLADE_VOCAB = 30_522  # BERT wordpiece vocabulary
UINT16_VOCAB = 65_536  # indices are uint16, so no term id can be above this
SLIDEON_REPO = "lightonai/SlideOn-sparse-vectors"
SLIDEON_CORPUS_DIR = "beir-v1.0.0-nq/corpus"
DEFAULT_MAX_DOCS = 2_000_000
TOP = 10
JSON_CHUNK_DOCS = 50_000  # documents converted to a matrix at once, for the JSON source
READ_BATCH_SIZE = 20_000  # rows read at once, for the parquet sources
GROUND_TRUTH_CELLS = 64_000_000  # size of the score block: queries x documents


def progress(*args: object, **kwargs: Any) -> None:
    """Progress goes to stderr, so stdout holds only the report."""
    print(*args, file=sys.stderr, **kwargs)


class Chunk(NamedTuple):
    ids: list[str]
    matrix: sparse.csr_matrix


class Source(NamedTuple):
    chunks: Iterator[Chunk]
    vocab_size: int
    limit: int  # documents to read, queries included


class RawVector(NamedTuple):
    doc_id: str
    coordinates: list[int]
    values: list[float]


@dataclass(frozen=True)
class Config:
    models: list[str]
    storage_dir: Path
    output_root: Path
    max_docs: int | None
    num_queries: int
    seed: int
    skip_ground_truth: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Convert NQ sparse vectors to the Seismic binary format of grip-rs."
    )
    parser.add_argument("models", nargs="+", choices=MODELS)
    parser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="each model is written to <output-root>/<model>",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=f"default: {DEFAULT_MAX_DOCS} for splade and slideon, all documents otherwise",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10_000,
        help="number of docs sampled at random and held out of the corpus, to be used as queries",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the sampling of the query docs"
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="do not compute the exact top 10 of each query",
    )
    args = parser.parse_args()
    return Config(
        args.models,
        args.storage_dir,
        args.output_root,
        args.max_docs,
        args.num_queries,
        args.seed,
        args.skip_ground_truth,
    )


# ---- reading the sources ----


def stream_vectors(archive: Path) -> Iterator[RawVector]:
    """Yield vectors one by one from every JSON file in a tar.gz, in archive order."""
    # "r|gz" reads the archive as a stream: no seeking, no full extraction
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".json"):
                continue
            progress(f"Reading {member.name}")
            file = tar.extractfile(member)
            assert file is not None
            for item in ijson.items(file, "vectors.item", use_float=True):
                yield RawVector(item["id"], item["coordinates"], item["values"])


def to_csr(vectors: list[RawVector], vocab_size: int) -> sparse.csr_matrix:
    """Pack vectors into a CSR matrix: one row per vector, one column per term."""
    lengths = np.array([len(v.coordinates) for v in vectors])
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    total = int(indptr[-1])
    indices = np.fromiter(
        chain.from_iterable(v.coordinates for v in vectors), np.int32, count=total
    )
    data = np.fromiter(
        chain.from_iterable(v.values for v in vectors), np.float32, count=total
    )
    return sparse.csr_matrix((data, indices, indptr), shape=(len(vectors), vocab_size))


def clean(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Sort the terms of each row, merge repeated terms, and drop zero weights."""
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def splade_chunks(archive: Path, vocab_size: int) -> Iterator[Chunk]:
    pending: list[RawVector] = []
    for vector in stream_vectors(archive):
        pending.append(vector)
        if len(pending) == JSON_CHUNK_DOCS:
            yield Chunk([v.doc_id for v in pending], clean(to_csr(pending, vocab_size)))
            pending = []
    if pending:
        yield Chunk([v.doc_id for v in pending], clean(to_csr(pending, vocab_size)))


def batch_to_csr(batch: pa.RecordBatch, vocab_size: int) -> sparse.csr_matrix:
    """Rebuild sparse vectors from the `indices` and `weights` list columns."""
    indices_col = batch.column("indices")
    weights_col = batch.column("weights")
    # Arrow stores a list column as one flat array plus `offsets`. That is the
    # CSR `indptr`, once shifted to start at 0.
    offsets = indices_col.offsets.to_numpy()
    indptr = offsets - offsets[0]
    indices = indices_col.flatten().to_numpy().astype(np.int32)
    data = weights_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    assert len(indices) == len(data) == indptr[-1], "indices and weights do not match"
    return sparse.csr_matrix((data, indices, indptr), shape=(len(batch), vocab_size))


def parquet_chunks(paths: Iterator[Path], vocab_size: int) -> Iterator[Chunk]:
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_SIZE):
            matrix = clean(batch_to_csr(batch, vocab_size))
            yield Chunk(batch.column("id").to_pylist(), matrix)


def slideon_paths() -> Iterator[Path]:
    """The corpus parts of SlideOn, downloaded one at a time (or found in the cache)."""
    files = HfApi().list_repo_files(SLIDEON_REPO, repo_type="dataset")
    prefix = f"{SLIDEON_CORPUS_DIR}/"
    for name in sorted(
        f for f in files if f.startswith(prefix) and f.endswith(".parquet")
    ):
        progress(f"Fetching {name}")
        yield Path(hf_hub_download(SLIDEON_REPO, name, repo_type="dataset"))


def declared_vocab(folder: Path) -> int:
    """The vocabulary size written by the encoder, or the uint16 bound."""
    meta = folder / "meta.json"
    if not meta.exists():
        return UINT16_VOCAB
    return json.loads(meta.read_text()).get("vocab_size", UINT16_VOCAB)


def open_source(model: str, config: Config) -> Source:
    if model == "splade":
        archive = config.storage_dir / SPLADE_ARCHIVE
        limit = config.max_docs or DEFAULT_MAX_DOCS
        return Source(splade_chunks(archive, SPLADE_VOCAB), SPLADE_VOCAB, limit)

    folder = config.storage_dir / model
    vocab_size = declared_vocab(folder)
    if folder.exists():
        paths = sorted(folder.glob("docs-*.parquet"))
        available = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        limit = min(config.max_docs or available, available)
        return Source(parquet_chunks(iter(paths), vocab_size), vocab_size, limit)
    if model == "slideon":
        limit = config.max_docs or DEFAULT_MAX_DOCS
        return Source(parquet_chunks(slideon_paths(), vocab_size), vocab_size, limit)
    raise SystemExit(f"No folder {folder} for {model}")


def limit_chunks(chunks: Iterator[Chunk], limit: int) -> Iterator[Chunk]:
    """Stop after `limit` rows. The last chunk is cut by rows."""
    seen = 0
    for ids, matrix in chunks:
        if seen + len(ids) >= limit:
            keep = limit - seen
            yield Chunk(ids[:keep], matrix[:keep])
            return
        seen += len(ids)
        yield Chunk(ids, matrix)


# ---- the Seismic binary format ----
# A u32 count, then for each vector: a u32 length, the term ids as u32, and the
# weights as f32. All little endian. Everything is a 32-bit word, so a file is
# one array of u32 and a vector is a slice of it.


def encode_rows(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the words of all rows, and the word offset where each row starts."""
    indptr = matrix.indptr.astype(np.int64)
    lengths = np.diff(indptr)
    row_words = 1 + 2 * lengths
    starts = np.concatenate([[0], np.cumsum(row_words)[:-1]])
    words = np.empty(int(row_words.sum()), dtype="<u4")
    words[starts] = lengths
    # for each non-zero value: its row, and its place inside that row
    row_of = np.repeat(np.arange(len(lengths)), lengths)
    place = np.arange(matrix.nnz) - indptr[:-1][row_of]
    index_at = starts[row_of] + 1 + place
    words[index_at] = matrix.indices
    weights = np.ascontiguousarray(matrix.data, dtype="<f4")
    words[index_at + lengths[row_of]] = weights.view("<u4")
    return words, starts


def decode_rows(
    words: np.ndarray, starts: np.ndarray, vocab_size: int
) -> sparse.csr_matrix:
    """Read the rows that start at the given word offsets into a CSR matrix."""
    lengths = words[starts].astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    row_of = np.repeat(np.arange(len(starts)), lengths)
    place = np.arange(indptr[-1]) - indptr[:-1][row_of]
    index_at = starts[row_of] + 1 + place
    indices = words[index_at].astype(np.int32)
    data = words[index_at + lengths[row_of]].view("<f4")
    return sparse.csr_matrix((data, indices, indptr), shape=(len(starts), vocab_size))


def write_matrix(path: Path, matrix: sparse.csr_matrix) -> None:
    words, _ = encode_rows(matrix)
    with path.open("wb") as file:
        file.write(np.uint32(matrix.shape[0]).tobytes())
        words.tofile(file)


# ---- the conversion ----


class Converted(NamedTuple):
    doc_ids: list[str]
    starts: np.ndarray  # word offset of every document, then the end of the file
    max_term: int
    query_ids: list[str]
    queries: sparse.csr_matrix


def split_chunk(
    chunk: Chunk, first: int, query_rows: np.ndarray
) -> tuple[Chunk, Chunk]:
    """Cut a chunk in two: the documents of the corpus, and the rows held out as queries."""
    rows = np.arange(first, first + len(chunk.ids))
    is_query = np.isin(rows, query_rows)
    docs, queries = np.flatnonzero(~is_query), np.flatnonzero(is_query)
    return (
        Chunk([chunk.ids[i] for i in docs], chunk.matrix[docs]),
        Chunk([chunk.ids[i] for i in queries], chunk.matrix[queries]),
    )


def convert_documents(source: Source, path: Path, query_rows: np.ndarray) -> Converted:
    doc_ids: list[str] = []
    starts: list[np.ndarray] = []
    query_chunks: list[Chunk] = []
    next_word = 1  # word 0 is the count
    max_term = -1
    seen = 0
    with path.open("wb") as file:
        file.write(np.uint32(0).tobytes())  # the count, written at the end
        for chunk in limit_chunks(source.chunks, source.limit):
            if chunk.matrix.nnz:
                max_term = max(max_term, int(chunk.matrix.indices.max()))
            docs, queries = split_chunk(chunk, seen, query_rows)
            seen += len(chunk.ids)
            query_chunks.append(queries)
            if docs.ids:
                words, row_starts = encode_rows(docs.matrix)
                words.tofile(file)
                starts.append(row_starts + next_word)
                next_word += len(words)
                doc_ids.extend(docs.ids)
            progress(f"\r  {seen} documents read", end="", flush=True)
        file.seek(0)
        file.write(np.uint32(len(doc_ids)).tobytes())
    progress()
    if seen != source.limit:
        raise SystemExit(
            f"The source has {seen} documents, not {source.limit}. Run again with --max-docs {seen}"
        )
    starts.append(np.array([next_word]))
    held_out = [chunk for chunk in query_chunks if chunk.ids]
    return Converted(
        doc_ids,
        np.concatenate(starts),
        max_term,
        [query_id for chunk in held_out for query_id in chunk.ids],
        sparse.vstack([chunk.matrix for chunk in held_out], format="csr"),
    )


def sample_rows(n_docs: int, num_queries: int, seed: int) -> np.ndarray:
    """Pick `num_queries` different rows at random, in file order. The same seed gives the same rows."""
    if num_queries >= n_docs:
        raise SystemExit(f"Cannot hold out {num_queries} queries from {n_docs} docs")
    rows = np.random.default_rng(seed).choice(n_docs, size=num_queries, replace=False)
    return np.sort(rows)


def exact_top(
    words: np.ndarray,
    starts: np.ndarray,
    queries: sparse.csr_matrix,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The TOP highest dot products of each query, best first, by brute force.

    The documents are read in blocks, so the corpus never has to fit in memory.
    """
    n_docs = len(starts) - 1
    n_queries = queries.shape[0]
    keep = TOP
    best_ids = np.full((n_queries, keep), -1, dtype=np.int64)
    best_scores = np.full((n_queries, keep), -np.inf, dtype=np.float32)
    block = int(np.clip(GROUND_TRUTH_CELLS // n_queries, 1_000, 100_000))
    for first in range(0, n_docs, block):
        last = min(first + block, n_docs)
        docs = decode_rows(words, starts[first:last], vocab_size)
        scores = (queries @ docs.T).toarray().astype(np.float32)
        take = min(keep, scores.shape[1])
        # argpartition: the first `take` entries are the largest, in no order
        part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        candidates = np.concatenate(
            [best_scores, np.take_along_axis(scores, part, axis=1)], axis=1
        )
        candidate_ids = np.concatenate([best_ids, part + first], axis=1)
        order = np.argsort(-candidates, axis=1)[:, :keep]
        best_scores = np.take_along_axis(candidates, order, axis=1)
        best_ids = np.take_along_axis(candidate_ids, order, axis=1)
        progress(f"\r  {last}/{n_docs} documents", end="", flush=True)
    progress()
    return best_ids, best_scores


def write_ground_truth(path: Path, ids: np.ndarray, scores: np.ndarray) -> None:
    """One line for each result: query, document, rank, score."""
    with path.open("w") as file:
        for query, (query_ids, query_scores) in enumerate(
            zip(ids, scores, strict=True)
        ):
            for rank, (doc, score) in enumerate(zip(query_ids, query_scores), start=1):
                file.write(f"{query}\t{doc}\t{rank}\t{score:.6f}\n")


def write_lists(path: Path, converted: Converted, query_rows: np.ndarray) -> None:
    """The original ids: the line of doc_ids.txt is the position in documents.bin."""
    ids_file = path / "doc_ids.txt"
    ids_file.write_text("".join(f"{doc_id}\n" for doc_id in converted.doc_ids))
    lines = (
        f"{query}\t{row}\t{query_id}\n"
        for query, (row, query_id) in enumerate(
            zip(query_rows, converted.query_ids, strict=True)
        )
    )
    (path / "queries.tsv").write_text("".join(lines))


def nnz_stats(words: np.ndarray, starts: np.ndarray) -> dict[str, float]:
    lengths = words[starts[:-1]]
    return {
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
        "empty": int((lengths == 0).sum()),
    }


def prepare(model: str, config: Config) -> None:
    out = config.output_root / model
    out.mkdir(parents=True, exist_ok=True)
    source = open_source(model, config)

    query_rows = sample_rows(source.limit, config.num_queries, config.seed)
    progress(f"[{model}] converting documents, holding out {len(query_rows)} queries")
    converted = convert_documents(source, out / "documents.bin", query_rows)
    if converted.max_term >= source.vocab_size:
        raise SystemExit(
            f"Term id {converted.max_term} is not below the vocabulary size {source.vocab_size}"
        )
    n_docs = len(converted.doc_ids)
    words = np.memmap(out / "documents.bin", dtype="<u4", mode="r")

    write_matrix(out / "queries.bin", converted.queries)
    write_lists(out, converted, query_rows)

    if not config.skip_ground_truth:
        progress(f"[{model}] computing the exact top {TOP}")
        ids, scores = exact_top(
            words, converted.starts, converted.queries, source.vocab_size
        )
        write_ground_truth(out / "groundtruth.tsv", ids, scores)

    stats = nnz_stats(words, converted.starts)
    meta = {
        "model": model,
        "documents": n_docs,
        "queries": config.num_queries,
        "source_documents": source.limit,
        "seed": config.seed,
        "vocab_size": source.vocab_size,
        "max_term_id": converted.max_term,
        "non_zero_terms_per_document": stats,
        "ground_truth": not config.skip_ground_truth,
        "note": (
            "the queries are held out of documents.bin. The id of a document is its "
            "position in documents.bin and its line in doc_ids.txt"
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{model}: {n_docs} documents, vocab {source.vocab_size}, {stats} -> {out}")


def main() -> None:
    config = parse_args()
    for model in config.models:
        prepare(model, config)


if __name__ == "__main__":
    main()
