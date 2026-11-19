# grip-rs

Rust implementation of [GRIP](https://arxiv.org/pdf/2610.30425), a variation of [PiPNN](https://arxiv.org/pdf/2602.21247) for approximate nearest neighbor search over **sparse vectors** (for example SPLADE-style learned sparse embeddings).

GRIP builds a proximity graph by partitioning the collection into overlapping groups, building small local graphs inside each group, and merging the edges with HashPrune and RobustPrune. At query time it walks the graph starting from entry points picked from the query's heaviest terms, then rescores the best candidates against full-precision vectors.

The repository is a Cargo workspace with two crates:

| Crate | What it is |
| --- | --- |
| [`grip-rs`](crates/grip-rs) | The library, plus an optional `grip-index` CLI to build indexes and run searches |
| [`grip-server`](crates/grip-server) | A small [Actix](https://actix.rs) HTTP server that loads a built index and serves search requests |

> **Note:** this is a research project. It is not published on crates.io and will not be.

## Installation

Install straight from Git with Cargo:

```bash
# the index builder / search CLI (binary: grip-index)
cargo install --git https://github.com/AstraBert/grip-rs grip-rs --features cli

# the HTTP search server (binary: grip-server)
cargo install --git https://github.com/AstraBert/grip-rs grip-server
```

To use the library in your own project, add it as a Git dependency:

```toml
[dependencies]
grip-rs = { git = "https://github.com/AstraBert/grip-rs" }
```

## Usage

### 1. Build an index

Documents and queries use the Seismic binary format (little endian): a `u32` vector count, then for each vector a `u32` length `n`, `n` `u32` term ids, and `n` `f32` values. A vector's id is its position in the file.

```bash
grip-index build \
  --documents documents.bin \
  --output-dir my-index \
  --vocab-size 30522
```

This writes `index.bin`, `vectors.bin` and `sketches.bin` to the output directory. Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-documents` | all | Index only the first N documents |
| `--vocab-size` | max term id + 1 | Size of the vocabulary |
| `--m` | 12 | Number of HashPrune hash bits (max 16) |
| `--leaf-k` | 8 | Neighbors each point keeps inside a leaf |
| `--alpha` | 1.0 | RobustPrune alpha |
| `--entrypoint-seeds` | 64 | Seed documents stored per term |
| `--sketch-size` | 80 | Components kept in each document's routing sketch |
| `--fanout-depth0` / `--fanout-depth1` | 10 / 3 | Partition fanout at depth 0 and 1 |
| `--refine` | off | Run a refinement pass (search each point in the graph and prune the results) |
| `--refinement-query-dims`, `--refinement-ef`, `--refinement-patience` | 2 / 100 / 20 | Refinement search settings (require `--refine`) |
| `--seed` | 42 | Random seed |

Build phase timings and peak memory are printed to stderr (the `performance-timing` feature, on by default).

### 2. Search from the CLI

```bash
grip-index search \
  --index-dir my-index \
  --queries queries.bin \
  --vocab-size 30522 \
  --top-k 10 \
  --recall          # optional: compare against exact search (slow, small collections only)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--top-k` | 10 | Results per query |
| `--ef` | 100 | Size of the result set during the graph walk (at least `top_k`) |
| `--patience` | 20 | Stop after this many expansions without improvement |
| `--max-query-dims` | 2 | Number of query terms used to pick entry points |
| `--rescore-factor` | 5 | Rescore `top_k x rescore_factor` candidates with full-precision vectors |
| `--n-queries` | all | Use only the first N queries |
| `--output` | none | Write results as `query<TAB>document<TAB>score` |

Note that `--vocab-size` is required for search, and must match the one used at build time.

### 3. Serve the index over HTTP

```bash
grip-server --index-dir my-index --vocab-size 30522 --port 7433
```

Options: `--host` (default `localhost`), `--port` (default `7433`), `--num-workers` (default `4`), `--log-level` (`error`, `warn`, `info`, `debug`).

Endpoints:

- `GET /health`: liveness check.
- `GET /info`: returns `vocab_size`, `num_documents` and `scale`.
- `POST /search`: runs a search.

```bash
curl -X POST http://localhost:7433/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": {"indices": [101, 2054, 3020], "values": [0.8, 1.2, 0.4]},
    "top_k": 10,
    "ef": 100
  }'
```

`vector` is required. `top_k` (10), `ef` (100), `patience` (20), `max_query_dims` (2) and `rescore_factor` (5) are optional, with the defaults shown in parentheses. `ef` is capped at 500 and `top_k` at 100. The response has the form:

```json
{"ids": [12, 7, 903], "scores": [14.2, 13.9, 12.1], "search_time": 0.6, "total_time": 0.7}
```

## Scripts

The `scripts/` directory holds helpers for benchmarking:

- `prepare_nq.py`: prepares Natural Questions data with several sparse models (`splade`, `slideon`, `sparseup`, `granite`) into the Seismic format.
- `test_server.py`, `bench_server.py`: smoke-test and benchmark a running `grip-server`.
- `sweep.sh`, `sweep_nq.sh`: parameter sweeps over `ef`, `rescore_factor`, `patience` and `max_query_dims`.

The Python scripts are [uv](https://docs.astral.sh/uv/) scripts: run them directly (e.g. `./scripts/bench_server.py --help`).

Benchmark outputs live in `results/`.
