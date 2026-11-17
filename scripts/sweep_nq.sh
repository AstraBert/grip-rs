#!/usr/bin/env bash
# Small sweep for the NQ data: the server defaults, then a few variations of
# `ef` and `max_query_dims`. Quality is measured against the exact ground truth.
#
# Usage: ./scripts/sweep_nq.sh <data-dir> [extra bench_server.py args]
# Example: ./scripts/sweep_nq.sh data/nq/splade --url http://localhost:7433
#
# <data-dir> must hold `queries.bin` and `groundtruth.tsv`
# (the output of `prepare_nq.py` for one model).
set -euo pipefail
cd "$(dirname "$0")/.."

data_dir="${1:?usage: sweep_nq.sh <data-dir> [extra args]}"
shift

# Each entry holds the flags passed to the benchmark. The first one is empty,
# so the server defaults are used.
configs=(
  ""
  "--ef 50"
  "--ef 200"
  "--max-query-dims 10"
  "--max-query-dims 20"
  "--ef 50 --max-query-dims 10"
  "--ef 200 --max-query-dims 20"
)

for config in "${configs[@]}"; do
  echo "=== ${config:-defaults} ==="
  # `$config` is left unquoted on purpose, so it splits into separate flags.
  # shellcheck disable=SC2086
  ./scripts/bench_server.py \
    --queries "$data_dir/queries.bin" \
    --groundtruth "$data_dir/groundtruth.tsv" \
    --n-queries 10000 --levels 1 4 8 16 \
    $config "$@"
done
