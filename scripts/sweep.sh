#!/usr/bin/env bash
# Runs the benchmark with 500 queries for each combination of
# `ef`, `rescore_factor`, `patience` and `max_query_dims`.
set -euo pipefail
cd "$(dirname "$0")/.."

for ef in 50 100 200; do
  for rescore_factor in 3 5 8; do
    for patience in 10 20 40; do
      for max_query_dims in 2 10 20; do
        echo "=== ef=$ef rescore_factor=$rescore_factor patience=$patience max_query_dims=$max_query_dims ==="
        ./scripts/bench_server.py --n-queries 500 --levels 1 4 8 16 \
          --ef "$ef" --rescore-factor "$rescore_factor" \
          --patience "$patience" --max-query-dims "$max_query_dims"
      done
    done
  done
done
