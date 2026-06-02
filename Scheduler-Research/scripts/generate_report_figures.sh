#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm \
  -v "$PWD/scripts:/workspace/scripts" \
  -v "$PWD/reports:/workspace/reports" \
  dashboard \
  python3 scripts/generate_report_figures.py \
    --db /workspace/data/metrics.db \
    --output-dir /workspace/reports/figures \
    "$@"
