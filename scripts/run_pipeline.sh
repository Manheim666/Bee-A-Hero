#!/usr/bin/env bash
# End-to-end iNaturalist data-preparation pipeline (idempotent, non-destructive).
# Prereq: bash scripts/setup_env.sh  (creates .venv with pinned deps)
#
# Usage:  bash scripts/run_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=".venv/bin/python"
JUP=".venv/bin/jupyter"
if [ ! -x "$PY" ]; then
  echo "No .venv found — run: bash scripts/setup_env.sh" >&2
  exit 1
fi

echo "==> [1/4] Balancing: Insecta filter + exact dedup + 70/15/15 split"
"$PY" -m src.data_pipeline.inaturalist_prep --apply

echo "==> [2/4] Labels: regenerate + integrity validation"
"$PY" -m src.data_pipeline.label_tools

echo "==> [3/4] Data-ready notebook (gate)"
"$JUP" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=beehero \
  --ExecutePreprocessor.timeout=1800 notebooks/00_data_ready.ipynb

echo "==> [4/4] EDA notebook"
"$JUP" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=beehero \
  --ExecutePreprocessor.timeout=1800 notebooks/01_eda.ipynb

echo
echo "Done. Report: data/interim/reports/data_ready_report.json"
echo "       EDA:    data/interim/eda/  |  Labels: data/interim/labels/"
