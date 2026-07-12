#!/usr/bin/env bash
# Create/refresh the project virtualenv and install pinned dependencies.
# Usage:  bash scripts/setup_env.sh   (optionally: PYTHON=python3.14 bash ...)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "Creating virtualenv at .venv ..."
  "$PY" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Register a named Jupyter kernel bound to this venv (used by run_pipeline.sh).
.venv/bin/python -m ipykernel install --user --name beehero \
  --display-name "Bee-A-Hero (.venv)"

echo
echo "Environment ready. Activate with:"
echo "    source .venv/bin/activate"
echo "Then run the pipeline notebook:"
echo "    jupyter nbconvert --to notebook --execute notebooks/00_data_ready.ipynb --inplace"
