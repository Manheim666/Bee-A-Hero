#!/usr/bin/env bash
# Install the CV-stage dependencies (torch, ultralytics/YOLO26, tracker) into .venv.
# Prereq: bash scripts/setup_env.sh  (creates .venv with the data-stage deps).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv — run: bash scripts/setup_env.sh first" >&2
  exit 1
fi

.venv/bin/python -m pip install -r src/cv_engine/requirements-cv.txt
.venv/bin/python - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("ultralytics", ultralytics.__version__)
PY
echo "CV environment ready."
