#!/usr/bin/env bash
# One-command launcher: venv + deps + server + browser.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

command -v python3 >/dev/null 2>&1 || { echo "❌ python3 required"; exit 1; }

if [ ! -d "$VENV_DIR" ]; then
  echo "▶ Creating venv..."
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

MARKER="$VENV_DIR/.deps-installed"
if [ ! -f "$MARKER" ] || [ "$ROOT_DIR/requirements.txt" -nt "$MARKER" ]; then
  echo "▶ Installing deps (first run may take a couple minutes for torch)..."
  "$PIP" install --upgrade pip >/dev/null
  "$PIP" install -q -r "$ROOT_DIR/requirements.txt"
  touch "$MARKER"
fi

if [ ! -f "$ROOT_DIR/.env" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "▶ Created .env — edit DROIDCAM_URL to point at your phone, then re-run."
  echo "  File: $ROOT_DIR/.env"
  exit 0
fi

echo "▶ Starting server on http://localhost:8001 ..."
(sleep 3 && (command -v open >/dev/null && open http://localhost:8001/ || \
             command -v xdg-open >/dev/null && xdg-open http://localhost:8001/ >/dev/null 2>&1 || true)) &

cd "$ROOT_DIR" && "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8001
