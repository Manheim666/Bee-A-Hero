#!/usr/bin/env bash
# One-command launcher for macOS / Linux.
# Sets up backend venv, installs deps, seeds the DB, starts backend + frontend,
# and opens the browser. Re-run any time — it skips work that's already done.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
LOG_DIR="$ROOT_DIR/.logs"
mkdir -p "$LOG_DIR"

# --- prerequisites -----------------------------------------------------------
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "❌ '$1' is not installed. Please install it and try again." >&2
    exit 1
  }
}
need python3
need node
need npm

# --- backend -----------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "▶ Creating Python venv..."
  python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Install (or top up) deps only if requirements.txt is newer than the marker.
REQ_MARKER="$VENV_DIR/.deps-installed"
if [ ! -f "$REQ_MARKER" ] || [ "$BACKEND_DIR/requirements.txt" -nt "$REQ_MARKER" ]; then
  echo "▶ Installing backend deps (this may take a minute the first time)..."
  "$PIP" install --upgrade pip >/dev/null
  "$PIP" install -q -r "$BACKEND_DIR/requirements.txt"
  touch "$REQ_MARKER"
fi

if [ ! -f "$BACKEND_DIR/.env" ]; then
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
  echo "▶ Created backend/.env from example (set ANTHROPIC_API_KEY there for real AI)."
fi

if [ ! -f "$BACKEND_DIR/bee.db" ]; then
  echo "▶ Seeding demo user + sample video..."
  (cd "$BACKEND_DIR" && "$PYTHON" -m seed)
fi

# Stop any previous instances so re-running is safe.
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

echo "▶ Starting backend on http://localhost:8000 ..."
(cd "$BACKEND_DIR" && "$VENV_DIR/bin/uvicorn" app.main:app --port 8000 \
  > "$LOG_DIR/backend.log" 2>&1) &
BACKEND_PID=$!

# --- frontend ----------------------------------------------------------------
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "▶ Installing frontend deps (first run only)..."
  (cd "$FRONTEND_DIR" && npm install)
fi

echo "▶ Starting frontend on http://localhost:5173 ..."
(cd "$FRONTEND_DIR" && npm run dev > "$LOG_DIR/frontend.log" 2>&1) &
FRONTEND_PID=$!

cleanup() {
  echo ""
  echo "▶ Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Wait for backend to answer, then open browser.
echo -n "▶ Waiting for backend"
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 0.5
done

# Wait a bit more so Vite is definitely ready, then open.
sleep 2
if command -v open >/dev/null 2>&1; then
  open http://localhost:5173/
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://localhost:5173/ >/dev/null 2>&1 || true
fi

echo ""
echo "───────────────────────────────────────────────"
echo "  🐝 Bee-A-Hero is running"
echo "     App:  http://localhost:5173"
echo "     API:  http://localhost:8000/docs"
echo "     Login: demo@bee.dev / beehero123"
echo "  Logs: $LOG_DIR/backend.log, frontend.log"
echo "  Press Ctrl+C to stop both."
echo "───────────────────────────────────────────────"

wait
