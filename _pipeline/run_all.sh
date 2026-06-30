#!/usr/bin/env bash
# Orchestrator: wait for extraction -> run pipeline -> shutdown (only if clean).
# Detached/unattended. Does NOT depend on the chat session staying open.
set -u
ROOT="/c/Users/narim/Desktop/BEE_HERo"
cd "$ROOT" || exit 1
RLOG="_pipeline/run_all.log"
echo "[$(date)] run_all started" >> "$RLOG"

# --- 1. wait for extraction to finish (max 8h) ---
WAITED=0; MAX=28800
while true; do
  if grep -q "ALL EXTRACTION COMPLETE" extract.log 2>/dev/null; then
    echo "[$(date)] extraction complete" >> "$RLOG"; break
  fi
  if grep -q "ABORT" extract.log 2>/dev/null; then
    echo "[$(date)] extraction ABORTED (low disk) - NOT shutting down" >> "$RLOG"
    echo "EXTRACTION_ABORTED" > _pipeline/STATUS.txt
    exit 9
  fi
  if [ "$WAITED" -ge "$MAX" ]; then
    echo "[$(date)] timeout waiting for extraction - NOT shutting down" >> "$RLOG"
    echo "EXTRACTION_TIMEOUT" > _pipeline/STATUS.txt
    exit 8
  fi
  sleep 30; WAITED=$((WAITED+30))
done

# --- 2. run the pipeline ---
echo "[$(date)] launching pipeline.py" >> "$RLOG"
python _pipeline/pipeline.py >> "$RLOG" 2>&1
RC=$?
echo "[$(date)] pipeline exited rc=$RC" >> "$RLOG"

# --- 3. shutdown policy ---
# rc 0 = clean, rc 1 = non-fatal error (state saved) -> shutdown.
# rc 9 = low-disk safety abort -> stay ON so user can recover.
if [ "$RC" -eq 9 ]; then
  echo "[$(date)] low-disk abort -> leaving PC ON" >> "$RLOG"
  exit 9
fi
echo "[$(date)] scheduling shutdown in 90s" >> "$RLOG"
cmd //c "shutdown /s /t 90 /c \"BEE_HERo pipeline finished (rc=$RC) - shutting down\""
echo "[$(date)] shutdown scheduled" >> "$RLOG"
