#!/usr/bin/env bash
# Recreate the BEE_HERo .tar.gz archives from the intact extracted folders.
# Safety: build to a .partial file, integrity-test, then atomically rename.
# gzip -1 = fastest (the payload is already-compressed JPEGs, so higher levels
# waste CPU for ~no size gain).
set -u
ROOT="/c/Users/narim/Desktop/BEE_HERo"
LOG="$ROOT/_pipeline/recreate_archives.log"
STATUS="$ROOT/_pipeline/RECREATE_STATUS.txt"
cd "$ROOT" || { echo "cannot cd ROOT"; exit 1; }
: > "$LOG"; echo "STARTED" > "$STATUS"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# build_archive <src_folder> <out.tar.gz>
build_archive(){
  local src="$1" out="$2"
  if [ ! -d "$src" ]; then log "SKIP $out (missing folder $src)"; return 1; fi
  log "creating $out from $src/ ..."
  # -C ROOT so the archive stores paths as '<folder>/...' like the originals
  if tar -C "$ROOT" -I 'gzip -1' -cf "$out.partial" "$src"; then
    log "testing integrity of $out.partial ..."
    if gzip -t "$out.partial" 2>>"$LOG"; then
      mv -f "$out.partial" "$out"
      log "OK  $out  ($(du -h "$out" | cut -f1))"
    else
      log "FAIL integrity test for $out.partial (kept for inspection)"
      return 1
    fi
  else
    log "FAIL tar for $out"; rm -f "$out.partial"; return 1
  fi
}

rc=0

# 1) val.tar.gz -- restore the PRISTINE original from Downloads (best fidelity)
ORIG_VAL="/c/Users/narim/Downloads/val.tar.gz"
if [ -f "$ORIG_VAL" ]; then
  log "restoring val.tar.gz from pristine original in Downloads ..."
  cp -f "$ORIG_VAL" "$ROOT/val.tar.gz.partial" && \
  gzip -t "$ROOT/val.tar.gz.partial" 2>>"$LOG" && \
  mv -f "$ROOT/val.tar.gz.partial" "$ROOT/val.tar.gz" && \
  log "OK  val.tar.gz (copied original, $(du -h "$ROOT/val.tar.gz" | cut -f1))" || \
  { log "FAIL restoring original val; will rebuild from folder"; build_archive "val" "$ROOT/val.tar.gz" || rc=1; }
else
  log "no original val in Downloads; rebuilding from folder"
  build_archive "val" "$ROOT/val.tar.gz" || rc=1
fi

# 2) train_mini.tar.gz -- rebuild from filtered folder (raw original is gone)
build_archive "train_mini" "$ROOT/train_mini.tar.gz" || rc=1

# 3) public_test.tar.gz -- rebuild from intact 500k-image folder (slowest)
build_archive "public_test" "$ROOT/public_test.tar.gz" || rc=1

if [ "$rc" -eq 0 ]; then echo "COMPLETED_OK" > "$STATUS"; else echo "COMPLETED_WITH_ERRORS" > "$STATUS"; fi
log "=== done (status=$(cat "$STATUS")) ==="
log "final listing:"; ls -lh "$ROOT"/*.tar.gz 2>>"$LOG" | tee -a "$LOG"
