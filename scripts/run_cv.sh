#!/usr/bin/env bash
# End-to-end computer-vision pipeline: flower detector + insect detector +
# pollinator/non-pollinator classifier -> flower-visit counting on the test videos.
#
# Idempotent: any stage whose trained weights already exist is skipped, so the
# script can be re-run safely and resumes where it left off.
#
# Prereqs:
#   bash scripts/setup_env.sh && bash scripts/setup_cv.sh
#   Datasets under data/raw/ (see README "Getting the datasets").
#
# Usage:  bash scripts/run_cv.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python
RUNS=data/interim/cv_runs
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-convnext_large_mlp.laion2b_ft_augreg_inat21}"
EPOCHS="${EPOCHS:-60}"

[ -x "$PY" ] || { echo "No .venv — run scripts/setup_env.sh && scripts/setup_cv.sh"; exit 1; }
log() { echo -e "\n==> $*"; }

log "[1/5] Flower detection dataset + training"
[ -f data/interim/flower_det/data.yaml ] || "$PY" -m src.cv_engine.prepare_flower
[ -f "$RUNS/flower_yolo26n/weights/best.pt" ] || \
  "$PY" -m src.cv_engine.train --data data/interim/flower_det/data.yaml \
        --name flower_yolo26n --epochs "$EPOCHS"

log "[2/5] Insect detection + classifier datasets"
[ -d data/interim/insect_det1 ] || "$PY" - <<'PY'
from src.cv_engine.prepare_insect import run, export_single_class, export_classifier_crops
run(); export_single_class(); export_classifier_crops()
PY

log "[3/5] Insect detector (single-class, high mAP)"
[ -f "$RUNS/insect1cls_yolo26n/weights/best.pt" ] || \
  "$PY" -m src.cv_engine.train --data data/interim/insect_det1/data.yaml \
        --name insect1cls_yolo26n --epochs "$EPOCHS"

log "[4/5] Pollinator/non-pollinator classifier ($CLASSIFIER_MODEL)"
[ -f "$RUNS/insect_classifier/best.pt" ] || \
  "$PY" -m src.cv_engine.train_classifier --model "$CLASSIFIER_MODEL" --freeze --epochs 12

log "[5/5] Flower-visit counting on Test_Video"
shopt -s nullglob
for v in data/raw/Test_Video/*.mp4; do
  "$PY" -m src.cv_engine.visit_counter --video "$v" \
    --flower-weights   "$RUNS/flower_yolo26n/weights/best.pt" \
    --insect-weights   "$RUNS/insect1cls_yolo26n/weights/best.pt" \
    --classifier-weights "$RUNS/insect_classifier/best.pt" --save-video
done

echo -e "\nDone. Weights: $RUNS/*/weights/  |  Visits + annotated videos: $RUNS/visits/"
