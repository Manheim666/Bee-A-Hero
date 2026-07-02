"""Fine-tune a YOLO26 detector (flowers or insects) on the RTX 3050 6GB.

Thin, reusable wrapper over Ultralytics. Runs are written under
``data/interim/cv_runs/<name>/`` (git-ignored); best weights at
``.../weights/best.pt``.

CLI:
    python -m src.cv_engine.train --data data/interim/flower_det/data.yaml \
        --name flower_yolo26n --epochs 60 --batch 16
"""
from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

# Python 3.14 defaults to the "forkserver" start method, which makes Ultralytics'
# DataLoader workers re-import this module and spawn a runaway process swarm with
# no real training. Force "fork" so workers inherit state cleanly.
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

from ultralytics import YOLO

from src import config as C

RUNS_DIR = C.INTERIM_DIR / "cv_runs"
WEIGHTS_DIR = C.INTERIM_DIR / "weights"


def train(data: str, name: str, model: str = "yolo26n.pt", epochs: int = 60,
          imgsz: int = 640, batch: int = 16, device: int | str = 0,
          patience: int = 15, resume: bool = False, scale: float = 0.5) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # prefer a local pretrained copy if present (avoids re-download to cwd)
    local = WEIGHTS_DIR / model
    yolo = YOLO(str(local) if local.exists() else model)
    results = yolo.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=str(RUNS_DIR), name=name, seed=C.SEED, deterministic=True,
        amp=True, patience=patience, exist_ok=True, resume=resume, verbose=True,
        # scale jitter: larger range makes the detector see objects at more
        # scales (incl. small) -> tighter boxes on small/video insects.
        scale=scale,
    )
    best = RUNS_DIR / name / "weights" / "best.pt"
    # report validation mAP
    metrics = yolo.val(data=data, imgsz=imgsz, device=device, verbose=False)
    out = {
        "name": name, "best_weights": str(best),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--name", required=True, help="run name")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5, help="scale-jitter gain (aug)")
    args = ap.parse_args()
    import json
    print(json.dumps(train(args.data, args.name, args.model, args.epochs,
                           args.imgsz, args.batch, patience=args.patience,
                           resume=args.resume, scale=args.scale), indent=2))


if __name__ == "__main__":
    main()
