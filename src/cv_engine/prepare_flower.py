"""Build a single-class ``flower`` YOLO detection dataset from the Flower
*classification* folders (data/raw/Flower).

The source has no boxes (one centered flower per image), so boxes are generated
with OpenCV GrabCut foreground segmentation — reusing the proven ``auto_bbox``
from ``src/data_pipeline/flower/make_detection_dataset.py`` (no duplicated code).
All boxes are class 0 = ``flower`` (per the CV plan). Approximate/machine-made —
spot-check before trusting as ground truth.

Output (git-ignored, under data/interim/):
    flower_det/images/{train,val,test}/<Class>_<file>
    flower_det/labels/{train,val,test}/<Class>_<file>.txt   # "0 cx cy w h"
    flower_det/data.yaml                                     # Ultralytics config

CLI:  python -m src.cv_engine.prepare_flower
"""
from __future__ import annotations

import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from src import config as C
from src.data_pipeline.flower.make_detection_dataset import auto_bbox

_MP = multiprocessing.get_context("fork")
SPLIT_MAP = {"Training Data": "train", "Validation Data": "val", "Testing Data": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT = C.INTERIM_DIR / "flower_det"


def _build_tasks() -> list[tuple[str, str, str]]:
    tasks, seen = [], set()
    for split_dir, split in SPLIT_MAP.items():
        sdir = C.FLOWER_DIR / split_dir
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                stem, dst, i = f"{cls_dir.name}_{f.stem}", f"{cls_dir.name}_{f.name}", 1
                while (split, Path(dst).stem) in seen:
                    dst = f"{stem}_{i}{f.suffix}"; i += 1
                seen.add((split, Path(dst).stem))
                tasks.append((str(f), split, dst))
    return tasks


def _process(task):
    src, split, dst = task
    img = cv2.imread(src)
    if img is None:
        return (split, "read_error")
    cx, cy, nw, nh, fb = auto_bbox(img)
    shutil.copy2(src, OUT / "images" / split / dst)
    (OUT / "labels" / split / f"{Path(dst).stem}.txt").write_text(
        f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    return (split, "fallback" if fb else "grabcut")


def _write_yaml() -> None:
    (OUT / "data.yaml").write_text(
        f"# single-class flower detector (GrabCut auto-boxes)\n"
        f"path: {OUT.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: [flower]\n")


def run(workers: int | None = None) -> dict:
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks()
    from collections import Counter
    stats: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP) as ex:
        for split, kind in ex.map(_process, tasks, chunksize=32):
            stats[f"{split}_{kind}"] += 1
            stats[split] += 1
    _write_yaml()
    summary = {"total": len(tasks), "counts": dict(stats), "data_yaml": str(OUT / "data.yaml")}
    return summary


if __name__ == "__main__":
    import json, os
    print(json.dumps(run(workers=os.cpu_count()), indent=2))
