"""Build a 2-class insect YOLO detection dataset: pollinator vs non_pollinator.

iNaturalist is a *classification* set (no boxes), so boxes are generated with
GrabCut (reusing ``auto_bbox``) on the centered organism. Classes come from the
Phase-4 manifest ``is_bee`` flag (bee families = pollinator). Non-pollinator is
heavily over-represented (147k vs 3.7k), so it is sub-sampled per split (seeded)
to balance 1:1 with pollinator.

Real bee boxes from the Roboflow COCO sets on the Desktop (iNat-sourced +
video-frame bees) are converted to YOLO and added to the **pollinator** class to
improve mAP and robustness on real video.

Classes: 0 = pollinator, 1 = non_pollinator.

Output (git-ignored): data/interim/insect_det/{images,labels}/{train,val,test}
                      + data.yaml

CLI:  python -m src.cv_engine.prepare_insect
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from src import config as C
from src.data_pipeline.flower.make_detection_dataset import auto_bbox

try:
    _MP = multiprocessing.get_context("fork")   # Linux/Mac
except ValueError:
    _MP = multiprocessing.get_context()          # Windows: default (spawn)
OUT = C.INTERIM_DIR / "insect_det"
NAMES = ["pollinator", "non_pollinator"]

# Real-box bee detection set (Roboflow COCO export).
# Only BEE.v8i (single iNat-sourced garden bees) — the Honey Bee hive export
# (~16 tiny dense bees/frame, hive-monitoring domain) tanks detection mAP and
# does not match the garden flower-visit videos, so it is excluded from detection.
# Portable location (git-ignored): data/raw/BEE_coco/ (see src/config.BEE_COCO_DIR).
# If absent, the pipeline still runs on iNaturalist alone (roboflow step skipped).
ROBOFLOW_SETS = [
    C.BEE_COCO_DIR,
]
_RF_SPLIT = {"train": "train", "valid": "val", "test": "test"}


# --------------------------------------------------------------------------- #
# iNaturalist -> GrabCut boxes, 2 classes, balanced
# --------------------------------------------------------------------------- #
def _select_inat_records() -> list[tuple[str, str, int, str]]:
    """Return (abs_path, split, class_id, dst_name); non_pollinator balanced 1:1."""
    rows = list(csv.DictReader(open(C.MANIFEST_DIR / "split_manifest.csv")))
    by_split_cls: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        cls = 0 if r["is_bee"] == "1" else 1
        by_split_cls[(r["split"], cls)].append(r)

    rng = random.Random(C.SEED)
    out = []
    for split in ("train", "val", "test"):
        poll = by_split_cls[(split, 0)]
        non = by_split_cls[(split, 1)]
        # balance: keep all pollinators, sample equal non-pollinators
        non = sorted(non, key=lambda r: r["path"])
        rng.shuffle(non)
        non = non[: len(poll)]
        for cls, recs in ((0, poll), (1, non)):
            for r in recs:
                p = C.REPO_ROOT / r["path"]
                dst = f"inat_{Path(r['path']).stem}.jpg"
                out.append((str(p), split, cls, dst))
    return out


def _process_inat(task):
    src, split, cls, dst = task
    img = cv2.imread(src)
    if img is None:
        return (split, "read_error")
    cx, cy, nw, nh, fb = auto_bbox(img)
    shutil.copy2(src, OUT / "images" / split / dst)
    (OUT / "labels" / split / f"{Path(dst).stem}.txt").write_text(
        f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    return (split, f"cls{cls}")


# --------------------------------------------------------------------------- #
# Roboflow COCO -> YOLO (all boxes -> class 0 pollinator)
# --------------------------------------------------------------------------- #
def _convert_roboflow() -> Counter:
    stats: Counter = Counter()
    for ds in ROBOFLOW_SETS:
        if not ds.is_dir():
            continue
        tag = ds.name.split(".")[0].replace(" ", "_")[:20]
        for rf_split, our_split in _RF_SPLIT.items():
            ann = ds / rf_split / "_annotations.coco.json"
            if not ann.exists():
                continue
            coco = json.load(open(ann))
            img_by_id = {im["id"]: im for im in coco["images"]}
            boxes_by_img: dict[int, list] = defaultdict(list)
            for a in coco["annotations"]:
                boxes_by_img[a["image_id"]].append(a["bbox"])  # [x,y,w,h] pixels
            for iid, im in img_by_id.items():
                src = ds / rf_split / im["file_name"]
                if not src.exists():
                    continue
                W, H = im["width"], im["height"]
                lines = []
                for x, y, w, h in boxes_by_img.get(iid, []):
                    cx, cy = (x + w / 2) / W, (y + h / 2) / H
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
                if not lines:                      # skip images with no bee box
                    continue
                dst = f"rf_{tag}_{our_split}_{iid}.jpg"
                shutil.copy2(src, OUT / "images" / our_split / dst)
                (OUT / "labels" / our_split / f"{Path(dst).stem}.txt").write_text(
                    "\n".join(lines) + "\n")
                stats[f"{our_split}_roboflow"] += 1
    return stats


def export_single_class(dst: Path | None = None) -> str:
    """Derive a single-class ``insect`` detection set from the 2-class boxes.

    Reuses the GrabCut/Roboflow boxes already in ``insect_det`` (no re-compute):
    images are symlinked, labels rewritten to class 0. Used to train the
    high-mAP detector; the 2-class set stays for building classifier crops.
    """
    dst = dst or (C.INTERIM_DIR / "insect_det1")
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (OUT / "images" / split).iterdir():
            link = dst / "images" / split / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
        for lbl in (OUT / "labels" / split).glob("*.txt"):
            lines = ["0 " + " ".join(l.split()[1:])
                     for l in lbl.read_text().splitlines() if l.strip()]
            (dst / "labels" / split / lbl.name).write_text("\n".join(lines) + "\n")
    (dst / "data.yaml").write_text(
        f"# single-class insect detector (high-mAP stage 1)\n"
        f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: [insect]\n")
    return str(dst / "data.yaml")


def export_classifier_crops(dst: Path | None = None, min_size: int = 20) -> dict:
    """Crop every labeled box from the 2-class set into an ImageFolder for the
    pollinator/non_pollinator classifier: ``insect_cls/<split>/<class>/<crop>.jpg``.
    """
    dst = dst or (C.INTERIM_DIR / "insect_cls")
    names = {0: "pollinator", 1: "non_pollinator"}
    counts: Counter = Counter()
    for split in ("train", "val", "test"):
        for cn in names.values():
            (dst / split / cn).mkdir(parents=True, exist_ok=True)
        for lbl in sorted((OUT / "labels" / split).glob("*.txt")):
            cand = list((OUT / "images" / split).glob(lbl.stem + ".*"))
            if not cand:
                continue
            img = cv2.imread(str(cand[0]))
            if img is None:
                continue
            H, W = img.shape[:2]
            for i, line in enumerate(lbl.read_text().splitlines()):
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
                x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
                x2, y2 = min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                cv2.imwrite(str(dst / split / names[c] / f"{lbl.stem}_{i}.jpg"),
                            img[y1:y2, x1:x2])
                counts[f"{split}_{names[c]}"] += 1
    return dict(counts)


def _write_yaml() -> None:
    (OUT / "data.yaml").write_text(
        f"# 2-class insect detector: pollinator vs non_pollinator\n"
        f"path: {OUT.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 2\nnames: [pollinator, non_pollinator]\n")


def run(workers: int | None = None) -> dict:
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    tasks = _select_inat_records()
    stats: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP) as ex:
        for split, kind in ex.map(_process_inat, tasks, chunksize=32):
            stats[f"inat_{split}_{kind}"] += 1
    stats.update(_convert_roboflow())
    _write_yaml()
    return {"inat_images": len(tasks), "counts": dict(stats),
            "data_yaml": str(OUT / "data.yaml")}


if __name__ == "__main__":
    import os
    print(json.dumps(run(workers=os.cpu_count()), indent=2))
