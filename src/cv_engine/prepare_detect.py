"""Build YOLO **detection** datasets (bbox, no segmentation) from the real,
video-domain datasets the user downloaded into ``data/raw/``.

Two datasets are produced (git-ignored, under ``data/interim/``):

* ``flower_det2``  — single class ``flower``. Sources: the Roboflow flower COCO
  exports + the per-frame flower ROI boxes from the "flower visits" (Ștefan et al.
  2025, time-lapse pollinator) dataset.
* ``insect_multidet`` — multi-class ``[bee, fly, beetle, bug, butterfly]``.
  Sources: "flower visits" full-frame arthropod boxes (taxonomic *order* → coarse
  type) + the Roboflow honey-bee COCO exports (all → ``bee``). Types are learned
  as detection classes, so no fragile separate species classifier is needed.

"flower visits" is time-lapse (many near-identical frames per plant), so splits are
grouped by ``plant_folder`` to avoid train/val leakage.

CLI:  python -m src.cv_engine.prepare_detect            # both
      python -m src.cv_engine.prepare_detect flower     # just flower
      python -m src.cv_engine.prepare_detect insect     # just insect
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src import config as C

RAW = C.RAW_DIR
FV = RAW / "flower visits"                       # Zenodo time-lapse pollinator set
FV_ANN = FV / "annotations" / "raw" / "annotations_full_frames.txt"

# Roboflow COCO exports (split dirs: train/valid/test, _annotations.coco.json)
FLOWER_COCO = ["flowerDetecter-v2.v1i.coco", "flower detection.v3i.coco"]
BEE_COCO = ["Bees detection model.v1i.coco", "Honey Bee Project.v1i.coco",
            "Bee Detection in the Wild/archive", "BEE_coco"]
# Honey Bee Detection Model.v4 is hive-monitoring (dense caged bees) — excluded:
# that domain hurt garden-video mAP in earlier experiments.

_RF_SPLIT = {"train": "train", "valid": "val", "test": "test"}
# coco category NAMES that are NOT a flower (supercategory placeholders / plant parts)
_NOT_FLOWER = {"flowers", "yolov5", "flower-detection", "anther"}
# taxonomic order -> coarse insect class
ORDER2CLS = {"hymenoptera": "bee", "diptera": "fly", "coleoptera": "beetle",
             "hemiptera": "bug", "lepidoptera": "butterfly"}
INSECT_NAMES = ["bee", "fly", "beetle", "bug", "butterfly"]
# per-class cap on flower-visits frames (balance; bee is hugely over-represented)
FV_INSECT_CAP = {"bee": 2000, "fly": 6000, "beetle": 3300, "bug": 900, "butterfly": 200}
FV_FLOWER_CAP = 2500
BEE_COCO_CAP = 1200          # cut bee dominance (bee AP already strong) -> rebalance


# --------------------------------------------------------------------------- #
# COCO (Roboflow) -> YOLO
# --------------------------------------------------------------------------- #
def _coco_split(ds_dir: Path, rf_split: str):
    """Yield (image_path, W, H, [(cls_name, x, y, w, h), ...]) for one split."""
    ann = ds_dir / rf_split / "_annotations.coco.json"
    if not ann.exists():
        return
    coco = json.load(open(ann))
    name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    boxes = defaultdict(list)
    for a in coco["annotations"]:
        boxes[a["image_id"]].append((name.get(a["category_id"], "?"), *a["bbox"]))
    for iid, im in imgs.items():
        p = ds_dir / rf_split / im["file_name"]
        if p.exists():
            yield p, im["width"], im["height"], boxes.get(iid, [])


def _write(out: Path, split: str, img: Path, W, H, lines: list[str]):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    link = out / "images" / split / img.name
    if not link.exists():
        try:
            link.symlink_to(img.resolve())
        except FileExistsError:
            pass
    (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def _clamp01(v):
    return min(1.0, max(0.0, v))


def _yolo_line(cls: int, x, y, w, h, W, H):
    # clamp the box to the image so no out-of-bounds coords (Ultralytics drops those)
    x1, y1 = _clamp01(x / W), _clamp01(y / H)
    x2, y2 = _clamp01((x + w) / W), _clamp01((y + h) / H)
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# --------------------------------------------------------------------------- #
# flower visits txt -> rows
# --------------------------------------------------------------------------- #
def _load_fv_rows():
    import csv as _csv
    rows = list(_csv.DictReader(open(FV_ANN), delimiter="\t"))
    return rows


def _fv_group_split(files: list[str], groups: dict[str, str], rng):
    """Assign each plant_folder group to train/val/test 80/10/10 (grouped)."""
    gset = sorted(set(groups[f] for f in files))
    rng.shuffle(gset)
    n = len(gset)
    val = set(gset[: max(1, n // 10)])
    test = set(gset[n // 10: n // 10 + max(1, n // 10)])
    def which(f):
        g = groups[f]
        return "val" if g in val else "test" if g in test else "train"
    return which


# --------------------------------------------------------------------------- #
# Flower dataset
# --------------------------------------------------------------------------- #
def build_flower() -> dict:
    out = C.INTERIM_DIR / "flower_det2"
    stats: Counter = Counter()
    for ds in FLOWER_COCO:
        d = RAW / ds
        for rf, our in _RF_SPLIT.items():
            for p, W, H, boxes in _coco_split(d, rf):
                lines = [_yolo_line(0, x, y, w, h, W, H)
                         for nm, x, y, w, h in boxes if nm not in _NOT_FLOWER and w > 1 and h > 1]
                if lines:
                    _write(out, our, p, W, H, lines); stats[f"{our}_coco"] += 1

    # flower-visits ROI (one flower box per frame), grouped-split, capped
    rows = _load_fv_rows()
    by_file = {}
    groups = {}
    for r in rows:
        f = r["filename_full_frame"]; groups[f] = r["plant_folder"]
        by_file.setdefault(f, r)   # ROI identical across boxes of a frame
    files = list(by_file)
    rng = random.Random(C.SEED); rng.shuffle(files); files = files[:FV_FLOWER_CAP * 2]
    which = _fv_group_split(files, groups, rng)
    kept = Counter()
    for f in files:
        our = which(f)
        if kept[our] >= FV_FLOWER_CAP:
            continue
        r = by_file[f]
        img = FV / "raw" / f
        if not img.exists():
            continue
        try:
            W, H = Image.open(img).size
        except Exception:
            continue
        try:
            x, y, w, h = float(r["x_roi"]), float(r["y_roi"]), float(r["width_roi"]), float(r["height_roi"])
        except ValueError:
            continue
        _write(out, our, img, W, H, [_yolo_line(0, x, y, w, h, W, H)])
        stats[f"{our}_fv"] += 1; kept[our] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: [flower]\n")
    return {"out": str(out), "counts": dict(stats)}


# --------------------------------------------------------------------------- #
# Insect multi-class dataset
# --------------------------------------------------------------------------- #
def build_insect() -> dict:
    out = C.INTERIM_DIR / "insect_multidet"
    cid = {n: i for i, n in enumerate(INSECT_NAMES)}
    stats: Counter = Counter()

    # ---- flower-visits full-frame boxes (order -> class), grouped-split, capped
    rows = _load_fv_rows()
    groups = {r["filename_full_frame"]: r["plant_folder"] for r in rows}
    boxes_by_file = defaultdict(list)
    for r in rows:
        cls = ORDER2CLS.get((r["order"] or "").lower())
        if cls is None:
            continue
        boxes_by_file[r["filename_full_frame"]].append((cls, r))
    files = list(boxes_by_file)
    rng = random.Random(C.SEED); rng.shuffle(files)
    which = _fv_group_split(files, groups, rng)
    cap = Counter()
    for f in files:
        # primary class = first box's class (for capping/balance)
        prim = boxes_by_file[f][0][0]
        if cap[prim] >= FV_INSECT_CAP.get(prim, 3000):
            continue
        img = FV / "raw" / f
        if not img.exists():
            continue
        try:
            W, H = Image.open(img).size
        except Exception:
            continue
        lines = []
        for cls, r in boxes_by_file[f]:
            try:
                x, y, w, h = float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])
            except ValueError:
                continue
            if w > 1 and h > 1:
                lines.append(_yolo_line(cid[cls], x, y, w, h, W, H))
        if lines:
            _write(out, which(f), img, W, H, lines)
            stats[f"{which(f)}_fv_{prim}"] += 1; cap[prim] += 1

    # ---- bee COCO sets -> all boxes = bee
    for ds in BEE_COCO:
        d = RAW / ds
        kept = 0
        for rf, our in _RF_SPLIT.items():
            for p, W, H, bxs in _coco_split(d, rf):
                if our == "train" and kept >= BEE_COCO_CAP:
                    break
                lines = [_yolo_line(cid["bee"], x, y, w, h, W, H)
                         for nm, x, y, w, h in bxs if w > 1 and h > 1]
                if lines:
                    _write(out, our, p, W, H, lines)
                    stats[f"{our}_bee_{Path(ds).name[:10]}"] += 1
                    if our == "train":
                        kept += 1

    names = ", ".join(INSECT_NAMES)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(INSECT_NAMES)}\nnames: [{names}]\n")
    return {"out": str(out), "counts": dict(stats)}


def aug_insect_from_inat(cap_per_class: int = 2500) -> dict:
    """Boost the weak insect classes with iNaturalist crops (GrabCut boxes).

    Per-class AP showed fly/beetle/bug lagging (few, small, camouflaged on flowers),
    while bee/butterfly are strong. So top up the weak classes from iNat: for each
    taxonomic order, take up to ``cap_per_class`` images, box the centred organism
    with GrabCut and add it to ``insect_multidet``. Butterfly is already strong so it
    gets a smaller top-up.
    """
    import csv as _csv
    import cv2
    from src.data_pipeline.flower.make_detection_dataset import auto_bbox
    out = C.INTERIM_DIR / "insect_multidet"
    cid = {n: i for i, n in enumerate(INSECT_NAMES)}
    order_cls = {"Diptera": "fly", "Coleoptera": "beetle", "Hemiptera": "bug",
                 "Lepidoptera": "butterfly"}
    per_cls_cap = {"fly": cap_per_class, "beetle": cap_per_class,
                   "bug": cap_per_class, "butterfly": 1200}
    by_cls = defaultdict(list)
    for r in _csv.DictReader(open(C.MANIFEST_DIR / "split_manifest.csv")):
        cl = order_cls.get(r["order"])
        if cl:
            by_cls[cl].append(r)
    rng = random.Random(C.SEED)
    stats: Counter = Counter()
    for cl, recs in by_cls.items():
        rng.shuffle(recs); recs = recs[:per_cls_cap.get(cl, cap_per_class)]
        for r in recs:
            split = {"train": "train", "val": "val", "test": "test"}.get(r["split"], "train")
            img = C.REPO_ROOT / r["path"]
            im = cv2.imread(str(img))
            if im is None:
                continue
            cx, cy, nw, nh, _ = auto_bbox(im)
            name = f"inat_{cl}_{Path(r['path']).stem}.jpg"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            link = out / "images" / split / name
            if not link.exists():
                try:
                    link.symlink_to(img.resolve())
                except FileExistsError:
                    pass
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
                f"{cid[cl]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            stats[f"{split}_{cl}"] += 1
    return dict(stats)


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("both", "flower"):
        print("FLOWER:", json.dumps(build_flower(), indent=2))
    if what in ("both", "insect"):
        print("INSECT:", json.dumps(build_insect(), indent=2))
    if what in ("both", "insect", "aug"):
        print("AUG:", json.dumps(aug_insect_from_inat(), indent=2))
