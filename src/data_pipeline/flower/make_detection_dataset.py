"""
make_detection_dataset.py
-------------------------
Turn the merged Flower Classification dataset (class subfolders) into an
OBJECT-DETECTION-ready dataset in YOLO format, AND emit a labels.csv that
maps every image to its class.

IMPORTANT / HONEST NOTE
-----------------------
The source is a *classification* dataset: one centered flower per image, with
NO ground-truth boxes. This script GENERATES boxes automatically with OpenCV
GrabCut foreground segmentation (a tight box around the detected flower region).
These boxes are approximate and machine-made, not human-verified. They give you
"clear boundaries" to train/prototype a detector, but you should spot-check and
hand-correct a sample before trusting them as ground truth.

INPUT  (created by merge_flowers.py):
    merged_dataset/
        Training Data/<Class>/*.jpeg
        Validation Data/<Class>/*.jpeg
        Testing Data/<Class>/*.jpeg

OUTPUT:
    yolo_dataset/
        images/{train,val,test}/<Class>_<orig>.jpeg
        labels/{train,val,test}/<Class>_<orig>.txt     # "<cls> cx cy w h" (normalized)
        data.yaml
        classes.txt
    labels.csv         # image -> class mapping (classification)
    annotations.csv    # image -> class + pixel bbox (detection, human-readable)

Usage:
    python make_detection_dataset.py
    python make_detection_dataset.py --src merged_dataset --out yolo_dataset --workers 16
    python make_detection_dataset.py --limit 20    # quick smoke test (20 imgs/class)
"""

import argparse
import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpeg", ".jpg", ".png"}
SPLIT_MAP = {
    "Training Data": "train",
    "Validation Data": "val",
    "Testing Data": "test",
}
GC_SIZE = 160          # work resolution for GrabCut (speed)
GC_ITERS = 3           # GrabCut iterations
MIN_AREA_FRAC = 0.02   # if fg smaller than this -> fallback box
MAX_AREA_FRAC = 0.985  # if fg bigger than this  -> fallback box
FALLBACK_FRAC = 0.92   # centered fallback box size (fraction of image)


def auto_bbox(img):
    """Return (cx,cy,w,h) normalized [0,1] for the main flower region.
    Falls back to a centered box if segmentation is unreliable."""
    h, w = img.shape[:2]
    scale = GC_SIZE / max(h, w)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh))

    mask = np.zeros((sh, sw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = 0.06  # border margin -> assumed background
    rect = (int(sw * m), int(sh * m), int(sw * (1 - 2 * m)), int(sh * (1 - 2 * m)))

    used_fallback = False
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            area_frac = (bw * bh) / float(sw * sh)
            if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
                used_fallback = True
        else:
            used_fallback = True
    except Exception:
        used_fallback = True

    if used_fallback:
        f = FALLBACK_FRAC
        cx, cy, nw, nh = 0.5, 0.5, f, f
    else:
        cx = (x + bw / 2) / sw
        cy = (y + bh / 2) / sh
        nw = bw / sw
        nh = bh / sh

    # clamp
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    nw, nh = min(nw, 1), min(nh, 1)
    return cx, cy, nw, nh, used_fallback


def process_one(task):
    """Worker: compute bbox for a single image. Returns dict or None on read error."""
    src, split, cls, cls_id, dst_name = task
    img = cv2.imread(src)
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy, nw, nh, fb = auto_bbox(img)
    # pixel coords for the human-readable annotations.csv
    bw_px, bh_px = nw * w, nh * h
    xmin = int(round(cx * w - bw_px / 2))
    ymin = int(round(cy * h - bh_px / 2))
    xmax = int(round(cx * w + bw_px / 2))
    ymax = int(round(cy * h + bh_px / 2))
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)
    return {
        "src": src, "split": split, "cls": cls, "cls_id": cls_id,
        "dst_name": dst_name, "w": w, "h": h,
        "cx": cx, "cy": cy, "nw": nw, "nh": nh,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "fallback": fb,
    }


def build_tasks(src_root, limit):
    classes = set()
    tasks = []
    per_class_counter = {}
    for split_dir, split_key in SPLIT_MAP.items():
        sdir = src_root / split_dir
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            classes.add(cls)
    class_list = sorted(classes)
    class_id = {c: i for i, c in enumerate(class_list)}

    seen_names = set()
    for split_dir, split_key in SPLIT_MAP.items():
        sdir = src_root / split_dir
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            key = (split_key, cls)
            per_class_counter.setdefault(key, 0)
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue
                if limit and per_class_counter[key] >= limit:
                    break
                per_class_counter[key] += 1
                # unique, collision-proof destination name (flat per split).
                # Dedup on the STEM (not full name) so that e.g. "X.jpeg" and
                # "X.png" don't collide onto the same label .txt file.
                base = f"{cls}_{f.name}"
                stem = Path(base).stem
                dst = base
                i = 1
                while (split_key, Path(dst).stem) in seen_names:
                    dst = f"{stem}_{i}{f.suffix}"
                    i += 1
                seen_names.add((split_key, Path(dst).stem))
                tasks.append((str(f), split_key, cls, class_id[cls], dst))
    return tasks, class_list, class_id


# repo root = .../Bee-A-Hero  (this file is at src/data_pipeline/flower/make_detection_dataset.py)
REPO = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO / "data" / "processed" / "flower" / "classification"
DEFAULT_OUT = REPO / "data" / "processed" / "flower" / "yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="classification dataset (default: data/processed/flower/classification)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="YOLO output (default: data/processed/flower/yolo)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="max images per class+split (0 = all)")
    args = ap.parse_args()

    src_root = Path(args.src)
    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    tasks, class_list, class_id = build_tasks(src_root, args.limit)
    print(f"Images to process: {len(tasks)}  |  classes: {len(class_list)}  |  workers: {args.workers}")

    labels_rows = []       # image_path, split, class, class_id
    ann_rows = []          # image_path, width, height, class, class_id, xmin, ymin, xmax, ymax
    fallback_count = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r is None:
                continue
            if r["fallback"]:
                fallback_count += 1

            split = r["split"]
            dst_img = out / "images" / split / r["dst_name"]
            dst_lbl = out / "labels" / split / (Path(r["dst_name"]).stem + ".txt")

            shutil.copy2(r["src"], dst_img)
            with open(dst_lbl, "w") as fh:
                fh.write(f"{r['cls_id']} {r['cx']:.6f} {r['cy']:.6f} {r['nw']:.6f} {r['nh']:.6f}\n")

            rel = f"images/{split}/{r['dst_name']}"
            labels_rows.append([rel, split, r["cls"], r["cls_id"]])
            ann_rows.append([rel, r["w"], r["h"], r["cls"], r["cls_id"],
                             r["xmin"], r["ymin"], r["xmax"], r["ymax"]])

            if done % 2000 == 0:
                print(f"  ...{done}/{len(tasks)}")

    # sort for stable output
    labels_rows.sort()
    ann_rows.sort()

    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split", "class", "class_id"])
        w.writerows(labels_rows)

    with open(out / "annotations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "width", "height", "class", "class_id",
                    "xmin", "ymin", "xmax", "ymax"])
        w.writerows(ann_rows)

    with open(out / "classes.txt", "w") as fh:
        fh.write("\n".join(class_list) + "\n")

    with open(out / "data.yaml", "w") as fh:
        fh.write(f"path: {out.resolve()}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("test: images/test\n")
        fh.write(f"nc: {len(class_list)}\n")
        fh.write("names:\n")
        for c in class_list:
            fh.write(f"  - {c}\n")

    print("\n" + "=" * 60)
    print("DETECTION DATASET READY ->", out.resolve())
    print("=" * 60)
    print(f"images written : {len(labels_rows)}")
    print(f"fallback boxes : {fallback_count} "
          f"({100*fallback_count/max(1,len(labels_rows)):.1f}% used centered box)")
    print(f"classes        : {class_list}")
    print("files          : data.yaml, classes.txt, labels.csv, annotations.csv")


if __name__ == "__main__":
    main()
