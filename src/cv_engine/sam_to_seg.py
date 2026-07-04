"""Bootstrap a YOLO instance-segmentation dataset for insects using **box-prompted** SAM.

Why box prompts (not a center point): a center-point prompt makes SAM grab the most
salient blob at the image centre — on real flower-visit video that blob is often the
*flower*, so the earlier model learned to segment flowers. Instead we prompt SAM with
the **existing YOLO detection boxes** (real garden-bee scenes from BEE.v8i + iNat
crops), so every mask is tightly anchored to an actual insect box. Multiple boxes per
image → multiple instances.

Two more quality levers baked in:
  * **Real-scene data**: uses ``insect_det`` (includes BEE.v8i garden bees in natural
    scenes with flowers/background), not just centred crops → the model learns to
    segment the insect *within* a scene, matching video.
  * **Flower background negatives**: a fraction of pure-flower images are added with
    **empty** label files, teaching the segmenter that flowers are NOT insects — this
    directly fixes the "mask bleeds onto the flower" failure.

Single class ``insect`` (type comes from the species classifier downstream).

Output (git-ignored): data/interim/insect_seg/{images,labels}/{train,val,test} + data.yaml

CLI:  python -m src.cv_engine.sam_to_seg
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from src import config as C

SRC = C.INTERIM_DIR / "insect_det"           # 2-class boxes: iNat crops + BEE.v8i scenes
FLOWER_SRC = C.INTERIM_DIR / "flower_det"    # flower detection images -> negatives
OUT = C.INTERIM_DIR / "insect_seg"
NEG_FRAC = 0.15                              # flower-only negatives as a fraction of positives


def _mask_to_polygon(mask, W, H, eps_frac=0.004, min_area_frac=0.002):
    """Largest external contour -> normalized YOLO-seg polygon string, or None."""
    cnts, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * W * H:
        return None
    ap = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in ap)


def _read_boxes(lbl_path, W, H):
    """Read a YOLO label file -> list of pixel xyxy boxes (class ignored -> insect)."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        boxes.append([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
    return boxes


def run() -> dict:
    from ultralytics import SAM
    model = SAM("mobile_sam.pt")
    stats = {"labeled": 0, "instances": 0, "skipped": 0, "negatives": 0}
    rng = random.Random(C.SEED)

    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        img_dir = SRC / "images" / split
        lbl_dir = SRC / "labels" / split
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            img = cv2.imread(str(p))
            if img is None:
                stats["skipped"] += 1; continue
            H, W = img.shape[:2]
            boxes = _read_boxes(lbl_dir / f"{p.stem}.txt", W, H)
            if not boxes:
                stats["skipped"] += 1; continue
            r = model.predict(img, bboxes=boxes, labels=[1] * len(boxes), verbose=False)[0]
            if r.masks is None or len(r.masks) == 0:
                stats["skipped"] += 1; continue
            lines = []
            for m in r.masks.data.cpu().numpy():
                if m.shape != (H, W):
                    m = cv2.resize(m.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
                poly = _mask_to_polygon(m > 0, W, H)
                if poly is not None:
                    lines.append(f"0 {poly}")
            if not lines:
                stats["skipped"] += 1; continue
            link = OUT / "images" / split / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
            (OUT / "labels" / split / f"{p.stem}.txt").write_text("\n".join(lines) + "\n")
            stats["labeled"] += 1; stats["instances"] += len(lines)
            if stats["labeled"] % 1000 == 0:
                print(f"labeled {stats['labeled']}  instances {stats['instances']}", flush=True)

        # flower-only background negatives (empty label) -> "flowers are not insects"
        fdir = FLOWER_SRC / "images" / split
        if fdir.is_dir():
            flowers = sorted(fdir.iterdir())
            rng.shuffle(flowers)
            n_neg = int(stats["labeled"] * NEG_FRAC / 3)  # rough per-split share
            for fp in flowers[:n_neg]:
                name = f"neg_{fp.name}"
                link = OUT / "images" / split / name
                if not link.exists():
                    try:
                        link.symlink_to(fp.resolve())
                    except FileNotFoundError:
                        continue
                (OUT / "labels" / split / f"neg_{fp.stem}.txt").write_text("")  # empty = background
                stats["negatives"] += 1

    (OUT / "data.yaml").write_text(
        f"# single-class insect instance segmentation (box-prompted SAM + flower negatives)\n"
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: [insect]\n")
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
