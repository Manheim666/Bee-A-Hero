"""Bootstrap a YOLO instance-segmentation dataset for insects using SAM.

Insect segmentation-mask data is scarce, so we generate it: each already-boxed
insect image (iNat + BEE.v8i, single centered organism) is passed to Segment
Anything (MobileSAM, bundled with Ultralytics) with a center-point prompt; the
resulting mask is converted to a YOLO-seg polygon label. Single class ``insect``
(type comes from the species classifier downstream) → highest, cleanest mask mAP.

Output (git-ignored): data/interim/insect_seg/{images,labels}/{train,val,test} + data.yaml

CLI:  python -m src.cv_engine.sam_to_seg
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config as C

SRC = C.INTERIM_DIR / "insect_det1"          # existing boxed insect images
OUT = C.INTERIM_DIR / "insect_seg"


def _mask_to_polygon(mask, W, H, eps_frac=0.004, min_area_frac=0.005):
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


def run() -> dict:
    from ultralytics import SAM
    model = SAM("mobile_sam.pt")
    stats = {"labeled": 0, "skipped": 0}
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in sorted((SRC / "images" / split).iterdir()):
            img = cv2.imread(str(p))
            if img is None:
                stats["skipped"] += 1; continue
            H, W = img.shape[:2]
            r = model.predict(img, points=[[W / 2, H / 2]], labels=[1], verbose=False)[0]
            if r.masks is None or len(r.masks) == 0:
                stats["skipped"] += 1; continue
            m = r.masks.data[0].cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
            poly = _mask_to_polygon(m > 0, W, H)
            if poly is None:
                stats["skipped"] += 1; continue
            link = OUT / "images" / split / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
            (OUT / "labels" / split / f"{p.stem}.txt").write_text(f"0 {poly}\n")
            stats["labeled"] += 1
            if stats["labeled"] % 1000 == 0:
                print(f"labeled {stats['labeled']}", flush=True)
    (OUT / "data.yaml").write_text(
        f"# single-class insect instance segmentation (SAM-bootstrapped)\n"
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: [insect]\n")
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
