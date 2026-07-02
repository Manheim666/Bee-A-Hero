"""Tight bounding boxes for centered iNaturalist organisms via SAM.

iNaturalist crops show one organism, roughly centered. Prompting Segment
Anything (MobileSAM, bundled with Ultralytics — no extra dependency) with a
center point yields a tight mask of just the insect, whose bounding box hugs the
animal far better than GrabCut's near-full-frame boxes.

``sam_box(path)`` returns ``(cx, cy, w, h)`` normalized [0,1], or a centered
fallback if SAM finds nothing usable. The SAM model is loaded once (lazy).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_FALLBACK = (0.5, 0.5, 0.85, 0.85)


@lru_cache(maxsize=1)
def _model():
    from ultralytics import SAM
    return SAM("mobile_sam.pt")


def sam_box(path: str, min_frac: float = 0.01, max_frac: float = 0.999):
    """Tight (cx, cy, w, h) normalized box around the centered organism."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return _FALLBACK
    h, w = img.shape[:2]
    r = _model().predict(img, points=[[w / 2, h / 2]], labels=[1], verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return _FALLBACK
    mask = r.masks.data[0].cpu().numpy().astype("uint8")
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return _FALLBACK
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    if not (min_frac <= bw * bh <= max_frac):
        return _FALLBACK
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return (float(cx), float(cy), float(bw), float(bh))
