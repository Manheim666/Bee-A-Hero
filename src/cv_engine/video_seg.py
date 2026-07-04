"""Flower-visit counting on video with **instance-segmented** insects (v2 seg pipeline).

Difference from ``visit_counter`` (the bbox pipeline): insects are handled by a
YOLO26 **segmentation** model, so each tracked insect gets a pixel mask (drawn as
a colored overlay) instead of a plain box. Everything else is shared and imported
from ``visit_counter`` to keep one source of truth:

  1. Flowers: per-frame YOLO **detection** with stable IDs (``FlowerTracker``,
     IoU association) -> separate box per flower (never unified), dilated ROI.
  2. Insects: single-class YOLO26-**seg** + BoT-SORT -> one mask + track ID each.
  3. Type: each tracked insect crop -> species classifier -> coarse type
     (bee / fly / ant / butterfly / ...), majority vote over the track's life.
  4. Visit: counted when a tracked insect's **mask centroid** enters a flower ROI
     (enter-transition, debounced -> fly-off + return is not a second visit).

Outputs:
  * ``<video>_visits.csv``  -> flower_id, total, <per insect type...>
  * annotated ``<video>_annotated.mp4`` (flower boxes + insect masks + live counts)

CLI:
    python -m src.cv_engine.video_seg --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect_seg/best.pt \
        --classifier-weights .../insect_species/best.pt --save-video
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src import config as C
from src.cv_engine.visit_counter import (
    Classifier, FlowerTracker, POLLINATOR_TYPES, _center, _in, _load_types,
)

# stable per-type overlay colors (BGR); unknown types fall back to gray
_TYPE_COLORS = {
    "bee": (0, 180, 255), "wasp": (0, 220, 255), "fly": (255, 120, 0),
    "butterfly": (255, 0, 200), "ant": (40, 40, 200), "beetle": (0, 140, 0),
    "bug": (180, 80, 0),
}


def _mask_centroid(mask, box):
    """Centroid of a boolean mask; fall back to the box center if mask empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return _center(box)
    return float(xs.mean()), float(ys.mean())


def count_visits_seg(video, flower_weights, insect_weights, classifier_weights,
                     out_dir: Path, conf=0.1, debounce=20, save_video=False,
                     flower_interval=5, vid_stride=2) -> dict:
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    classifier = Classifier(classifier_weights) if classifier_weights else None

    types = _load_types() if classifier is not None else {}
    visits = defaultdict(Counter)
    track_votes: dict[int, Counter] = defaultdict(Counter)
    last_flower: dict[int, str | None] = {}
    last_visit_frame: dict[tuple[int, str], int] = {}
    ftracker = FlowerTracker(flower_model, conf)
    writer = None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False, vid_stride=vid_stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        H, W = frame.shape[:2]
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        if fi == 0 and save_video:
            writer = cv2.VideoWriter(str(out_dir / (Path(video).stem + "_annotated.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        boxes = res.boxes
        # per-insect full-frame masks aligned to boxes (seg model), else None
        masks = None
        if res.masks is not None and len(res.masks):
            masks = res.masks.data.cpu().numpy()  # (n, h, w) at model res
        labels: dict[int, str] = {}
        overlays: list[tuple[np.ndarray, tuple]] = []  # (mask_bool_HxW, color) to blend
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for k, (tid, b) in enumerate(zip(ids, xyxy)):
                # resolve the insect's mask (resize model-res mask -> frame size)
                m = None
                if masks is not None and k < len(masks):
                    m = masks[k]
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
                    m = m > 0
                if classifier is not None:
                    x1, y1, x2, y2 = map(int, b)
                    sp = classifier.predict(frame[y1:y2, x1:x2])
                    track_votes[tid][types.get(sp, sp)] += 1
                    cls = track_votes[tid].most_common(1)[0][0]
                else:
                    cls = "insect"
                labels[tid] = cls
                col = _TYPE_COLORS.get(cls, (200, 200, 200))
                if m is not None:
                    overlays.append((m, col))
                pt = _mask_centroid(m, b) if m is not None else _center(b)
                cur = next((fid for fid, fb in flowers if _in(fb, pt)), None)
                if cur is not None and last_flower.get(tid) != cur:
                    if fi - last_visit_frame.get((tid, cur), -10**9) > debounce:
                        visits[cur]["total"] += 1
                        visits[cur][cls] += 1
                        last_visit_frame[(tid, cur)] = fi
                last_flower[tid] = cur
        if writer is not None:
            writer.write(_annotate_seg(frame, flowers, res, labels, overlays, visits))
    if writer is not None:
        writer.release()

    for fid in ftracker.seen:
        visits[fid]
    ctypes = sorted({t for v in visits.values() for t in v if t != "total"})
    rows = [{"flower_id": k, "total": v["total"], **{t: v.get(t, 0) for t in ctypes}}
            for k, v in sorted(visits.items())]
    csv_path = out_dir / (Path(video).stem + "_visits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "total"] + ctypes)
        w.writeheader(); w.writerows(rows)
    return {"video": Path(video).name, "flowers": len(ftracker.seen),
            "visits": {r["flower_id"]: r["total"] for r in rows}, "csv": str(csv_path)}


def _annotate_seg(frame, flowers, res, labels, overlays, visits):
    # insect masks first (blended), so flower boxes/text stay crisp on top
    if overlays:
        blend = frame.copy()
        for m, col in overlays:
            blend[m] = col
        cv2.addWeighted(blend, 0.45, frame, 0.55, 0, frame)
        for m, col in overlays:
            cnts, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, col, 2)
    for fid, (x1, y1, x2, y2) in flowers:  # separate box per flower (not unified)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{visits[fid]['total']}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if res.boxes is not None and res.boxes.id is not None:
        for tid, b in zip(res.boxes.id.int().cpu().tolist(), res.boxes.xyxy.cpu().numpy()):
            x1, y1 = int(b[0]), int(b[1])
            cls = labels.get(tid, "insect")
            poll = cls in POLLINATOR_TYPES
            col = _TYPE_COLORS.get(cls, (200, 200, 200))
            tag = f"{cls}{' (pollinator)' if poll else ''} #{tid}"
            cv2.putText(frame, tag, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True, help="YOLO26-seg insect weights")
    ap.add_argument("--classifier-weights", default="")
    ap.add_argument("--out", default=str(C.REPO_ROOT / "test_video_result"))
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--debounce", type=int, default=20)
    ap.add_argument("--flower-interval", type=int, default=5)
    ap.add_argument("--vid-stride", type=int, default=2)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits_seg(args.video, args.flower_weights, args.insect_weights,
                                      args.classifier_weights, Path(args.out), args.conf,
                                      args.debounce, args.save_video, args.flower_interval,
                                      args.vid_stride), indent=2))


if __name__ == "__main__":
    main()
