"""Flower-visit counting on video with **instance-segmented** insects (v2 seg pipeline).

Designed for a single real-time camera stream: the incoming video is resampled to a
fixed **target fps (24)** so tracking is stable and compute is bounded regardless of
the source frame rate.

Per frame:
  1. Flowers: per-frame YOLO **detection** with stable IDs (``FlowerTracker``) -> a
     **separate** box per flower (never unified).
  2. Insects: single-class YOLO26-**seg** + BoT-SORT -> one pixel **mask** + track ID
     each. Masks that cover too much of the frame are dropped (a flower false positive,
     not an insect).
  3. Type: each insect crop -> species classifier -> coarse type (majority vote / track).
  4. Visit: counted when an insect **mask actually overlaps a flower box**
     (mask∩box / mask_area > threshold), as an enter-transition, debounced so a
     fly-off + return is not a second visit. Each visit is stamped with a **timestamp**
     (seconds into the clip).

Rendering uses ``supervision`` so every insect instance gets its **own colour** (by
track ID — bee #1 and bee #2 differ), with the flower boxes drawn on top.

Outputs (to ``test_video_result/`` by default):
  * ``<video>_visits.csv``    -> flower_id, total, <per insect type ...>
  * ``<video>_timeline.csv``  -> flower_id, track_id, type, t_enter_s  (one row per visit)
  * annotated ``<video>_annotated.mp4`` (per-instance masks + flower boxes + counts)

CLI:
    python -m src.cv_engine.video_seg --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect_seg/best.pt \
        --classifier-weights .../species/best.pt --save-video
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
    Classifier, FlowerTracker, POLLINATOR_TYPES, _in, _load_types,
)

TARGET_FPS = 24                # single-camera stream is resampled to this
MAX_MASK_FRAC = 0.35           # drop insect masks bigger than this (flower false positive)
OVERLAP_THR = 0.10             # mask∩flowerbox / mask_area needed to count as "on flower"


def _box_overlap_frac(mask: np.ndarray, box) -> float:
    """Fraction of an insect mask's pixels that fall inside a flower box."""
    a = int(mask.sum())
    if a == 0:
        return 0.0
    x1, y1, x2, y2 = (int(v) for v in box)
    inside = int(mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum())
    return inside / a


def count_visits_seg(video, flower_weights, insect_weights, classifier_weights,
                     out_dir: Path, conf=0.25, save_video=False,
                     flower_interval=5, target_fps=TARGET_FPS, flower_conf=0.15) -> dict:
    import supervision as sv
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    classifier = Classifier(classifier_weights) if classifier_weights else None
    types = _load_types() if classifier is not None else {}

    in_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(in_fps / target_fps))          # resample -> ~target fps
    out_fps = in_fps / stride

    visits = defaultdict(Counter)
    timeline: list[dict] = []
    track_votes: dict[int, Counter] = defaultdict(Counter)
    counted: set[tuple[int, str]] = set()   # (track_id, flower_id) counted once ever
    ftracker = FlowerTracker(flower_model, flower_conf)   # lower conf -> catch more flowers
    mask_ann = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK, opacity=0.55)
    writer = None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False, vid_stride=stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        H, W = frame.shape[:2]
        t_s = round(fi * stride / in_fps, 2)             # timestamp (s) of this frame
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        if fi == 0 and save_video:
            writer = cv2.VideoWriter(str(out_dir / (Path(video).stem + "_annotated.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

        det = sv.Detections.from_ultralytics(res)
        # keep only tracked insects with a mask that is not flower-sized
        det_draw, labels = sv.Detections.empty(), []
        if det.tracker_id is not None and det.mask is not None and len(det):
            keep = np.zeros(len(det), dtype=bool)
            for k in range(len(det)):
                m = det.mask[k]
                if m.sum() / (H * W) > MAX_MASK_FRAC:     # too big -> flower, not insect
                    continue
                keep[k] = True
                tid = int(det.tracker_id[k])
                if classifier is not None:
                    x1, y1, x2, y2 = (int(v) for v in det.xyxy[k])
                    sp = classifier.predict(frame[y1:y2, x1:x2])
                    track_votes[tid][types.get(sp, sp)] += 1
                    cls = track_votes[tid].most_common(1)[0][0]
                else:
                    cls = "insect"
                # assign to the flower whose box the mask overlaps most
                best_fid, best_ov = None, OVERLAP_THR
                for fid, fb in flowers:
                    ov = _box_overlap_frac(m, fb)
                    if ov >= best_ov:
                        best_ov, best_fid = ov, fid
                # count each (insect, flower) pair only once ever -> a fly-off and
                # return to the same flower is NOT a second visit
                if best_fid is not None and (tid, best_fid) not in counted:
                    counted.add((tid, best_fid))
                    visits[best_fid]["total"] += 1
                    visits[best_fid][cls] += 1
                    timeline.append({"flower_id": best_fid, "track_id": tid,
                                     "type": cls, "t_enter_s": t_s})
            det_draw = det[keep]
            labels = [f"{track_votes[int(t)].most_common(1)[0][0]} #{int(t)}"
                      for t in det_draw.tracker_id]

        if writer is not None:
            annotated = frame.copy()
            if len(det_draw):
                annotated = mask_ann.annotate(annotated, det_draw)
            annotated = _draw_boxes_labels(annotated, det_draw, labels, flowers, visits, sv)
            writer.write(annotated)
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
    tl_path = out_dir / (Path(video).stem + "_timeline.csv")
    with open(tl_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "track_id", "type", "t_enter_s"])
        w.writeheader(); w.writerows(timeline)
    return {"video": Path(video).name, "flowers": len(ftracker.seen),
            "out_fps": round(out_fps, 1), "visits": {r["flower_id"]: r["total"] for r in rows},
            "csv": str(csv_path), "timeline": str(tl_path)}


def _draw_boxes_labels(frame, det, labels, flowers, visits, sv):
    # per-instance insect label (colour matches the mask, i.e. the track)
    for k in range(len(det)):
        x1, y1 = int(det.xyxy[k][0]), int(det.xyxy[k][1])
        col = sv.ColorPalette.DEFAULT.by_idx(int(det.tracker_id[k])).as_bgr()
        cv2.putText(frame, labels[k], (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    # flower boxes on top (separate box per flower) + live count
    for fid, (x1, y1, x2, y2) in flowers:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{visits[fid]['total']}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True, help="YOLO26-seg insect weights")
    ap.add_argument("--classifier-weights", default="")
    ap.add_argument("--out", default=str(C.REPO_ROOT / "test_video_result"))
    ap.add_argument("--conf", type=float, default=0.25, help="insect seg confidence")
    ap.add_argument("--flower-conf", type=float, default=0.15, help="flower detect confidence")
    ap.add_argument("--flower-interval", type=int, default=5)
    ap.add_argument("--target-fps", type=int, default=TARGET_FPS)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits_seg(args.video, args.flower_weights, args.insect_weights,
                                      args.classifier_weights, Path(args.out), args.conf,
                                      args.save_video, args.flower_interval,
                                      args.target_fps, args.flower_conf), indent=2))


if __name__ == "__main__":
    main()
