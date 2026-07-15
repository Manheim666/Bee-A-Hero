"""Flower-visit counting on video (two-stage insect recognition).

Pipeline:
  1. Detect flowers on **every frame** and keep stable IDs across frames via IoU
     association (handles moving camera/flowers); each flower's box is dilated
     into an approach ROI (``flower_1, flower_2, ...``).
  2. Detect and track insects across frames with a single-class YOLO26 detector
     + BoT-SORT (one track ID per insect).
  3. Classify each tracked insect crop with the iNaturalist-pretrained classifier
     as ``pollinator`` or ``non_pollinator`` (majority vote over a track's life).
  4. Count a visit whenever a tracked insect enters a flower ROI (enter-transition,
     debounced so one dwell = one visit and tracker flicker does not double-count).

Outputs:
  * ``<video>_visits.csv``  -> flower_id, total, pollinator, non_pollinator
  * annotated ``<video>_annotated.mp4`` (flowers + tracks + live counts) if --save-video

CLI:
    python -m src.cv_engine.visit_counter --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect/best.pt \
        --classifier-weights .../insect_classifier/best.pt --save-video
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src import config as C


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _in(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


class Classifier:
    """iNaturalist-pretrained pollinator/non_pollinator classifier (lazy torch)."""

    def __init__(self, weights: str):
        import timm
        import torch
        self.torch = torch
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.classes = ckpt["classes"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg, is_training=False)

    def predict(self, crop_bgr) -> str:
        from PIL import Image
        if crop_bgr.size == 0:
            return self.classes[0]
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            idx = int(self.model(x).argmax(1).item())
        return self.classes[idx]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _detect_flowers_raw(model, frame, conf, dilate=0.0):
    """Raw flower boxes. ``dilate`` pads each box (fraction of its size). Default 0: padding a
    large close-up flower inflated it to the whole frame, which the MAX_FLOWER_FRAC plausibility
    gate then rejected -> such videos silently counted 0 flowers/landings. Gate and draw on the
    true detector box (what the live path already does)."""
    res = model.predict(frame, conf=conf, verbose=False)[0]
    H, W = frame.shape[:2]
    out = []
    for b in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        dw, dh = (x2 - x1) * dilate, (y2 - y1) * dilate
        out.append((max(0, x1 - dw), max(0, y1 - dh), min(W, x2 + dw), min(H, y2 + dh)))
    return out


class FlowerTracker:
    """Per-frame flower detection with stable IDs across frames (IoU association).

    Handles dynamic video (moving camera/flowers): each frame the flowers are
    re-detected and matched to existing tracks by IoU so ``flower_1`` stays the
    same flower as it moves. Robustness (no retraining needed):

      * **Global one-to-one matching** -- every (track, detection) pair is ranked by
        IoU and the strongest pairs are assigned first, so two neighbouring flowers
        cannot swap boxes (stops the multi-flower ID mix-up).
      * **EMA box smoothing** (``smooth``) -- the matched box is blended into the
        track instead of replaced, damping per-frame jitter (a running box average).
      * **Presence hold** (``hold``) -- a flower's last-known box keeps being drawn
        through a few missed detections, so the box no longer blinks off/on.

    A longer ``max_missed`` grace keeps the *ID* alive for re-association even after
    the box stops being drawn. ``seen`` records every flower ID ever assigned.
    """

    def __init__(self, model, conf, dilate=0.0, iou_thr=0.3, max_missed=45,
                 smooth=0.5, hold=6):
        self.model, self.conf, self.dilate = model, conf, dilate
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.smooth, self.hold = smooth, hold
        self.tracks: dict[str, dict] = {}
        self.next_id = 1
        self.seen: set[str] = set()

    def update(self, frame):
        dets = _detect_flowers_raw(self.model, frame, self.conf, self.dilate)
        # rank every (track, det) IoU pair, assign strongest first -> global 1:1 match
        pairs = []
        for fid in self.tracks:
            for j, d in enumerate(dets):
                iou = _iou(self.tracks[fid]["box"], d)
                if iou >= self.iou_thr:
                    pairs.append((iou, fid, j))
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched, used = set(), set()
        for iou, fid, j in pairs:
            if fid in matched or j in used:
                continue
            prev = self.tracks[fid]["box"]                 # EMA-smooth the box (damp jitter)
            sm = tuple(self.smooth * p + (1 - self.smooth) * n for p, n in zip(prev, dets[j]))
            self.tracks[fid] = {"box": sm, "missed": 0}
            matched.add(fid); used.add(j)
        for fid in list(self.tracks):                      # unmatched track: hold last box, age it
            if fid not in matched:
                self.tracks[fid]["missed"] += 1
                if self.tracks[fid]["missed"] > self.max_missed:
                    del self.tracks[fid]
        for j, d in enumerate(dets):                       # unmatched det: mint a new flower
            if j not in used:
                fid = f"flower_{self.next_id}"; self.next_id += 1
                self.tracks[fid] = {"box": d, "missed": 0}
                self.seen.add(fid)
        return self.current()

    def current(self):
        """Active flower boxes (last known), held through <= ``hold`` misses to stop blink."""
        return [(fid, t["box"]) for fid, t in self.tracks.items() if t["missed"] <= self.hold]


POLLINATOR_TYPES = {"bee", "butterfly", "wasp", "fly"}  # highlighted + rolled up in CSV


def _load_types():
    """Map species class_id (str) -> coarse insect type from the taxonomy manifest."""
    import csv as _csv
    from src import config as _C
    bee = set(_C.BEE_FAMILIES)
    out = {}
    try:
        for r in _csv.DictReader(open(_C.MANIFEST_DIR / "split_manifest.csv")):
            o, f = r["order"], r["family"]
            if f in bee: t = "bee"
            elif o == "Lepidoptera": t = "butterfly"
            elif f == "Formicidae": t = "ant"
            elif o == "Coleoptera": t = "beetle"
            elif o == "Diptera": t = "fly"
            elif o == "Hymenoptera": t = "wasp"
            elif o == "Hemiptera": t = "bug"
            else: t = (o or "insect").lower()
            out[r["class_id"]] = t
    except FileNotFoundError:
        pass
    return out


def count_visits(video, flower_weights, insect_weights, classifier_weights,
                 out_dir: Path, conf=0.1, debounce=20, save_video=False,
                 flower_interval=5, vid_stride=2) -> dict:
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    classifier = Classifier(classifier_weights) if classifier_weights else None

    types = _load_types() if classifier is not None else {}
    visits = defaultdict(Counter)
    track_votes: dict[int, Counter] = defaultdict(Counter)   # track_id -> class votes
    last_flower: dict[int, str | None] = {}
    last_visit_frame: dict[tuple[int, str], int] = {}
    ftracker = FlowerTracker(flower_model, conf)
    writer = None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False, vid_stride=vid_stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        # re-detect flowers every `flower_interval` frames (they move slowly);
        # reuse the tracked boxes in between -> dynamic but ~interval× faster.
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        if fi == 0 and save_video:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / (Path(video).stem + "_annotated.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        boxes = res.boxes
        labels: dict[int, str] = {}
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for tid, b in zip(ids, xyxy):
                if classifier is not None:                    # majority vote per track
                    x1, y1, x2, y2 = map(int, b)
                    sp = classifier.predict(frame[y1:y2, x1:x2])
                    track_votes[tid][types.get(sp, sp)] += 1   # species id -> type
                    cls = track_votes[tid].most_common(1)[0][0]
                else:
                    cls = "insect"
                labels[tid] = cls
                cur = next((fid for fid, fb in flowers if _in(fb, _center(b))), None)
                if cur is not None and last_flower.get(tid) != cur:
                    if fi - last_visit_frame.get((tid, cur), -10**9) > debounce:
                        visits[cur]["total"] += 1
                        visits[cur][cls] += 1
                        last_visit_frame[(tid, cur)] = fi
                last_flower[tid] = cur
        if writer is not None:
            writer.write(_annotate(frame, flowers, res, labels, visits))
    if writer is not None:
        writer.release()

    for fid in ftracker.seen:                     # include flowers seen with 0 visits
        visits[fid]
    ctypes = sorted({t for v in visits.values() for t in v if t != "total"})
    rows = [{"flower_id": k, "total": v["total"],
             "pollinator": sum(v.get(t, 0) for t in POLLINATOR_TYPES),
             "non_pollinator": v["total"] - sum(v.get(t, 0) for t in POLLINATOR_TYPES),
             **{t: v.get(t, 0) for t in ctypes}}
            for k, v in sorted(visits.items())]
    csv_path = out_dir / (Path(video).stem + "_visits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "total", "pollinator", "non_pollinator"] + ctypes)
        w.writeheader(); w.writerows(rows)
    return {"video": Path(video).name, "flowers": len(flowers),
            "visits": {r["flower_id"]: r["total"] for r in rows}, "csv": str(csv_path)}


def _annotate(frame, flowers, res, labels, visits):
    for fid, (x1, y1, x2, y2) in flowers:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{visits[fid]['total']}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if res.boxes is not None and res.boxes.id is not None:
        for tid, b in zip(res.boxes.id.int().cpu().tolist(), res.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, b)
            cls = labels.get(tid, "insect")
            poll = cls in POLLINATOR_TYPES
            col = (0, 180, 255) if poll else (0, 0, 255)   # pollinator=orange, else red
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            tag = f"{cls}{' (pollinator)' if poll else ''} #{tid}"
            cv2.putText(frame, tag, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True)
    ap.add_argument("--classifier-weights", default="")
    ap.add_argument("--out", default=str(C.INTERIM_DIR / "cv_runs" / "visits"))
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--debounce", type=int, default=20)
    ap.add_argument("--flower-interval", type=int, default=5,
                    help="re-detect flowers every N frames (dynamic video)")
    ap.add_argument("--vid-stride", type=int, default=2,
                    help="process every Nth frame (lower effective fps, stabler tracks)")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits(args.video, args.flower_weights, args.insect_weights,
                                  args.classifier_weights, Path(args.out), args.conf,
                                  args.debounce, args.save_video, args.flower_interval,
                                  args.vid_stride), indent=2))


if __name__ == "__main__":
    main()
