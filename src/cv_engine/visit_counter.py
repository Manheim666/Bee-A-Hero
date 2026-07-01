"""Flower-visit counting on video (two-stage insect recognition).

Pipeline:
  1. Detect flowers on the first frame (static camera); each gets a stable ID
     ``flower_1, flower_2, ...`` and its box is dilated into an approach ROI.
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


def detect_flowers(model, frame, conf, dilate=0.15):
    res = model.predict(frame, conf=conf, verbose=False)[0]
    H, W = frame.shape[:2]
    out = []
    for i, b in enumerate(res.boxes.xyxy.cpu().numpy(), start=1):
        x1, y1, x2, y2 = b
        dw, dh = (x2 - x1) * dilate, (y2 - y1) * dilate
        out.append((f"flower_{i}", (max(0, x1 - dw), max(0, y1 - dh),
                                    min(W, x2 + dw), min(H, y2 + dh))))
    return out


def count_visits(video, flower_weights, insect_weights, classifier_weights,
                 out_dir: Path, conf=0.25, debounce=20, save_video=False) -> dict:
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    classifier = Classifier(classifier_weights) if classifier_weights else None

    visits = defaultdict(lambda: {"total": 0, "pollinator": 0, "non_pollinator": 0})
    track_votes: dict[int, Counter] = defaultdict(Counter)   # track_id -> class votes
    last_flower: dict[int, str | None] = {}
    last_visit_frame: dict[tuple[int, str], int] = {}
    flowers, writer = [], None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        if fi == 0:
            flowers = detect_flowers(flower_model, frame, conf)
            if save_video:
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
                    track_votes[tid][classifier.predict(frame[y1:y2, x1:x2])] += 1
                    cls = track_votes[tid].most_common(1)[0][0]
                else:
                    cls = "pollinator"
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

    for fid, _ in flowers:
        visits[fid]
    rows = [{"flower_id": k, **v} for k, v in sorted(visits.items())]
    csv_path = out_dir / (Path(video).stem + "_visits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "total", "pollinator", "non_pollinator"])
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
            cls = labels.get(tid, "?")
            col = (255, 120, 0) if cls == "pollinator" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{cls}#{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True)
    ap.add_argument("--classifier-weights", default="")
    ap.add_argument("--out", default=str(C.INTERIM_DIR / "cv_runs" / "visits"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--debounce", type=int, default=20)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits(args.video, args.flower_weights, args.insect_weights,
                                  args.classifier_weights, Path(args.out), args.conf,
                                  args.debounce, args.save_video), indent=2))


if __name__ == "__main__":
    main()
