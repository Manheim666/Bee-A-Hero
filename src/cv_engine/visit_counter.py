"""Flower-visit counting on video.

Pipeline (per the CV plan):
  1. Detect flowers on the first frame (static camera) -> each gets a stable ID
     ``flower_1, flower_2, ...``.
  2. Detect + classify + **track** insects across frames with BoT-SORT
     (pollinator / non_pollinator), giving each insect a track ID.
  3. Count a **visit** whenever a tracked insect enters a flower's bounding box
     (enter-transition, debounced so one continuous dwell = one visit and brief
     tracker flicker does not double-count). Re-entry after leaving = new visit.

Output:
  * ``visits.csv`` -> flower_id, total_visits, pollinator_visits, non_pollinator_visits
  * annotated ``.mp4`` (flowers + insect tracks + live counts) if --save-video

CLI:
    python -m src.cv_engine.visit_counter --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect/best.pt --save-video
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from src import config as C

INSECT_NAMES = {0: "pollinator", 1: "non_pollinator"}


def _center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in(box, pt) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def detect_flowers(model: YOLO, frame, conf: float, dilate: float = 0.15):
    """Return [(flower_id, (x1,y1,x2,y2))] from one frame; boxes dilated to a ROI."""
    res = model.predict(frame, conf=conf, verbose=False)[0]
    flowers = []
    H, W = frame.shape[:2]
    for i, b in enumerate(res.boxes.xyxy.cpu().numpy(), start=1):
        x1, y1, x2, y2 = b
        dw, dh = (x2 - x1) * dilate, (y2 - y1) * dilate      # dilate ROI (approach zone)
        flowers.append((f"flower_{i}",
                        (max(0, x1 - dw), max(0, y1 - dh),
                         min(W, x2 + dw), min(H, y2 + dh))))
    return flowers


def count_visits(video: str, flower_weights: str, insect_weights: str,
                 out_dir: Path, conf: float = 0.25, debounce: int = 20,
                 save_video: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)

    visits: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "pollinator": 0, "non_pollinator": 0})
    last_flower: dict[int, str | None] = {}
    last_visit_frame: dict[tuple[int, str], int] = {}
    flowers: list = []
    writer = None

    results = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                 persist=True, conf=conf, verbose=False)
    for fi, res in enumerate(results):
        frame = res.orig_img
        if fi == 0:
            flowers = detect_flowers(flower_model, frame, conf)
            if save_video:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / (Path(video).stem + "_annotated.mp4")),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))

        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for tid, cls, b in zip(ids, clss, xyxy):
                cur = next((fid for fid, fb in flowers if _point_in(fb, _center(b))), None)
                if cur is not None and last_flower.get(tid) != cur:
                    if fi - last_visit_frame.get((tid, cur), -10**9) > debounce:
                        cname = INSECT_NAMES.get(cls, "pollinator")
                        visits[cur]["total"] += 1
                        visits[cur][cname] += 1
                        last_visit_frame[(tid, cur)] = fi
                last_flower[tid] = cur

        if writer is not None:
            writer.write(_annotate(frame, flowers, res, visits))
    if writer is not None:
        writer.release()

    # ensure every detected flower appears, even with 0 visits
    for fid, _ in flowers:
        visits[fid]
    rows = [{"flower_id": k, **v} for k, v in sorted(visits.items())]
    csv_path = out_dir / (Path(video).stem + "_visits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "total", "pollinator", "non_pollinator"])
        w.writeheader(); w.writerows(rows)
    return {"video": video, "flowers": len(flowers),
            "visits": {r["flower_id"]: r["total"] for r in rows}, "csv": str(csv_path)}


def _annotate(frame, flowers, res, visits):
    for fid, (x1, y1, x2, y2) in flowers:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{visits[fid]['total']}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if res.boxes is not None and res.boxes.id is not None:
        for tid, cls, b in zip(res.boxes.id.int().cpu().tolist(),
                               res.boxes.cls.int().cpu().tolist(),
                               res.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, b)
            col = (255, 120, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{INSECT_NAMES.get(cls,'?')}#{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True)
    ap.add_argument("--out", default=str(C.INTERIM_DIR / "cv_runs" / "visits"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--debounce", type=int, default=20)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits(args.video, args.flower_weights, args.insect_weights,
                                  Path(args.out), args.conf, args.debounce, args.save_video),
                     indent=2))


if __name__ == "__main__":
    main()
