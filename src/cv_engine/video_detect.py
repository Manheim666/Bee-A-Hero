"""Flower-visit counting on video by **detection + tracking** (bbox, no segmentation).

Single real-time camera stream, resampled to a fixed **24 fps**. Per frame:

  1. Flowers: per-frame YOLO detection with stable IDs (``FlowerTracker``) -> a
     **separate** box per flower (never unified).
  2. Insects: multi-class YOLO26 detector (``bee, fly, beetle, bug, butterfly``)
     + BoT-SORT -> one **track ID + type** per insect. Each insect keeps its own
     colour (by track ID) so bee #1 and bee #2 are distinct.
  3. Type: taken directly from the detector, **majority-voted over the track's life**
     for stability (no separate classifier).
  4. Visit: counted when a tracked insect's box centre enters a flower box, each
     ``(insect, flower)`` pair **once ever** (a fly-off + return is not a new visit),
     stamped with a **timestamp**.

Outputs (to ``test_video_result/``):
  * ``<video>_visits.csv``    -> flower_id, total, <per type ...>
  * ``<video>_timeline.csv``  -> flower_id, track_id, type, t_enter_s  (one row / visit)
  * annotated ``<video>_annotated.mp4`` (flower boxes + per-insect boxes/IDs + counts)

CLI:
    python -m src.cv_engine.video_detect --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect/best.pt --save-video
"""
from __future__ import annotations

import argparse
import csv
import colorsys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src import config as C
from src.cv_engine.visit_counter import FlowerTracker, _center, _in

TARGET_FPS = 24
POLLINATORS = {"bee", "butterfly", "fly"}      # rolled up in the CSV as "pollinator"


def _color(tid: int):
    """Deterministic distinct BGR colour per track ID (golden-ratio hue hop)."""
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def count_visits_det(video, flower_weights, insect_weights, out_dir: Path,
                     conf=0.25, flower_conf=0.15, save_video=False,
                     flower_interval=5, target_fps=TARGET_FPS) -> dict:
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    names = insect_model.names                                 # {cls_id: type}

    in_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(in_fps / target_fps))
    out_fps = in_fps / stride

    visits = defaultdict(Counter)
    timeline: list[dict] = []
    votes: dict[int, Counter] = defaultdict(Counter)          # track_id -> type votes
    counted: set[tuple[int, str]] = set()                     # (track_id, flower_id) once
    ftracker = FlowerTracker(flower_model, flower_conf)
    writer = None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False, vid_stride=stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        H, W = frame.shape[:2]
        t_s = round(fi * stride / in_fps, 2)
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        if fi == 0 and save_video:
            writer = cv2.VideoWriter(str(out_dir / (Path(video).stem + "_annotated.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        drawn = []
        b = res.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.int().cpu().tolist()
            for tid, box, c in zip(ids, xyxy, cls):
                votes[tid][names[c]] += 1
                typ = votes[tid].most_common(1)[0][0]          # stable majority type
                cur = next((fid for fid, fb in flowers if _in(fb, _center(box))), None)
                if cur is not None and (tid, cur) not in counted:
                    counted.add((tid, cur))
                    visits[cur]["total"] += 1
                    visits[cur][typ] += 1
                    timeline.append({"flower_id": cur, "track_id": tid, "type": typ, "t_enter_s": t_s})
                drawn.append((tid, box, typ))
        if writer is not None:
            writer.write(_annotate(frame, flowers, drawn, visits))
    if writer is not None:
        writer.release()

    for fid in ftracker.seen:
        visits[fid]
    ctypes = sorted({t for v in visits.values() for t in v if t != "total"})
    rows = [{"flower_id": k, "total": v["total"],
             "pollinator": sum(v.get(t, 0) for t in POLLINATORS),
             **{t: v.get(t, 0) for t in ctypes}}
            for k, v in sorted(visits.items())]
    csv_path = out_dir / (Path(video).stem + "_visits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "total", "pollinator"] + ctypes)
        w.writeheader(); w.writerows(rows)
    tl = out_dir / (Path(video).stem + "_timeline.csv")
    with open(tl, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["flower_id", "track_id", "type", "t_enter_s"])
        w.writeheader(); w.writerows(timeline)
    return {"video": Path(video).name, "flowers": len(ftracker.seen), "out_fps": round(out_fps, 1),
            "visits": {r["flower_id"]: r["total"] for r in rows}, "csv": str(csv_path), "timeline": str(tl)}


def aggregate_csvs(out_dir: Path) -> dict:
    """Merge every per-video CSV into two team-friendly tables the ML/LLM team can fetch:

      * ``ALL_visits.csv``   -> video, flower_id, total, pollinator, <types...>
      * ``ALL_timeline.csv`` -> video, flower_id, track_id, type, t_enter_s
    """
    import glob
    out_dir = Path(out_dir)
    for kind, key in (("visits", "_visits.csv"), ("timeline", "_timeline.csv")):
        rows, fields = [], []
        for f in sorted(glob.glob(str(out_dir / f"*{key}"))):
            if Path(f).name.startswith("ALL_"):
                continue
            video = Path(f).name[: -len(key)]
            for r in csv.DictReader(open(f)):
                for k in r:
                    if k not in fields:
                        fields.append(k)
                rows.append({"video": video, **r})
        dst = out_dir / f"ALL_{kind}.csv"
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["video"] + fields, restval=0)
            w.writeheader(); w.writerows(rows)
    return {"all_visits": str(out_dir / "ALL_visits.csv"),
            "all_timeline": str(out_dir / "ALL_timeline.csv")}


def _annotate(frame, flowers, drawn, visits):
    for tid, box, typ in drawn:                                # per-insect box + id + type
        x1, y1, x2, y2 = map(int, box)
        col = _color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{typ} #{tid}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    for fid, (x1, y1, x2, y2) in flowers:                      # separate box per flower
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{visits[fid]['total']}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--flower-weights", required=True)
    ap.add_argument("--insect-weights", required=True, help="multi-class YOLO26 insect detector")
    ap.add_argument("--out", default=str(C.REPO_ROOT / "test_video_result"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--flower-conf", type=float, default=0.15)
    ap.add_argument("--flower-interval", type=int, default=5)
    ap.add_argument("--target-fps", type=int, default=TARGET_FPS)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(count_visits_det(args.video, args.flower_weights, args.insect_weights,
                                      Path(args.out), args.conf, args.flower_conf,
                                      args.save_video, args.flower_interval, args.target_fps), indent=2))


if __name__ == "__main__":
    main()
