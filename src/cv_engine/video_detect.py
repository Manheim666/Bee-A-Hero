"""Flower-visit counting on video by **detection + tracking** (bbox, no segmentation).

Single real-time camera stream, resampled to a fixed **24 fps**. Per frame:

  1. Flowers: per-frame YOLO detection with stable IDs (``FlowerTracker``) -> a
     **separate** box per flower (never unified).
  2. Insects: multi-class YOLO26 detector (``bee, fly, beetle, bug, butterfly``)
     + BoT-SORT -> one **track ID + type** per insect. Each insect keeps its own
     colour (by track ID) so bee #1 and bee #2 are distinct.
  3. Type: taken directly from the detector, **confidence-weighted cumulative vote over
     the track's life** for stability (no separate classifier). The on-screen label adds
     hysteresis (a challenger must clearly out-weigh the current label to switch), so a
     brief mislabel -- e.g. bee flickering to fly for a few frames -- does not flip it.
  4. Landing: a contiguous span where an insect is on a flower -- either its box centre
     is inside a flower ROI (detected) **or** it is near-motionless with no ROI (inferred,
     stationary formula, catches undetected flowers). Enter/exit/duration are recorded;
     ``landing_s >= MIN_LAND_S`` (2s) = a *real* landing (feeding, not a fly-through).
     A fly-off + return **is** a new landing. Honeybees are split from other bees when a
     subclassifier is supplied (honeybee weighted ~10x for pollination value).

Outputs are grouped under ``test_video_result/``:
  * ``csv/<video>_landings.csv``        -> one row / landing episode (enter, exit, duration,
        type, is_honeybee, is_real_landing, flower_detected, pollination_weight, ...)
  * ``csv/<video>_flower_summary.csv``  -> one row / flower (per-type counts, total/mean dwell,
        pollination_score, species [NaN], timestamps [NaN for test videos])
  * ``csv/ALL_*.csv``                   -> merged tables across all videos (for the ML/LLM phase)
  * ``videos/<video>_annotated.mp4``    -> flower boxes + per-insect boxes/IDs + live counts

Setup (once):
    pip install -r src/cv_engine/requirements-cv.txt   # torch, ultralytics, opencv

CLI:
    python -m src.cv_engine.video_detect --video data/raw/Test_Video/clip.mp4 \
        --flower-weights .../flower/best.pt --insect-weights .../insect/best.pt --save-video
"""
from __future__ import annotations

import argparse
import csv
import colorsys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from src import config as C
from src.cv_engine.visit_counter import FlowerTracker, Classifier, _center, _in

TARGET_FPS = 24
POLLINATORS = {"honeybee", "bee", "butterfly", "fly"}   # rolled up in the CSV as "pollinator"

# --- landing semantics --------------------------------------------------------
INSECT_TYPES = ["honeybee", "bee", "fly", "beetle", "bug", "butterfly"]
MIN_LAND_S = 2.0          # dwell >= this = a *real* landing (feeding); shorter = fly-through
STATIONARY_TAU = 0.5      # normalised speed (insect-body-lengths/s) below which = settled
LAND_GRACE_S = 0.5        # bridge tracker flicker / brief exits inside one landing episode
LABEL_SWITCH_MARGIN = 1.5 # displayed label only switches when the leader's cumulative vote
                          #   weight is >= this * the current label's -> kills brief flips
INSECT_BOX_SMOOTH = 0.5   # EMA on the drawn insect box -> damps butterfly wing-flap size swings
INSECT_HOLD_MAX = 72      # keep drawing a lost insect box up to ~3s (24fps) so it doesn't blink;
                          #   holding stops earlier if the box reaches the frame edge (insect left)
INSECT_EDGE_FRAC = 0.02   # box within 2% of any frame border -> treat as left-frame, stop holding
MIN_TRACK_DRAW = 3        # a track must be detected in >= this many frames before its box is drawn
                          #   -> a 1-2 frame false blip (e.g. flower momentarily read as bee) never shows
MAX_INSECT_FRAME_FRAC = 0.18  # reject any insect box bigger than this fraction of the frame: a real
                          #   insect is small; a flower-sized box (whole flower read as fly) is a
                          #   false positive -> drop it from tracking, drawing and landings entirely
UNK_FLOWER_RADIUS = 2.0   # * sqrt(insect_area): attribute an inferred landing to a flower within,
                          #   else mint a synthetic flower_unk_N at that spot
SCORE_CAP_S = 30.0        # cap one landing's duration contribution to the pollination score
# pollination weight per landing (honeybee ~10x other bees for pollination value)
SPECIES_WEIGHT = {"honeybee": 10.0, "bee": 1.0, "butterfly": 2.0,
                  "fly": 0.5, "beetle": 0.5, "bug": 0.2}
DEFAULT_WEIGHT = 0.3


def _color(tid: int):
    """Deterministic distinct BGR colour per track ID (golden-ratio hue hop)."""
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _at_edge(box, W, H, frac):
    """True if the box hugs a frame border -> the insect has likely left the view."""
    m = frac * max(W, H)
    x1, y1, x2, y2 = box
    return x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m


def _speed_norm(prev_c, prev_t, c, t, area):
    """Insect centroid speed normalised by body size -> body-lengths/second (scale-free)."""
    dt = t - prev_t
    if dt <= 0 or area <= 0:
        return 0.0
    d = ((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2) ** 0.5
    return (d / dt) / (area ** 0.5)


def _attribute_flower(c, area, flowers, unk_reg, next_unk):
    """Assign a flower id to an *inferred* (no-ROI) landing at centre ``c``.

    Nearest current flower within R = UNK_FLOWER_RADIUS*sqrt(insect_area); else reuse a
    prior synthetic flower within R; else mint a fresh ``flower_unk_N``. Returns (fid, next_unk).
    """
    R = UNK_FLOWER_RADIUS * (area ** 0.5)
    best, bd = None, R
    for fid, box in flowers:
        fc = _center(box)
        d = ((c[0] - fc[0]) ** 2 + (c[1] - fc[1]) ** 2) ** 0.5
        if d <= bd:
            bd, best = d, fid
    if best is not None:
        return best, next_unk
    for uid, (ux, uy) in unk_reg.items():
        if ((c[0] - ux) ** 2 + (c[1] - uy) ** 2) ** 0.5 <= R:
            return uid, next_unk
    uid = f"flower_unk_{next_unk}"
    unk_reg[uid] = c
    return uid, next_unk + 1


def count_visits_det(video, flower_weights, insect_weights, out_dir: Path,
                     conf=0.25, flower_conf=0.15, save_video=False,
                     flower_interval=5, target_fps=TARGET_FPS,
                     honeybee_weights="", on_landing=None, live=False) -> dict:
    """Detect+track insects on flowers and emit landing-level pollination data.

    A **landing episode** = a contiguous span where a tracked insect is either inside a
    flower ROI (detected) *or* near-motionless with no ROI (inferred, stationary formula).
    Brief drop-outs < LAND_GRACE_S are bridged; leaving then returning is a *new* landing.
    Each episode yields enter/exit/duration; ``landing_s >= MIN_LAND_S`` marks a real landing.
    """
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir, csv_dir = out_dir / "videos", out_dir / "csv"   # group outputs: videos/ and csv/
    vid_dir.mkdir(exist_ok=True); csv_dir.mkdir(exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    names = insect_model.names                                 # {cls_id: type}
    subclf = Classifier(honeybee_weights) if honeybee_weights else None  # bee -> honeybee/bee

    in_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(in_fps / target_fps))
    out_fps = in_fps / stride

    votes: dict[int, Counter] = defaultdict(Counter)          # track -> conf-weighted type votes
    disp: dict[int, str] = {}                                 # track -> sticky display label (hysteresis)
    bee_votes: dict[int, Counter] = defaultdict(Counter)      # track -> honeybee/bee votes
    st: dict[int, dict] = {}                                   # track -> dwell state
    unk_reg: dict[str, tuple] = {}                            # synthetic flower id -> centre
    next_unk = 1
    landings: list[dict] = []
    flower_count: dict[str, int] = defaultdict(int)           # fid -> real landings (live overlay)
    ftracker = FlowerTracker(flower_model, flower_conf)
    writer = None
    t_s = 0.0
    vid_stem = Path(video).stem

    def finalize(tid):
        ep = st.get(tid, {}).get("ep")
        if not ep:
            return
        landing_s = round(ep["last_t"] - ep["enter_t"], 2)
        typ = votes[tid].most_common(1)[0][0] if votes[tid] else "insect"
        is_hb = ""                                             # "" = unknown (no subclassifier)
        if typ == "bee":
            if bee_votes[tid]:
                is_hb = bee_votes[tid].most_common(1)[0][0] == "honeybee"
                if is_hb:
                    typ = "honeybee"
        else:
            is_hb = False
        real = int(landing_s >= MIN_LAND_S)
        if real:
            flower_count[ep["flower"]] += 1
        # live cameras record the wall-clock exit time so daily counts can bucket by date;
        # test videos leave it blank (they have no real calendar time).
        ts = datetime.now().isoformat(timespec="seconds") if live else ""
        row = {
            "video": vid_stem, "flower_id": ep["flower"], "flower_species": "",
            "track_id": tid, "insect_type": typ, "is_honeybee": is_hb,
            "t_enter_s": round(ep["enter_t"], 2), "t_exit_s": round(ep["last_t"], 2),
            "landing_s": landing_s, "is_real_landing": real,
            "flower_detected": ep["detected"], "timestamp": ts,
            "pollination_weight": SPECIES_WEIGHT.get(typ, DEFAULT_WEIGHT),
            "conf_mean": round(ep["conf_sum"] / max(1, ep["conf_n"]), 3),
        }
        landings.append(row)
        if on_landing is not None:               # live sink: emit each landing as it completes
            on_landing(row)
        st[tid]["ep"] = None

    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, verbose=False, vid_stride=stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        H, W = frame.shape[:2]
        t_s = round(fi * stride / in_fps, 2)
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        if fi == 0 and save_video:
            writer = cv2.VideoWriter(str(vid_dir / (vid_stem + "_annotated.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        drawn = []
        present = set()
        b = res.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.int().cpu().tolist()
            confs = b.conf.cpu().numpy() if b.conf is not None else [0.0] * len(ids)
            for tid, box, c_id, cf in zip(ids, xyxy, cls, confs):
                if (box[2] - box[0]) * (box[3] - box[1]) > MAX_INSECT_FRAME_FRAC * W * H:
                    continue                                   # box too big to be an insect -> flower/bg FP
                present.add(tid)
                votes[tid][names[c_id]] += float(cf)           # confidence-weighted cumulative vote
                lead = votes[tid].most_common(1)[0][0]         # current top type by cumulative weight
                cur_lab = disp.get(tid)
                if cur_lab is None or (lead != cur_lab and
                        votes[tid][lead] >= votes[tid][cur_lab] * LABEL_SWITCH_MARGIN):
                    disp[tid] = lead                           # switch only on a clear margin (hysteresis)
                typ = disp[tid]
                if subclf is not None and typ == "bee":        # honeybee vs other-bee vote
                    x1, y1, x2, y2 = map(int, box)
                    bee_votes[tid][subclf.predict(frame[y1:y2, x1:x2])] += 1
                cen = _center(box)
                area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                sub = st.setdefault(tid, {"prev_c": cen, "prev_t": t_s, "ep": None,
                                          "n": 0, "dbox": None, "miss": 0, "dtyp": typ})
                sub["n"] += 1                                  # frames this track was actually detected
                s_norm = _speed_norm(sub["prev_c"], sub["prev_t"], cen, t_s, area)
                sub["prev_c"], sub["prev_t"] = cen, t_s
                cur = next((fid for fid, fb in flowers if _in(fb, cen)), None)
                settled = cur is not None or s_norm < STATIONARY_TAU
                if settled:
                    if cur is not None:
                        fid_use, det = cur, "detected"
                    else:                                       # inferred: stationary, no ROI
                        fid_use, next_unk = _attribute_flower(cen, area, flowers, unk_reg, next_unk)
                        det = "inferred"
                    ep = sub["ep"]
                    if ep is None:
                        sub["ep"] = {"flower": fid_use, "enter_t": t_s, "last_t": t_s,
                                     "detected": det, "conf_sum": float(cf), "conf_n": 1}
                    else:
                        ep["last_t"] = t_s
                        ep["conf_sum"] += float(cf); ep["conf_n"] += 1
                        if det == "detected" and ep["detected"] == "inferred":
                            ep["detected"] = "detected"; ep["flower"] = fid_use
                else:
                    ep = sub["ep"]
                    if ep is not None and t_s - ep["last_t"] > LAND_GRACE_S:
                        finalize(tid)
                # EMA-smooth the drawn box (damps butterfly wing-flap size swings), reset hold
                sub["dbox"] = tuple(float(v) for v in box) if sub["dbox"] is None else \
                    tuple(INSECT_BOX_SMOOTH * o + (1 - INSECT_BOX_SMOOTH) * v
                          for o, v in zip(sub["dbox"], box))
                sub["dtyp"], sub["miss"] = typ, 0
                if sub["n"] >= MIN_TRACK_DRAW:                 # persistence gate: skip transient false blips
                    drawn.append((tid, sub["dbox"], typ))
        # tracks that vanished this frame: finalize once past the grace window
        for tid, sub in list(st.items()):
            ep = sub.get("ep")
            if tid not in present and ep is not None and t_s - ep["last_t"] > LAND_GRACE_S:
                finalize(tid)
        # brief/long drop-out: keep drawing the last insect box until it leaves the frame or the
        # hold cap is hit -> box stays put instead of blinking/vanishing mid-scene
        for tid, sub in st.items():
            if (tid not in present and sub.get("dbox") is not None
                    and sub["n"] >= MIN_TRACK_DRAW and sub["miss"] < INSECT_HOLD_MAX
                    and not _at_edge(sub["dbox"], W, H, INSECT_EDGE_FRAC)):
                sub["miss"] += 1
                drawn.append((tid, sub["dbox"], sub["dtyp"]))
        if writer is not None:
            writer.write(_annotate(frame, flowers, drawn, flower_count))
    for tid in list(st.keys()):
        finalize(tid)
    if writer is not None:
        writer.release()

    # -- landings.csv : one row per landing episode -------------------------------
    land_fields = ["video", "flower_id", "flower_species", "track_id", "insect_type",
                   "is_honeybee", "t_enter_s", "t_exit_s", "landing_s", "is_real_landing",
                   "flower_detected", "timestamp", "pollination_weight", "conf_mean"]
    land_path = csv_dir / (vid_stem + "_landings.csv")
    with open(land_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=land_fields)
        w.writeheader(); w.writerows(landings)

    # -- flower_summary.csv : one row per flower ----------------------------------
    all_flowers = set(ftracker.seen) | set(unk_reg.keys()) | {l["flower_id"] for l in landings}
    summ = {fid: {"video": vid_stem, "flower_id": fid, "flower_species": "",
                  "n_landings": 0, "n_real_landings": 0,
                  **{f"n_{t}": 0 for t in INSECT_TYPES},
                  "total_landing_s": 0.0, "mean_landing_s": 0.0, "pollination_score": 0.0,
                  "timestamp_first": "", "timestamp_last": ""} for fid in sorted(all_flowers)}
    for l in landings:
        s = summ[l["flower_id"]]
        s["n_landings"] += 1
        if l["is_real_landing"]:
            s["n_real_landings"] += 1
            if l["insect_type"] in INSECT_TYPES:
                s[f"n_{l['insect_type']}"] += 1
            s["total_landing_s"] += l["landing_s"]
            s["pollination_score"] += l["pollination_weight"] * min(l["landing_s"], SCORE_CAP_S)
    for s in summ.values():
        if s["n_real_landings"]:
            s["mean_landing_s"] = round(s["total_landing_s"] / s["n_real_landings"], 2)
        s["total_landing_s"] = round(s["total_landing_s"], 2)
        s["pollination_score"] = round(s["pollination_score"], 2)
    summ_fields = ["video", "flower_id", "flower_species", "n_landings", "n_real_landings",
                   *[f"n_{t}" for t in INSECT_TYPES], "total_landing_s", "mean_landing_s",
                   "pollination_score", "timestamp_first", "timestamp_last"]
    summ_path = csv_dir / (vid_stem + "_flower_summary.csv")
    with open(summ_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=summ_fields)
        w.writeheader(); w.writerows(summ.values())

    real_total = sum(s["n_real_landings"] for s in summ.values())
    return {"video": Path(video).name, "flowers": len(all_flowers), "out_fps": round(out_fps, 1),
            "landings": len(landings), "real_landings": real_total,
            "landings_csv": str(land_path), "summary_csv": str(summ_path)}


def aggregate_csvs(out_dir: Path) -> dict:
    """Merge every per-video CSV into two team-friendly tables for the ML/LLM phase:

      * ``ALL_landings.csv``        -> video + full per-landing schema
      * ``ALL_flower_summary.csv``  -> video + per-flower rollup (counts, durations, score)
    """
    import glob
    csv_dir = Path(out_dir) / "csv"                          # per-video + ALL_*.csv live here
    outs = {}
    for kind, key in (("landings", "_landings.csv"), ("flower_summary", "_flower_summary.csv")):
        rows, fields = [], []
        for f in sorted(glob.glob(str(csv_dir / f"*{key}"))):
            if Path(f).name.startswith("ALL_"):
                continue
            for r in csv.DictReader(open(f)):
                for k in r:
                    if k not in fields:
                        fields.append(k)
                rows.append(r)
        dst = csv_dir / f"ALL_{kind}.csv"
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval=0)
            w.writeheader(); w.writerows(rows)
        outs[f"all_{kind}"] = str(dst)
    return outs


def _annotate(frame, flowers, drawn, flower_count):
    for tid, box, typ in drawn:                                # per-insect box + id + type
        x1, y1, x2, y2 = map(int, box)
        col = _color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{typ} #{tid}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    for fid, (x1, y1, x2, y2) in flowers:                      # separate box per flower
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"{fid}:{flower_count.get(fid, 0)}", (int(x1), int(y1) - 6),
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
    ap.add_argument("--honeybee-weights", default="",
                    help="optional honeybee-vs-other-bee subclassifier (relabels bee crops)")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()
    import json
    out = Path(args.out)

    def _run(v):
        return count_visits_det(str(v), args.flower_weights, args.insect_weights, out,
                                args.conf, args.flower_conf, args.save_video,
                                args.flower_interval, args.target_fps, args.honeybee_weights)

    vp = Path(args.video)
    if vp.is_dir():                                            # batch: whole folder -> out + ALL_*.csv
        vids = sorted(p for p in vp.iterdir()
                      if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
        if not vids:
            raise SystemExit(f"no videos found in {vp}")
        results = []
        for i, v in enumerate(vids, 1):
            print(f"[{i}/{len(vids)}] {v.name}", flush=True)
            results.append(_run(v))
        agg = aggregate_csvs(out)
        print(json.dumps({"processed": len(results), "aggregate": agg, "videos": results}, indent=2))
    else:
        print(json.dumps(_run(vp), indent=2))


if __name__ == "__main__":
    main()
