"""Streaming landing logger for the live camera.

The offline pipeline (`src/cv_engine/video_detect.count_visits_det`) sees a whole clip and
can stitch tracks in a second pass. A live feed is endless, so landings must be detected and
written **incrementally**: as an insect settles on a flower and later leaves, one row is
appended to a rolling CSV + JSON so the ML phase can consume live data the same way it
consumes `test_video_result/`.

Given per frame:
  * insect tracks: (track_id, box, type, conf)   -- track_id from BoT-SORT (persist=True)
  * flower boxes:  [box, ...]                      -- stabilised to sticky ids here (IoU match)

A **landing episode** = a contiguous span where an insect is on a flower (its centre inside a
flower ROI) or near-motionless (stationary formula, catches undetected flowers). Brief flicker
< ``grace_s`` is bridged. When the insect leaves (moves off / track vanishes past the grace),
the episode closes; ``landing_s >= min_land_s`` marks a *real* landing (a counted visit).
Each closed episode is appended to ``live_landings.csv`` and mirrored into ``live_landings.json``.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

POLLINATORS = {"honeybee", "bee", "butterfly", "fly"}

# re-link a track that vanished (occluded behind a petal) to one that reappears near the same
# spot within this window/radius -> the same bee is one visit, not two.
RELINK_MAX_S = 5.0
RELINK_RADIUS_K = 4.0

_FIELDS = [
    "timestamp", "t_enter_s", "t_exit_s", "landing_s", "is_real_landing",
    "flower_id", "track_id", "insect_type", "is_pollinator", "is_honeybee",
    "flower_detected", "conf_mean",
]


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _area(b):
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (_area(a) + _area(b) - inter)


def _contains(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# Inflate a flower box by this fraction ONLY when testing if an insect is 'on' it, so an insect on
# a petal edge still counts as a landing (the flower is still detected/drawn on its true box).
FLOWER_TOUCH_PAD = 0.15


def _pad(box, frac=FLOWER_TOUCH_PAD):
    w, h = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (box[0] - w, box[1] - h, box[2] + w, box[3] + h)


class FlowerRegistry:
    """Sticky-id flower boxes that cumulatively-average, so a static flower box is steady.

    Each flower's box is a heavy EMA (running average) of its detections. Two guards keep the
    drawn box honest against a live feed:
      * ``min_hits`` -- a flower must be detected on at least this many frames before it is
        emitted for drawing, so a single-frame false positive (e.g. the flower model firing on
        a passing insect/hand) never flashes a "flower out of nowhere".
      * ``forget_s`` -- a flower is dropped this many seconds after its last detection. Kept
        SHORT (just enough to bridge one dropped frame) so the box vanishes when the flower
        leaves the frame instead of lingering for several seconds."""

    def __init__(self, iou_thresh: float = 0.3, forget_s: float = 0.60, ema: float = 0.85,
                 min_hits: int = 2, memory_s: float = 1.5) -> None:
        self._iou = iou_thresh
        self._forget = forget_s          # stop DRAWING the box this long after its last detection
        self._memory = max(memory_s, forget_s)  # keep the record (hit count) this long, so a
        self._ema = ema                  #   flickering-but-present flower still reaches min_hits
        self._min_hits = min_hits        # frames a flower must be seen before it is drawn (kills a
        self._flowers: dict[str, dict] = {}   # 1-frame false positive) -> fid -> box/last_t/hits
        self._next = 1

    def update(self, boxes: list, t_s: float) -> list:
        used: set = set()
        for box in boxes:
            best, best_iou = None, self._iou
            for fid, v in self._flowers.items():
                if fid in used:
                    continue
                i = _iou(box, v["box"])
                if i >= best_iou:
                    best, best_iou = fid, i
            if best is None:
                best = f"flower_{self._next}"
                self._next += 1
                self._flowers[best] = {"box": tuple(map(float, box)), "last_t": t_s, "hits": 1}
            else:                              # cumulative average -> rock-steady box
                ob = self._flowers[best]["box"]
                self._flowers[best] = {
                    "box": tuple(self._ema * o + (1 - self._ema) * n for o, n in zip(ob, box)),
                    "last_t": t_s,
                    "hits": self._flowers[best]["hits"] + 1,
                }
            used.add(best)
        # Prune only after the (longer) memory window so a flower that flickers across frames keeps
        # accumulating hits and reaches min_hits. Emit for DRAWING only while both fresh (<= forget,
        # so the box vanishes quickly once the flower truly leaves) and confirmed (>= min_hits).
        out = []
        for fid, v in list(self._flowers.items()):
            age = t_s - v["last_t"]
            if age > self._memory:
                del self._flowers[fid]
                continue
            if age <= self._forget and v["hits"] >= self._min_hits:
                out.append((fid, v["box"]))
        return out


class LandingLogger:
    """Detects landing episodes from a live stream and appends them to CSV + JSON."""

    def __init__(self, out_dir: Path, min_land_s: float, grace_s: float,
                 stationary_tau: float) -> None:
        self._dir = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv = self._dir / "live_landings.csv"
        self._json = self._dir / "live_landings.json"
        self._min_land_s = min_land_s
        self._grace_s = grace_s
        self._tau = stationary_tau
        self._flowers = FlowerRegistry()
        self._state: dict[int, dict] = {}     # track_id -> episode/dwell state
        self._parked: list[dict] = []         # episodes whose track vanished (occlusion) awaiting re-link
        self._land_seq = 0                     # running id per written landing
        self._recent: list[dict] = []         # in-memory tail for /api/landings
        self._lock = threading.Lock()
        self._t0 = time.time()
        if not self._csv.exists():
            with open(self._csv, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_FIELDS).writeheader()

    # ------------------------------------------------------------------ per frame
    def observe(self, insect_tracks: list, flower_boxes: list) -> list:
        """Feed one frame. Returns the stabilised flowers [(fid, box)] for drawing."""
        t_s = round(time.time() - self._t0, 2)
        flowers = self._flowers.update(flower_boxes, t_s)

        # parked (occluded) episodes that never reappeared within the window -> write them
        keep = []
        for p in self._parked:
            if t_s - p["last_t"] > RELINK_MAX_S:
                self._write_ep(p["ep"], p["votes"])
            else:
                keep.append(p)
        self._parked = keep

        present: set = set()
        for tid, box, typ, conf in insect_tracks:
            present.add(tid)
            cen = _center(box)
            area = _area(box)
            if tid not in self._state:
                # new BoT-SORT id: if it appears where a track just vanished (behind a petal),
                # it is the SAME bee -> resume its episode instead of counting a new landing.
                carry = self._try_relink(cen, area)
                self._state[tid] = carry or {"ep": None, "votes": Counter(), "bee": Counter()}
                self._state[tid].update({"prev_c": cen, "prev_t": t_s})
            sub = self._state[tid]
            sub["votes"][typ] += float(conf)
            sub["last_area"] = area
            dt = t_s - sub["prev_t"]
            speed = 0.0
            if dt > 0:
                d = ((cen[0] - sub["prev_c"][0]) ** 2 + (cen[1] - sub["prev_c"][1]) ** 2) ** 0.5
                speed = (d / dt) / (area ** 0.5)
            sub["prev_c"], sub["prev_t"] = cen, t_s
            # A landing is ONLY an insect whose centre is inside a REAL detected flower's bbox.
            # (Previously a slow insect with no flower under it opened a synthetic "flower_unk"
            #  episode -> insects noted on flowers that were not there.) No flower under it =>
            #  not a landing, even if it is motionless.
            cur = next((fid for fid, fb in flowers if _contains(_pad(fb), cen)), None)
            if cur is not None:
                ep = sub["ep"]
                if ep is None:
                    sub["ep"] = {"flower": cur, "enter_t": t_s, "last_t": t_s,
                                 "detected": "detected", "conf_sum": float(conf), "conf_n": 1,
                                 "enter_wall": datetime.now().isoformat(timespec="seconds")}
                else:
                    ep["last_t"] = t_s
                    ep["conf_sum"] += float(conf); ep["conf_n"] += 1
                    ep["flower"] = cur                 # follow the flower the insect is on
            else:
                ep = sub["ep"]
                if ep is not None and t_s - ep["last_t"] > self._grace_s:
                    self._write_ep(ep, sub["votes"])   # moved off the flower -> a real landing end
                    sub["ep"] = None
        # vanished tracks: PARK an open episode (likely occlusion) for re-link; else drop stale
        for tid, sub in list(self._state.items()):
            if tid in present:
                continue
            ep = sub.get("ep")
            if ep is not None and t_s - ep["last_t"] > self._grace_s:
                self._parked.append({"ep": ep, "votes": sub["votes"],
                                     "last_c": sub["prev_c"], "last_t": t_s,
                                     "area": sub.get("last_area", 1.0)})
                del self._state[tid]
            elif ep is None and t_s - sub["prev_t"] > 5.0:
                del self._state[tid]
        return flowers

    def _try_relink(self, cen, area):
        """Adopt the nearest parked (occluded) episode within radius of `cen` -> same bee."""
        R = RELINK_RADIUS_K * (max(area, 1.0) ** 0.5)
        best, best_d = None, None
        for i, p in enumerate(self._parked):
            d = ((cen[0] - p["last_c"][0]) ** 2 + (cen[1] - p["last_c"][1]) ** 2) ** 0.5
            if d <= R and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            return None
        p = self._parked.pop(best)
        return {"ep": p["ep"], "votes": p["votes"], "bee": Counter()}

    def _write_ep(self, ep: dict, votes: Counter) -> None:
        if ep is None:
            return
        landing_s = round(ep["last_t"] - ep["enter_t"], 2)
        typ = votes.most_common(1)[0][0] if votes else "insect"
        self._land_seq += 1
        row = {
            "timestamp": ep["enter_wall"],
            "t_enter_s": ep["enter_t"], "t_exit_s": ep["last_t"], "landing_s": landing_s,
            "is_real_landing": int(landing_s >= self._min_land_s),
            "flower_id": ep["flower"], "track_id": self._land_seq, "insect_type": typ,
            "is_pollinator": typ.lower() in POLLINATORS, "is_honeybee": "",
            "flower_detected": ep["detected"],
            "conf_mean": round(ep["conf_sum"] / max(1, ep["conf_n"]), 3),
        }
        self._append(row)

    def _append(self, row: dict) -> None:
        with self._lock:
            with open(self._csv, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_FIELDS).writerow(row)
            self._recent.append(row)
            self._recent = self._recent[-200:]
            tmp = self._json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._recent, indent=2))
            tmp.replace(self._json)

    # ------------------------------------------------------------------ read side
    def snapshot(self) -> dict:
        with self._lock:
            real = sum(1 for r in self._recent if r["is_real_landing"])
            return {
                "total_landings": len(self._recent),
                "real_landings": real,
                "recent": self._recent[-25:][::-1],   # newest first
            }
