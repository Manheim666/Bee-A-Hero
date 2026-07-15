"""Flower-visit counting on video by **detection + tracking** (bbox, no segmentation).

Single real-time camera stream, resampled to a fixed **20 fps**. Pipeline is **two-pass**
so a momentary detector drop-out never becomes a phantom visit and the drawn box glides
smoothly across the gap instead of blinking or snapping:

  Pass 1 (track + collect)
  ------------------------
  1. Flowers: per-frame YOLO detection with stable IDs (``FlowerTracker``) -> a
     **separate** box per flower (never unified).
  2. Insects: multi-class YOLO26 detector (``bee, fly, beetle, bug, butterfly``)
     + BoT-SORT -> one **raw track ID + type** per insect per detected frame. Nothing is
     drawn or counted yet -- every detection is just recorded (box, class, conf, frame idx).
  3. Type votes: confidence-weighted cumulative vote per raw track. Honeybee-vs-bee crops
     are scored by the optional subclassifier as they are seen.

  Stitch + interpolate (offline, between passes)
  ----------------------------------------------
  4. **Stitch**: BoT-SORT loses/re-mints an ID whenever an insect is briefly occluded or the
     detector blinks. Two raw tracks are merged into one *unified* track when the later one
     starts within ``RELINK_MAX_S`` (~3 s) of the earlier one ending **and** its first centre
     is within ``RELINK_RADIUS_K * sqrt(area)`` of where the earlier one vanished (the insect
     did not really move -> same insect). This is what stops one visit being counted twice.
  5. **Interpolate**: every gap inside a unified track is filled by linearly dragging the box
     corner-to-corner from the vanish point to the reappear point -> the box glides across the
     gap. A mild EMA damps butterfly wing-flap size swings.

  Landings (on the stitched, gap-filled timeline)
  -----------------------------------------------
  6. A **landing episode** = a contiguous span where an insect is on a flower -- box centre
     inside a flower ROI (detected) **or** near-motionless with no ROI (inferred, stationary
     formula, catches undetected flowers). A bridged occlusion stays **one** episode (no more
     phantom split). A real *fly-off* (the insect moves away and the track keeps moving, or it
     returns later/elsewhere beyond the stitch window) is still a **new** landing.
     ``landing_s >= MIN_LAND_S`` (2 s) = a *real* landing (feeding, not a fly-through).

  Pass 2 (render, only with --save-video)
  ---------------------------------------
  7. Re-decode the video (no inference) and draw the interpolated unified boxes + IDs + live
     counts + flower boxes. Written with a browser-playable H.264 codec when available.

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

TARGET_FPS = 20          # resample rate; lower than source -> smoother tracks + cheaper
POLLINATORS = {"honeybee", "bee", "butterfly", "fly"}   # rolled up in the CSV as "pollinator"

# --- landing semantics --------------------------------------------------------
INSECT_TYPES = ["honeybee", "bee", "fly", "beetle", "bug", "butterfly"]
MIN_LAND_S = 2.0          # dwell >= this = a *real* landing (feeding); shorter = fly-through
STATIONARY_TAU = 0.5      # normalised speed (insect-body-lengths/s) below which = settled
LAND_GRACE_S = 0.5        # bridge tracker flicker / brief exits inside one landing episode
LABEL_SWITCH_MARGIN = 1.5 # displayed label only switches when the leader's cumulative vote
                          #   weight is >= this * the runner-up's -> kills brief flips

# --- track stitching (kills phantom double-counts across detector drop-outs) ---
RELINK_MAX_S = 5.0        # a track that reappears within this many seconds of vanishing ...
RELINK_RADIUS_K = 4.0     #   ... and within RELINK_RADIUS_K * sqrt(area) of the vanish point is
                          #   the SAME insect (occlusion behind a petal / detector blink), not a new
                          #   one. Merge the two raw tracks -> one unified track, one visit, box
                          #   interpolated across the gap. A real fly-off returns far / late -> stays
                          #   separate. Window+radius are generous so a bee dipping behind a petal
                          #   for a few seconds is never counted twice.

# --- flower persistence (flowers are static -> hold + cumulatively average boxes) ---
FLOWER_BOX_EMA = 0.85     # heavy EMA on a flower's box: a static flower's position is a running
                          #   (cumulative) average of its detections -> a rock-steady box.
FLOWER_TOUCH_PAD = 0.15   # inflate a flower box by this fraction ONLY when testing if an insect is
                          #   'on' it -> an insect on a petal edge still counts as a landing, while
                          #   the flower is detected/gated/drawn on its true (tight) box
FLOWER_HOLD_S = 0.8       # bridge a couple of MISSED detections so a static flower's box doesn't
                          #   flicker -- but drop it promptly once the flower is truly gone (no
                          #   multi-second "ghost" box lingering after the flower leaves).

# --- drawing ------------------------------------------------------------------
INSECT_BOX_SMOOTH = 0.6   # EMA on the drawn (interpolated) box -> damps wing-flap size swings
MIN_TRACK_DRAW = 2        # a unified track needs >= this many *detected* frames before it is
                          #   drawn or counted -> a 1-frame false blip never shows, but a real
                          #   insect (e.g. a bee already on the flower) is drawn almost immediately
MAX_INSECT_FRAME_FRAC = 0.18  # reject any insect box bigger than this fraction of the frame: a real
                          #   insect is small; a flower-sized box (whole flower read as fly) is a
                          #   false positive -> drop it from tracking, drawing and landings entirely
INSECT_FLOWER_IOU = 0.80  # an insect box matching a flower box this closely IS the flower (the whole
                          #   flower mislabelled as one insect). Kept high on purpose: several bees
                          #   covering much of a flower each have IoU < this, so they are NOT vetoed.
INSECT_MAX_ASPECT = 4.0   # insect box longer:shorter side above this = a sliver/edge, not an insect
FLOWER_MAX_ASPECT = 3.0   # flower box aspect above this = a sliver/random object, not a flower
MIN_BOX_FRAC = 0.0006     # any box smaller than this fraction of the frame = noise -> reject
MAX_FLOWER_FRAC = 0.90    # reject only a near-frame-filling box (whole-scene greenery/wall). A
                          #   flower shot close to the camera legitimately fills most of the frame,
                          #   so keep it -> matches the live viewer's flower_max_frac (0.92) and
                          #   stops close-up upload clips returning 0 flowers / 0 visits.
FLOWER_NMS_IOU = 0.45     # two flower boxes overlapping this much are one flower -> keep one
                          #   (kills "2 flowers for 1" and boxes stacked on the same bloom)
FLOWER_MERGE_K = 1.6      # canonicalise flowers by CENTRE distance (robust to box-size jitter):
                          #   detections whose centres are within this * sqrt(area) are the same
                          #   bloom. Generous, so one jittery flower stays one id, not several.
UNK_FLOWER_RADIUS = 3.5   # * sqrt(insect_area): attribute an inferred landing to a flower within,
                          #   else mint a synthetic flower_unk_N. Generous so a bee crawling around
                          #   one bloom does not spawn several synthetic flowers.
FLOWER_MIN_SPAN_S = 1.0   # a detected flower must be present at least this long to count as a real
                          #   flower (drops brief background false positives that flicker in)
FLOWER_CLUSTER_FRAC = 0.16  # flowers whose centres are within this fraction of the frame diagonal
                          #   are one bloom -> a bee crawling across a flower and landing at a few
                          #   spots is counted as a single flower, not several
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


CONCURRENT_MERGE_IOU = 0.45   # two tracks that co-exist on the same frames and overlap this much
                              #   are one insect with two BoT-SORT ids -> merge (no duplicate box)
DRAW_NMS_IOU = 0.45           # at draw time, never show two insect boxes overlapping more than this
                              #   -> kills "multiple bboxes" on one bug in the annotated video


def _person_veto(box, persons, iou_thr) -> bool:
    """True if `box` coincides with a COCO person box (IoU >= iou_thr) -> a human misread as a
    flower/insect. IoU-based (not containment) so a small object held by a person is kept."""
    if not persons:
        return False
    return any(_iou_box(box, p) >= iou_thr for p in persons)


def _pad_box(box, frac):
    """Expand a box by `frac` of its size on each side. Used ONLY to test whether an insect is
    'on' a flower — the flower is detected/drawn on its true (tight) box, but an insect crawling
    on a petal edge has its centre just outside that box, so containment uses a padded copy."""
    w, h = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (box[0] - w, box[1] - h, box[2] + w, box[3] + h)


def _box_area(box):
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def _iou_box(a, b):
    ox1, oy1 = max(a[0], b[0]), max(a[1], b[1])
    ox2, oy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
    if inter <= 0:
        return 0.0
    return inter / (_box_area(a) + _box_area(b) - inter)


def _cluster_count(points, radius):
    """Greedy spatial cluster count: points within `radius` of a kept representative merge."""
    reps = []
    for p in points:
        if all(((p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2) ** 0.5 >= radius for r in reps):
            reps.append(p)
    return len(reps)


def _aspect(box):
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    return max(w / h, h / w)


def _plausible_insect(box):
    """Geometric sanity: an insect box is not an extreme sliver (edge/artefact)."""
    return _aspect(box) <= INSECT_MAX_ASPECT


def _plausible_flower(box, frame_area):
    """Geometric sanity for a flower box: roughly compact, not noise, not the whole frame."""
    a = _box_area(box)
    return MIN_BOX_FRAC * frame_area <= a <= MAX_FLOWER_FRAC * frame_area and _aspect(box) <= FLOWER_MAX_ASPECT


def _is_flower_box(ibox, flowers):
    """True if an insect box is really a flower (high IoU / near flower size with overlap).

    A genuine insect sitting on a flower is a small box inside it -> low IoU -> passes.
    NOTE: the area-based branch is intentionally conservative (needs high IoU too) so two
    bees covering much of one flower are NOT mistaken for the flower itself."""
    ia = _box_area(ibox)
    for _fid, fb in flowers:
        fa = _box_area(fb)
        ox1, oy1 = max(ibox[0], fb[0]), max(ibox[1], fb[1])
        ox2, oy2 = min(ibox[2], fb[2]), min(ibox[3], fb[3])
        inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        if inter <= 0:
            continue
        if inter / (ia + fa - inter) >= INSECT_FLOWER_IOU:
            return True                              # box ~ the whole flower, not an insect on it
    return False


def _dedup_insects(cands):
    """Drop duplicate/nested insect boxes on the same bug (an "insect inside an insect").

    ``cands``: [(tid, box, conf, cls_name)]. Class-aware NMS keeps a bee-box and an overlapping
    butterfly-box on one insect; this class-agnostic pass keeps only the highest-confidence of
    any heavily-overlapping / contained set. Returns the kept items in original order."""
    order = sorted(range(len(cands)), key=lambda i: cands[i][2], reverse=True)
    kept_boxes, keep = [], []
    for i in order:
        box = cands[i][1]
        ia = _box_area(box)
        drop = False
        for kb in kept_boxes:
            ox1, oy1 = max(box[0], kb[0]), max(box[1], kb[1])
            ox2, oy2 = min(box[2], kb[2]), min(box[3], kb[3])
            inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            if inter <= 0:
                continue
            ka = _box_area(kb)
            if inter / (ia + ka - inter) >= 0.6 or inter / min(ia, ka) >= 0.75:
                drop = True
                break
        if not drop:
            kept_boxes.append(box)
            keep.append(i)
    return [cands[i] for i in sorted(keep)]


def _lerp_box(a, b, w):
    """Linear blend of two boxes, w in [0,1] (0 -> a, 1 -> b)."""
    return tuple(a[i] * (1 - w) + b[i] * w for i in range(4))


def _to_browser_mp4(src: Path, dst: Path) -> bool:
    """Transcode to H.264 + yuv420p + faststart so browsers can play the <video>.

    OpenCV's bundled FFMPEG often lacks an H.264 encoder (writes only mp4v, which Chrome
    won't play), so we render with mp4v then re-encode with the system ffmpeg if present.
    Returns True on success; on any failure the caller keeps the mp4v file.
    """
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             # libx264 + yuv420p needs EVEN width/height; force it so odd-sized clips
             # (e.g. many "_medium" videos) still transcode instead of falling back to mp4v.
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-movflags", "+faststart", "-loglevel", "error", str(dst)],
            check=True,
        )
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def _stitch_tracks(raw_tracks, out_fps):
    """Merge raw BoT-SORT tracks that a drop-out split into one *unified* track.

    ``raw_tracks``: {raw_tid: {"det": {fi: (box, conf, cls_name)}, "bee": Counter}}.
    Greedy chaining by end-time then start-time: a later track chains onto an earlier one
    when it starts within ``RELINK_MAX_S`` of the earlier ending AND its first centre is
    within ``RELINK_RADIUS_K * sqrt(area)`` of the earlier one's last centre. Returns a list
    of unified tracks, each a merged dict of the same shape (dets keyed by frame index).
    """
    max_gap = RELINK_MAX_S * out_fps
    # summarise each raw track by its first/last detected frame, centre and size
    info = {}
    for rid, tr in raw_tracks.items():
        fis = sorted(tr["det"])
        if not fis:
            continue
        f0, f1 = fis[0], fis[-1]
        b0, b1 = tr["det"][f0][0], tr["det"][f1][0]
        info[rid] = {"f0": f0, "f1": f1, "c0": _center(b0), "c1": _center(b1),
                     "area1": _box_area(b1)}
    order = sorted(info, key=lambda r: info[r]["f0"])   # by start frame
    parent = {r: r for r in order}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    # merge tracks that co-exist on the same frames and coincide spatially — BoT-SORT gave one
    # insect two ids (e.g. a bee-box and an overlapping butterfly-box). One bug -> one track,
    # so the annotated video draws a single box and the visit is counted once.
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if find(a) == find(b):
                continue
            da, db = raw_tracks[a]["det"], raw_tracks[b]["det"]
            common = set(da) & set(db)
            if len(common) < 2:
                continue
            miou = sum(_iou_box(da[f][0], db[f][0]) for f in common) / len(common)
            if miou >= CONCURRENT_MERGE_IOU:
                parent[find(a)] = find(b)

    # for each track, try to attach it to the best earlier track that just ended nearby
    ends = sorted(order, key=lambda r: info[r]["f1"])   # by end frame
    for later in order:
        li = info[later]
        best, best_gap = None, None
        for earlier in ends:
            if earlier == later:
                continue
            ei = info[earlier]
            gap = li["f0"] - ei["f1"]
            if gap < 0 or gap > max_gap:
                continue
            if find(earlier) == find(later):
                continue
            radius = RELINK_RADIUS_K * (max(ei["area1"], li["area1"]) ** 0.5)
            d = ((li["c0"][0] - ei["c1"][0]) ** 2 + (li["c0"][1] - ei["c1"][1]) ** 2) ** 0.5
            if d <= radius and (best_gap is None or gap < best_gap):
                best, best_gap = earlier, gap
        if best is not None:
            parent[find(later)] = find(best)

    groups: dict = defaultdict(list)
    for r in order:
        groups[find(r)].append(r)

    unified = []
    for members in groups.values():
        det: dict = {}
        bee = Counter()
        votes = Counter()
        for rid in members:
            tr = raw_tracks[rid]
            bee.update(tr["bee"])
            for fi, (box, conf, cls_name) in tr["det"].items():
                votes[cls_name] += float(conf)
                if fi in det and det[fi][1] >= conf:
                    continue                      # rare overlap in a chain -> keep higher-conf box
                det[fi] = (box, conf, cls_name)
        unified.append({"det": det, "bee": bee, "votes": votes})
    return unified


def _interpolate(det):
    """Fill every gap between detected keyframes with linearly-dragged boxes.

    ``det``: {fi: (box, conf, cls_name)}. Returns {fi: box} for every fi in [first, last],
    lightly EMA-smoothed for draw stability. Detected frames keep near-exact boxes.
    """
    keys = sorted(det)
    dense: dict = {}
    for k0, k1 in zip(keys, keys[1:]):
        b0, b1 = det[k0][0], det[k1][0]
        span = k1 - k0
        for fi in range(k0, k1):
            dense[fi] = _lerp_box(b0, b1, (fi - k0) / span)
    dense[keys[-1]] = det[keys[-1]][0]
    # mild EMA over the dense sequence to damp wing-flap size swings
    smooth: dict = {}
    prev = None
    for fi in range(keys[0], keys[-1] + 1):
        box = dense[fi]
        if prev is None:
            prev = tuple(float(v) for v in box)
        else:
            prev = tuple(INSECT_BOX_SMOOTH * p + (1 - INSECT_BOX_SMOOTH) * v
                         for p, v in zip(prev, box))
        smooth[fi] = prev
    return smooth


class FlowerPersistence:
    """Canonicalise flower detections into one stable id per bloom.

    Flowers are static, so every detection near the same spot is the SAME flower even when the
    raw tracker churns its id. Each detection is snapped to a canonical cluster (by IoU); the
    cluster box is a heavy EMA (running average) and is held for FLOWER_HOLD_S after its last
    detection. Result: a static flower never flickers, gets one steady box, and is counted once
    (no "2 flowers for 1"). Two genuinely separate blooms stay two canonicals."""

    def __init__(self, out_fps):
        self._hold = FLOWER_HOLD_S * max(1.0, out_fps)
        self._canon: dict = {}                     # cid -> {"box": ema_box, "last": fi}
        self._next = 1
        self._kept: set = set()                    # canonical ids ever emitted (real flowers)
        self._life: dict = {}                      # cid -> [first_fi, last_fi] (persists after expiry)

    @property
    def kept_ids(self) -> set:
        return self._kept

    def stable_ids(self, min_span_frames) -> set:
        """Canonical flowers present long enough to be real (drops brief background FPs)."""
        return {cid for cid, v in self._life.items() if v[1] - v[0] >= min_span_frames}

    def centroids(self) -> dict:
        """cid -> (cx, cy) for every canonical flower ever seen (for spatial clustering)."""
        return {cid: v[2] for cid, v in self._life.items()}

    def update(self, flowers, fi):
        used: set = set()
        for _fid, box in flowers:                  # raw tracker id is ignored — location is identity
            box = tuple(float(v) for v in box)
            cen = _center(box)
            R = FLOWER_MERGE_K * (_box_area(box) ** 0.5)
            best, best_d = None, R
            for cid, r in self._canon.items():
                if cid in used:
                    continue
                rc = _center(r["box"])
                d = ((cen[0] - rc[0]) ** 2 + (cen[1] - rc[1]) ** 2) ** 0.5
                if d <= best_d:                    # nearest canonical within the merge radius
                    best, best_d = cid, d
            if best is None:
                cid = f"flower_{self._next}"; self._next += 1
                self._canon[cid] = {"box": box, "last": fi}
                self._life[cid] = [fi, fi, cen]
            else:
                cid = best
                r = self._canon[cid]
                r["box"] = tuple(FLOWER_BOX_EMA * o + (1 - FLOWER_BOX_EMA) * n
                                 for o, n in zip(r["box"], box))
                r["last"] = fi
                self._life[cid][1] = fi
                self._life[cid][2] = _center(r["box"])
            used.add(cid)
        out = []
        for cid, r in list(self._canon.items()):
            if fi - r["last"] > self._hold:
                del self._canon[cid]               # gone far longer than a plausible occlusion
                continue
            out.append((cid, r["box"]))
        self._kept.update(cid for cid, _ in out)
        return out


def count_visits_det(video, flower_weights, insect_weights, out_dir: Path,
                     conf=0.20, flower_conf=0.20, save_video=False,
                     flower_interval=5, target_fps=TARGET_FPS,
                     honeybee_weights="", on_landing=None, live=False,
                     insect_imgsz=768, person_veto_iou=0.0) -> dict:
    """Detect+track insects on flowers and emit landing-level pollination data.

    Two-pass: track+collect, then stitch drop-out-split tracks into unified tracks (so one
    occluded insect is one visit, not several), interpolate the box across every gap, and
    derive landing episodes from that gap-filled timeline. A bridged occlusion is one episode;
    a genuine fly-off + return (far / beyond RELINK_MAX_S) is a new episode. ``landing_s >=
    MIN_LAND_S`` marks a real landing. Renders the annotated mp4 in a second decode pass.
    """
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir, csv_dir = out_dir / "videos", out_dir / "csv"   # group outputs: videos/ and csv/
    vid_dir.mkdir(exist_ok=True); csv_dir.mkdir(exist_ok=True)
    flower_model, insect_model = YOLO(flower_weights), YOLO(insect_weights)
    names = insect_model.names                                 # {cls_id: type}
    # Optional COCO person veto (web path): drop flower/insect boxes that ARE a human. Off for
    # CLI (person_veto_iou=0) so offline test_video_result reproduction is byte-identical.
    person_model = None
    if person_veto_iou and person_veto_iou > 0:
        try:
            person_model = YOLO("yolov8n.pt")                  # auto-downloads if absent
        except Exception:
            person_model = None
    subclf = Classifier(honeybee_weights) if honeybee_weights else None  # bee -> honeybee/bee

    in_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(in_fps / target_fps))
    out_fps = in_fps / stride
    vid_stem = Path(video).stem

    ftracker = FlowerTracker(flower_model, flower_conf)
    flower_persist = FlowerPersistence(out_fps)               # hold+average static flower boxes
    unk_reg: dict[str, tuple] = {}                            # synthetic flower id -> centre
    next_unk = 1

    # ---- Pass 1: track + collect (draw nothing, count nothing yet) --------------
    raw_tracks: dict[int, dict] = defaultdict(lambda: {"det": {}, "bee": Counter()})
    flowers_by_fi: dict[int, list] = {}                      # fi -> [(fid, box)] snapshot
    n_frames = 0
    frame_diag = 1.0                                         # set from the first frame's size
    # Infer at the detector's TRAIN size (768). Measured: inferring above it (1024) actually
    # dropped detections and over-merged/under-detected, and 640 over-fragmented one bee into
    # many track ids; 768 gives the cleanest tracks -> most accurate visit counts. Settable arg.
    stream = insect_model.track(source=video, stream=True, tracker="botsort.yaml",
                                persist=True, conf=conf, imgsz=insect_imgsz,
                                verbose=False, vid_stride=stride)
    for fi, res in enumerate(stream):
        frame = res.orig_img
        H, W = frame.shape[:2]
        frame_diag = (W * W + H * H) ** 0.5
        persons = []
        if person_model is not None:
            for pr in person_model.predict(frame, conf=0.35, classes=[0], verbose=False):
                if pr.boxes is None:
                    continue
                for pb in pr.boxes.xyxy.cpu().numpy():
                    persons.append(tuple(float(v) for v in pb))
        flowers = ftracker.update(frame) if fi % flower_interval == 0 else ftracker.current()
        flowers = [(fid, fb) for fid, fb in flowers if _plausible_flower(fb, W * H)]
        flowers = [(fid, fb) for fid, fb in flowers if not _person_veto(fb, persons, person_veto_iou)]
        flowers = flower_persist.update(flowers, fi)          # hold + average -> no flicker/disappear
        flowers_by_fi[fi] = list(flowers)
        n_frames = fi + 1
        b = res.boxes
        if b is None or b.id is None:
            continue
        ids = b.id.int().cpu().tolist()
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.int().cpu().tolist()
        confs = b.conf.cpu().numpy() if b.conf is not None else [0.0] * len(ids)
        cands = []
        for tid, box, c_id, cf in zip(ids, xyxy, cls, confs):
            if (box[2] - box[0]) * (box[3] - box[1]) > MAX_INSECT_FRAME_FRAC * W * H:
                continue                                       # box too big to be an insect -> FP
            box = tuple(float(v) for v in box)
            if _person_veto(box, persons, person_veto_iou):
                continue                                       # a human mislabelled as an insect
            if _is_flower_box(box, flowers):
                continue                                       # a flower mislabelled as an insect
            if not _plausible_insect(box):
                continue                                       # implausible shape -> not an insect
            cands.append((tid, box, float(cf), names[c_id]))
        # drop duplicate/nested boxes on one bug (an "insect inside an insect")
        for tid, box, cf, cls_name in _dedup_insects(cands):
            raw_tracks[tid]["det"][fi] = (box, cf, cls_name)
            if subclf is not None and cls_name == "bee":       # honeybee vs other-bee vote
                x1, y1, x2, y2 = map(int, box)
                raw_tracks[tid]["bee"][subclf.predict(frame[y1:y2, x1:x2])] += 1

    # ---- Stitch raw tracks split by drop-outs into unified tracks ---------------
    unified = _stitch_tracks(raw_tracks, out_fps)

    # ---- Derive landing episodes per unified track on the gap-filled timeline ---
    landings: list[dict] = []
    flower_events: list[tuple] = []            # (exit_fi, flower_id) for real landings -> live overlay
    draw_tracks: list[dict] = []               # per unified track: {uid, typ, boxes:{fi:box}}
    uid_seq = 0
    for u in unified:
        det = u["det"]
        if len(det) < MIN_TRACK_DRAW:          # persistence gate: drop transient false blips
            continue
        uid_seq += 1
        uid = uid_seq
        dense = _interpolate(det)              # {fi: box} across [first, last]
        typ = u["votes"].most_common(1)[0][0] if u["votes"] else "insect"
        is_hb = ""
        if typ == "bee" and u["bee"]:
            is_hb = u["bee"].most_common(1)[0][0] == "honeybee"
            if is_hb:
                typ = "honeybee"
        elif typ != "bee":
            is_hb = False
        draw_tracks.append({"uid": uid, "typ": typ, "boxes": dense})

        fis = sorted(dense)
        ep = None
        prev_c, prev_t = None, None
        for fi in fis:
            box = dense[fi]
            cen = _center(box)
            area = _box_area(box)
            t_s = round(fi * stride / in_fps, 2)
            flowers = flowers_by_fi.get(fi, [])
            s_norm = 0.0 if prev_c is None else _speed_norm(prev_c, prev_t, cen, t_s, area)
            prev_c, prev_t = cen, t_s
            # A landing is ONLY an insect whose centre is inside a REAL detected flower box.
            # (Dropped the old "or stationary" branch: a motionless bee with no flower under it
            #  used to mint a synthetic flower_unk -> a bee auto-assuming a flower that isn't
            #  there. No flower detected => no landing, no phantom flower.)
            cur = next((fid for fid, fb in flowers if _in(_pad_box(fb, FLOWER_TOUCH_PAD), cen)), None)
            conf_here = det.get(fi, (None, 0.0, None))[1]
            if cur is not None:
                if ep is None:
                    ep = {"flower": cur, "enter_t": t_s, "last_t": t_s,
                          "detected": "detected", "conf_sum": conf_here, "conf_n": 1,
                          "last_fi": fi}
                else:
                    ep["last_t"] = t_s; ep["last_fi"] = fi
                    ep["conf_sum"] += conf_here; ep["conf_n"] += 1
                    ep["flower"] = cur                 # follow the flower the insect is on
            else:
                if ep is not None and t_s - ep["last_t"] > LAND_GRACE_S:
                    _emit_landing(landings, flower_events, ep, uid, typ, is_hb, live, on_landing, vid_stem)
                    ep = None
        if ep is not None:
            _emit_landing(landings, flower_events, ep, uid, typ, is_hb, live, on_landing, vid_stem)

    # ---- Pass 2: render annotated mp4 (no inference, just decode + draw) ---------
    if save_video and n_frames:
        _render(video, vid_dir / (vid_stem + "_annotated.mp4"), stride, out_fps,
                flowers_by_fi, draw_tracks, flower_events)

    # -- landings.csv : one row per landing episode -------------------------------
    land_fields = ["video", "flower_id", "flower_species", "track_id", "insect_type",
                   "is_honeybee", "t_enter_s", "t_exit_s", "landing_s", "is_real_landing",
                   "flower_detected", "timestamp", "pollination_weight", "conf_mean"]
    land_path = csv_dir / (vid_stem + "_landings.csv")
    with open(land_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=land_fields)
        w.writeheader(); w.writerows(landings)

    # -- flower_summary.csv : one row per flower ----------------------------------
    # count only flowers that survived gating + NMS (real blooms), plus any that got a landing
    # or an inferred synthetic flower — NOT every transient id the raw tracker ever emitted.
    all_flowers = set(flower_persist.kept_ids) | set(unk_reg.keys()) | {l["flower_id"] for l in landings}
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
    # Headline flower count = physically real blooms: flowers detected persistently, plus any
    # flower (incl. an inferred one) that actually received a real landing. This drops empty
    # synthetic flowers and brief background false positives so the count matches what's on screen.
    min_span = FLOWER_MIN_SPAN_S * out_fps
    real_flowers = flower_persist.stable_ids(min_span) | {
        l["flower_id"] for l in landings if l["is_real_landing"]}
    # collapse flower locations that are really one bloom (bee crawling across it) into one count
    cmap = flower_persist.centroids()
    cmap.update(unk_reg)
    pts = [cmap[fid] for fid in real_flowers if fid in cmap]
    n_flowers = _cluster_count(pts, FLOWER_CLUSTER_FRAC * frame_diag)
    return {"video": Path(video).name, "flowers": n_flowers, "out_fps": round(out_fps, 1),
            "landings": len(landings), "real_landings": real_total,
            "landings_csv": str(land_path), "summary_csv": str(summ_path)}


def _emit_landing(landings, flower_events, ep, uid, typ, is_hb, live, on_landing, vid_stem):
    """Close one landing episode: build its CSV row, record the real-landing event, sink it."""
    landing_s = round(ep["last_t"] - ep["enter_t"], 2)
    real = int(landing_s >= MIN_LAND_S)
    if real:
        flower_events.append((ep["last_fi"], ep["flower"]))
    ts = datetime.now().isoformat(timespec="seconds") if live else ""
    row = {
        "video": vid_stem, "flower_id": ep["flower"], "flower_species": "",
        "track_id": uid, "insect_type": typ, "is_honeybee": is_hb,
        "t_enter_s": round(ep["enter_t"], 2), "t_exit_s": round(ep["last_t"], 2),
        "landing_s": landing_s, "is_real_landing": real,
        "flower_detected": ep["detected"], "timestamp": ts,
        "pollination_weight": SPECIES_WEIGHT.get(typ, DEFAULT_WEIGHT),
        "conf_mean": round(ep["conf_sum"] / max(1, ep["conf_n"]), 3),
    }
    landings.append(row)
    if on_landing is not None:
        on_landing(row)


def _render(video, out_path, stride, out_fps, flowers_by_fi, draw_tracks, flower_events):
    """Second pass: decode the video again and draw the interpolated unified tracks.

    No inference here -- just the precomputed per-frame boxes, so the drawn boxes glide across
    every stitched gap. Live flower counts tick up at each real landing's exit frame.
    """
    # per-fi -> [(uid, box, typ)]
    per_fi: dict[int, list] = defaultdict(list)
    for tr in draw_tracks:
        for fi, box in tr["boxes"].items():
            per_fi[fi].append((tr["uid"], box, tr["typ"]))
    # per-frame NMS: never draw two insect boxes overlapping > DRAW_NMS_IOU on one frame; keep the
    # longer (more reliable) track -> no "multiple bboxes" stacked on a single bug.
    uid_len = {tr["uid"]: len(tr["boxes"]) for tr in draw_tracks}
    for fkey in list(per_fi):
        items = sorted(per_fi[fkey], key=lambda it: -uid_len.get(it[0], 0))
        keep: list = []
        for it in items:
            if all(_iou_box(it[1], k[1]) < DRAW_NMS_IOU for k in keep):
                keep.append(it)
        per_fi[fkey] = keep
    events = sorted(flower_events)                          # (exit_fi, flower_id)

    tmp_path = out_path.with_name(out_path.stem + "_mp4v.mp4")   # mp4v render, transcoded after
    cap = cv2.VideoCapture(str(video))
    writer = None
    flower_count: dict[str, int] = defaultdict(int)
    ev_i = 0
    j = fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if j % stride == 0:                                # keep the same frames pass 1 sampled
            H, W = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         out_fps, (W, H))
            while ev_i < len(events) and events[ev_i][0] <= fi:
                flower_count[events[ev_i][1]] += 1
                ev_i += 1
            writer.write(_annotate(frame, flowers_by_fi.get(fi, []), per_fi.get(fi, []), flower_count))
            fi += 1
        j += 1
    cap.release()
    if writer is not None:
        writer.release()
    # Re-encode to browser-playable H.264; if ffmpeg is absent, keep the mp4v as-is.
    if tmp_path.exists() and _to_browser_mp4(tmp_path, out_path):
        tmp_path.unlink(missing_ok=True)
    elif tmp_path.exists():
        tmp_path.replace(out_path)


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
    for uid, box, typ in drawn:                                # per-insect box + id + type
        x1, y1, x2, y2 = map(int, box)
        col = _color(uid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{typ} #{uid}", (x1, max(12, y1 - 6)),
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
