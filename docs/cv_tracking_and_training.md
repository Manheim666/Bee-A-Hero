# Bee-A-Hero — CV: Training & Video-Tracking Technical Log

Everything technical and logical behind the detection + tracking pipeline: how the
models were trained, why the video output behaved the way it did, and every
post-processing fix applied to make the annotated videos stable — **without retraining**.

---

## 1. Pipeline at a glance

```
video ─┬─► flower detector (YOLO26m, 1 class) ──► FlowerTracker (IoU + smoothing) ─┐
       │                                                                            ├─► landing
       └─► insect detector (YOLO26m, 5 class) ──► BoT-SORT track ──► type vote ──────┘  state machine
                                                        │                                    │
                                              honeybee sub-classifier                        ▼
                                              (bee crop → honeybee/bee)          landings.csv + flower_summary.csv
                                                                                 + annotated .mp4
```

- **Flower detector** — single class `flower`. Per frame it returns raw flower boxes.
- **Insect detector** — 5 classes: `bee, fly, beetle, bug, butterfly`. Tracked with
  **BoT-SORT** (`persist=True`) so each insect keeps one track ID.
- **Honeybee sub-classifier** — runs only on `bee` crops, splits `honeybee` (Apis) from
  other bees (honeybee weighted ~10× for pollination value).
- **Landing state machine** — a contiguous span where an insect is on a flower (box centre
  inside a flower ROI, *or* near-motionless with no ROI). Dwell ≥ `MIN_LAND_S` (2 s) = a
  *real* landing (feeding, not a fly-through). Fly-off + return = a new landing.

Source: `src/cv_engine/video_detect.py` (orchestration + landing logic),
`src/cv_engine/visit_counter.py` (`FlowerTracker`, geometry helpers).

---

## 2. Training

### 2.1 Data
iNaturalist-derived, balanced to **2 526 species × ~60 images**, 70/15/15 split,
151 545 images, 0 cross-split leakage (see `docs/results/*_summary.json`). The insect
detector's 5 coarse classes are rolled up from order/family taxonomy.

### 2.2 Models & the v1 → v2 jump
Both detectors are **YOLO26m**. v2 changed two things over v1:

- **imgsz 640 → 768** (more pixels on small insects on flowers).
- **stronger augmentation**: `mixup` + `copy-paste` (more object variety per image).

| Detector | Version | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| Flower  | v1 | 640 | 0.786 | 0.498 | — | — |
| Flower  | **v2 (used)** | 768 | **0.806** | **0.532** | 0.784 | 0.731 |
| Insect  | v1 | 640 | 0.604 | 0.397 | — | — |
| Insect  | **v2 (used)** | 768 | **0.664** | **0.472** | 0.706 | 0.646 |

v2 is the deployed ("v3 results") model in every notebook/CSV/video. The insect detector
gained the most (**+0.06 mAP@50**) from the imgsz + augmentation change.

### 2.3 Training cost (measured, RTX-class GPU, 40 epochs each)
| Detector | Epochs | Wall-clock | ≈ per epoch |
|---|---|---|---|
| Flower v2 | 40 | ~6.7 h | ~10 min |
| Insect v2 | 40 | ~16.7 h | ~25 min |

So a **full retrain ≈ 17 h** (insect) and a **short 10-epoch fine-tune ≈ 4 h**. Not "quick".

### 2.4 What the detector numbers *mean for the video*
This is the crucial link between the metrics and what you see on screen:

- **Insect recall = 0.646** → at any given frame the detector finds ~⅔ of the insects
  present. That is exactly why a 4-bee scene shows **"2, then 3 of 4"** — the missing bee
  simply isn't detected yet on that frame. No tracker can draw a box the detector never
  produced.
- **Insect precision = 0.706** → ~30% of raw insect detections are false. That is why a
  **flower is occasionally read as "bee"**. A single-frame false hit is a precision error,
  not a bug in the tracker.

**Conclusion:** the two "missing / spurious" behaviours are *detector* limits. Post-processing
can suppress the *transient* cases (below) but cannot invent a missed bee or perfectly erase a
*persistent* false positive — only a better detector (retrain / more data / higher conf) can.

---

## 3. Tracking & landing logic (baseline)

- **FlowerTracker** re-detects flowers every `flower_interval` (5) frames and matches
  detections to existing flower tracks by **IoU** so `flower_1` stays the same flower as the
  scene moves. `seen` records every flower ID for the report.
- **Insect tracking** is BoT-SORT, one ID per insect, one deterministic colour per ID.
- **Type** is taken from the detector and **voted over the whole track life** (a single noisy
  frame should not decide the species).
- **Landing** uses a normalised, scale-free speed (`body-lengths/second`) so "settled" is
  independent of insect size / camera distance; brief drop-outs `< LAND_GRACE_S` (0.5 s) are
  bridged inside one landing episode.

---

## 4. Video-stability upgrades (this session — all training-free)

Each fix targets a specific artefact seen in the annotated videos. All are pure
post-processing on top of the frozen detectors.

### 4.1 Insect type flips (bee → fly → bee)
- **Cause:** type was a *raw* running majority; an early or noisy burst of a wrong class
  could win momentarily, and every frame was free to switch.
- **Fix (2 parts):**
  1. **Confidence-weighted cumulative vote** — each frame's vote is weighted by the detection
     confidence (`votes[tid][cls] += conf`) instead of `+= 1`. A low-confidence "fly" blip
     barely moves the running total; the final CSV type is this full-life weighted argmax.
  2. **Display hysteresis** — the on-screen label only switches when the challenger's
     cumulative weight is `≥ LABEL_SWITCH_MARGIN (1.5×)` the current label's. A brief excursion
     can't steal the label.
- **Params:** `LABEL_SWITCH_MARGIN = 1.5`.

### 4.2 Flower boxes blinking off/on
- **Cause:** `current()` returned only tracks with `missed == 0`, so the instant a detect-frame
  failed to re-find a flower, its box vanished — even though the track was still alive.
- **Fix:** **presence hold** — keep drawing a flower's last-known box while `missed ≤ hold`.
- **Param:** `hold = 6` detect-frames (≈ 1.2 s of video). This is "voting on existence over
  time": a flower seen consistently is assumed still there through a short gap.

### 4.3 Multiple flowers swapping boxes
- **Cause:** association was greedy *per track, first-come* — the first track in dict order grabbed
  its best-IoU detection, so two nearby flowers could steal each other's box → ID swap.
- **Fix:** **global one-to-one matching** — rank *every* (track, detection) IoU pair, assign the
  strongest pairs first, each track/detection used once. Neighbouring flowers can no longer cross.

### 4.4 Flower box jitter + static-scene misses (camera still, wind)
- **Cause:** raw detector boxes wobble frame-to-frame; on a static scene, wind/leaf motion drops
  the flower's confidence for a frame and the box is lost.
- **Fix:**
  - **EMA box smoothing** — the matched box is blended into the track
    (`box = 0.5·old + 0.5·new`) instead of replaced → damps jitter, stabilises IoU.
  - **Longer grace on static scenes** — `hold = 6`, `max_missed = 45` detect-frames. Because a
    still camera means the flower is ~fixed, holding the last box across a longer gap is safe and
    stops the "loses the flower" effect.
- **Params:** `smooth = 0.5`, `hold = 6`, `max_missed = 45`.

### 4.5 Insect box disappearing for milliseconds, then returning
- **Cause:** the annotator drew only *this frame's* detections. When BoT-SORT dropped a track for
  1–3 frames, the box blinked out and back.
- **Fix:** **insect presence hold** — keep drawing the last box for a lost track until either the
  box reaches the frame edge (insect left) or a `INSECT_HOLD_MAX` cap (~3 s). This delivers the
  requested behaviour: **the box either isn't shown, or stays until the insect leaves the frame.**
- **Params:** `INSECT_HOLD_MAX = 72` frames, `INSECT_EDGE_FRAC = 0.02` (edge margin).

### 4.6 Butterfly wing-flap changing the box
- **Cause:** insect boxes had *no* spatial smoothing (only the type vote was smoothed). Open vs
  closed wings swing the box size every few frames.
- **Fix:** **EMA-smooth the drawn insect box** (`INSECT_BOX_SMOOTH = 0.5`). This damps the size
  oscillation while still tracking real motion. Landing logic still uses the *raw* centroid, so
  accuracy of "inside flower" is unaffected — only the drawn box is smoothed.

### 4.7 Transient false positives (flower momentarily read as "bee")
- **Cause:** insect precision ~0.71 → occasional 1–2 frame false hits on flower texture.
- **Fix:** **persistence gate** — a track must be detected in `≥ MIN_TRACK_DRAW (3)` frames before
  its box is *ever* drawn. A 1–2 frame false blip never reaches the threshold, so it never shows.
- **Limit:** a *stable* misdetection (a flower read as bee across many frames) survives the gate —
  that is a precision limit only a detector change can fix (see §6).
- **Param:** `MIN_TRACK_DRAW = 3`.

---

## 5. Measured effect of the upgrades
Re-running all 20 test videos with the fixes vs the previous pass:

- **Phantom flowers removed:** flower rows `69 → 63` across the set (e.g. `345137` lost
  `flower_8`, `flower_9`, `flower_unk_1` — all 0-landing ghosts born from blink / ID churn).
- **Real landings preserved:** `232 → 233` (smoothing did not destroy genuine landings).
- **Qualitative:** flower boxes hold steady (no blink, no swap); insect boxes persist through
  brief drop-outs and stop swinging on wing-flaps; species labels stop flickering.

---

## 6. When retraining *is* the answer (and the cost)
Post-processing cannot fix these — they need a better detector:

| Symptom | Root cause | Cheapest real fix | Approx cost |
|---|---|---|---|
| A bee is never detected (only 3 of 4) | insect **recall 0.65** | more data / longer train / lower `conf` (trades precision) | fine-tune ~4 h / full ~17 h |
| A flower is *persistently* read as bee | insect **precision 0.71** | hard-negative flower crops + retrain, or raise `conf` | fine-tune ~4 h |
| Tighter boxes on tiny insects | resolution / scale | train at imgsz ≥ 768 (already done in v2) | — |

A **~4 h fine-tune** (≈10 epochs, warm-started from v2 `best.pt`, adding hard-negative flower
crops as background) is the highest-leverage option if precision/recall must improve at the
source. A full retrain is ~17 h. Everything in §4 is free and already applied.

---

## 7. Parameter reference

| Param | File | Value | Controls |
|---|---|---|---|
| `LABEL_SWITCH_MARGIN` | video_detect.py | 1.5 | how decisively the vote must flip before the label switches |
| `INSECT_BOX_SMOOTH` | video_detect.py | 0.5 | EMA factor for the drawn insect box (wing-flap damping) |
| `INSECT_HOLD_MAX` | video_detect.py | 72 | max frames to hold a lost insect box (~3 s) |
| `INSECT_EDGE_FRAC` | video_detect.py | 0.02 | border margin that counts as "left the frame" |
| `MIN_TRACK_DRAW` | video_detect.py | 3 | frames a track must exist before its box is drawn |
| `MIN_LAND_S` | video_detect.py | 2.0 | dwell for a *real* landing |
| `LAND_GRACE_S` | video_detect.py | 0.5 | bridge brief exits inside one landing |
| `smooth` | visit_counter.py | 0.5 | EMA factor for flower boxes |
| `hold` | visit_counter.py | 6 | detect-frames a flower box is held through misses |
| `max_missed` | visit_counter.py | 45 | detect-frames a flower **ID** stays alive for re-association |

---

## 8. How to reproduce
```bash
pip install -r src/cv_engine/requirements-cv.txt          # torch, ultralytics, opencv
python -m src.cv_engine.video_detect \
    --video data/raw/Test_Video \
    --flower-weights   data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt \
    --insect-weights   data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt \
    --honeybee-weights data/interim/cv_runs/honeybee_clf/best.pt \
    --save-video
```
Outputs `<video>_landings.csv`, `<video>_flower_summary.csv`, `<video>_annotated.mp4`, plus
`ALL_*.csv` aggregates, into `test_video_result/`.
