# 03 · CV Engine (the core — Speaker 2)

The single source of truth is `src/cv_engine/video_detect.py :: count_visits_det()`. The website upload, the offline CSVs, and the live camera all run this same logic — the numbers cannot diverge.

---

## Models (all fine-tuned YOLO26-m, transfer-learned)
| Model | File | Role | Metric |
|---|---|---|---|
| Flower detector | `flower_det2_v2_yolo26m` | boxes each pomegranate flower (the ROI) | mAP@0.5 = **0.808** |
| Insect detector | `insect_multidet_v2_yolo26m` | multi-class: `bee, fly, beetle, bug, butterfly` | mAP@0.5 = **0.669** |
| Honeybee sub-classifier | `honeybee_clf` | relabels a `bee` crop → `honeybee` vs other-bee | acc = **0.978** |

**Why YOLO26:** NMS-free / end-to-end, ProgLoss + STAL refinements improve **small-object recall** (distant flowers/bees are small). **Why a separate sub-classifier:** a combined detect+species head hurts both tasks; honeybees carry ~10× the pollination weight, so the split is worth it.

**Tracking:** BoT-SORT (Ultralytics default) with `persist=True` — camera-motion compensation + refined Kalman filter, optional ReID. Precedent: a 2025 PLOS One study tracked honeybees with YOLOv8 + BoT-SORT at 98% mean accuracy, 36 fps.

---

## The pipeline is TWO passes (this is the key design)
A live single pass can't "drag a box smoothly from where it vanished to where it reappears," because you don't yet know the reappear point. So we split:

### Pass 1 — track + collect (draw nothing, count nothing)
- Resample the video to a fixed **20 fps** (`vid_stride`) — smoother tracks, cheaper, no jitter from native fps.
- Per frame: flower detection (every 5th frame; held between), insect detection+tracking (BoT-SORT ids).
- Record every raw detection: `(frame, box, confidence, class)`. Accumulate a **confidence-weighted cumulative type vote** per track over its whole life, plus honeybee-vs-bee votes from the sub-classifier on `bee` crops.

### Stitch (offline, between passes) — kills double-counting
BoT-SORT loses/re-mints an id whenever an insect is occluded or the detector blinks. Two merges:
1. **Concurrent merge:** two tracks that co-exist on the same frames and overlap spatially (mean IoU ≥ 0.45) are **one insect with two ids** (e.g. a `bee`-box and an overlapping `butterfly`-box) → merge.
2. **Gap re-link:** a track that vanishes and reappears **within 5 s** *and* **within ~4×√area** of the vanish point is the **same insect** (occlusion behind a petal / detector blink) → merge. A genuine fly-off that returns far/late stays separate.

### Interpolate
Every gap inside a unified track is filled by **linearly dragging the box** corner-to-corner from vanish → reappear point, with a mild EMA to damp butterfly wing-flap size swings. → the box glides across the gap, no blink/snap.

### Landings — the visit state machine (on the gap-filled timeline)
A **landing episode** = a contiguous span where an insect is *on* a flower:
- **Detected:** box centre inside a flower ROI, **or**
- **Inferred:** near-motionless with no ROI (normalised speed < `STATIONARY_TAU = 0.5` body-lengths/s) — catches undetected flowers.
- Brief drop-outs < `LAND_GRACE_S = 0.5 s` are **bridged** (one episode).
- `landing_s ≥ MIN_LAND_S = 2.0 s` ⇒ a **real** landing (feeding). Shorter = fly-through, not counted.
- **A fly-off then return is a NEW landing;** an occlusion that re-links is the **same** landing. That distinction is what keeps counts honest.

### Pass 2 — render the annotated video (no inference)
Re-decode the video, draw the interpolated unified boxes + ids + live counts + flower boxes. Written mp4v then **transcoded to browser-playable H.264 / yuv420p via system ffmpeg** (this OpenCV build has no H.264 encoder; even-dimension scale forces libx264 to accept odd-sized clips).

---

## Type decision — confidence-weighted cumulative vote + hysteresis
- Each detection adds its **confidence** to the track's vote for that class → the label is stable over the whole track, not per-frame flicker.
- The **displayed** label only switches when a challenger's cumulative weight ≥ **1.5×** the current leader's (`LABEL_SWITCH_MARGIN`) → a brief bee→fly flicker doesn't flip it.
- `bee` crops go to the honeybee sub-classifier; honeybee wins the split if its votes lead.

---

## The false-positive / quality gates (why the boxes are clean)
YOLO is **closed-set** — a person or a flower has no "none" class, so it snaps onto a trained class. We defend geometrically:

| Gate | Rule | Fixes |
|---|---|---|
| **Person veto** (live) | drop any detection whose centre sits in a COCO `person` box (yolov8n) | humans read as flowers/insects |
| **Max-size gate** | insect box > 18% of frame (`MAX_INSECT_FRAME_FRAC`) → drop | whole-flower/wall boxed as an insect |
| **Flower-as-insect veto** | insect box with IoU ≥ 0.80 to a flower → it *is* the flower → drop | "flower detected as butterfly". Kept high so a bee **on** a flower (small box, low IoU) survives |
| **Insect nesting dedup** | class-agnostic NMS: two insect boxes IoU ≥ 0.6 or one 75% inside another → keep the higher-conf | "insect inside an insect" |
| **Draw-time NMS** | never draw two insect boxes overlapping > 0.45 on one frame; keep the longer track | "multiple bboxes" stacked on one bug |
| **Flower plausibility** | reject flower boxes that are slivers, noise, or > half the frame | "whole screen / wall / greenery boxed as flower" |

## Flower steadiness & counting (flowers are static)
- **Canonicalise** flowers by **centre location** (one stable id per bloom, `FLOWER_MERGE_K = 1.6×√area`) — a jittery box doesn't spawn new ids.
- **Hold + EMA** each flower box for 8 s after its last detection → no flicker / disappearance.
- **Count only real blooms:** a flower must be present ≥ 1 s (drops brief background FPs) **or** have received a real landing; empty synthetic flowers are dropped; landing spots within ~16% of the frame diagonal are **spatially clustered** (a bee crawling across one bloom = one flower).

## Outputs (grouped under `test_video_result/` or `cv_out/`)
- `csv/<video>_landings.csv` — one row per landing episode: enter, exit, dwell, type, is_honeybee, is_real_landing, flower_detected, pollination_weight, conf_mean.
- `csv/<video>_flower_summary.csv` — one row per flower: per-type counts, total/mean dwell, **pollination_score**.
- `csv/ALL_*.csv` — merged across videos (feeds ML/LLM).
- `videos/<video>_annotated.mp4` — the H.264 overlay.

## Pollination score (per flower)
`pollination_score = Σ over real landings of  species_weight × min(dwell, 30 s)`
Species weights: **honeybee 10**, butterfly 2, bee 1, fly 0.5, beetle 0.5, bug 0.2 (default 0.3). Honeybees dominate because they are the effective pomegranate pollinators; dwell is capped at 30 s so one long sit doesn't swamp the signal.

## Algorithmic parameters cheat-sheet (all named constants)
`TARGET_FPS 20` · `MIN_LAND_S 2.0` · `STATIONARY_TAU 0.5` · `LAND_GRACE_S 0.5` · `LABEL_SWITCH_MARGIN 1.5` · `RELINK_MAX_S 5` / radius `4×√area` · `CONCURRENT_MERGE_IOU 0.45` · `DRAW_NMS_IOU 0.45` · `MAX_INSECT_FRAME_FRAC 0.18` · `INSECT_FLOWER_IOU 0.80` · `FLOWER_HOLD_S 8` · `FLOWER_MIN_SPAN_S 1.0` · `FLOWER_MERGE_K 1.6` · `insect conf 0.20`, `flower conf 0.20`.

## Anticipated CV questions → see `09_QA`
"How do you avoid double-counting?", "Bee vs fly errors?", "Why 2 seconds?", "Whole screen as flower — how fixed?", "Why two passes?", "What if the flower isn't detected?"
