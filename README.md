# Bee-A-Hero

A computer-vision system that watches flowers, **detects and tracks** the insects
that visit them, identifies the insect type, and turns those visits into a
quantitative pollination signal — a per-flower, per-insect-type visit count with
timestamps.

The project runs in two stages:

1. **Data stage** — turn the raw iNaturalist archive into a clean, balanced,
   fully-labelled dataset.
2. **CV stage** — train a **flower detector** and a **multi-class insect
   detector**, track insects through video with BoT-SORT, and count how many
   distinct insects of each type visit each flower.

---

## What it does

Given a video of flowers, the pipeline answers three questions:

- **Where are the flowers?** — a single-class YOLO26 detector marks each flower
  and gives it a stable ID (`flower_1`, `flower_2`, …) — a separate box per flower.
- **What is visiting them?** — insects are detected as one of five types
  (`bee, fly, beetle, bug, butterfly`) and tracked frame-to-frame with **BoT-SORT**,
  so each insect gets its own ID and colour (bee #1 ≠ bee #2). The type is a
  detection class, majority-voted over the track for stability.
- **How much?** — a visit is counted when a tracked insect's box enters a flower's
  box, and each `(insect, flower)` pair is counted **once ever** (a fly-off + return
  is not a new visit), stamped with a timestamp:

  ```
  video,flower_id,total,pollinator,bee,fly,beetle,bug,butterfly
  345137_medium,flower_1,4,4,3,1,0,0,0
  ```

## Architecture

```
                ┌──────────────────────┐
 video frame ──▶│ Flower detector       │─▶ per-flower box + stable ID
                │ YOLO26 (1 class)      │
                └──────────────────────┘
                ┌──────────────────────┐
 video frame ──▶│ Insect detector       │─ BoT-SORT ─▶ box + ID + type
                │ YOLO26 (5 classes)    │              (bee/fly/beetle/bug/butterfly)
                └──────────────────────┘                     │
                            └──────────── visit logic ───────┘
                              (insect box enters flower box = 1 visit,
                               counted once per (insect, flower))
                                            │
                                            ▼
              per-flower / per-type CSV + timeline (timestamps) + annotated video
```

The insect **type is a detection class** (not a separate classifier), so
localization and typing are decided together on the full-resolution box — this
removed the bee/fly confusion of an earlier crop-classifier design. An earlier
instance-**segmentation** approach (SAM-bootstrapped masks) was tried and dropped:
on hard scenes it masked the flower instead of the insect; clean bounding boxes are
more robust for the counting task.

## Results (Act 1 — trained on this machine, RTX 3050 6 GB)

| Model | Task | mAP@0.5 | mAP@0.5:0.95 |
|-------|------|---------|--------------|
| **Flower detector** (YOLO26m, 1 class) | flower detection | **0.776** | 0.506 |
| **Insect detector** (YOLO26m, 5 classes) | insect detection + type | **0.618** | 0.403 |

Per-class insect AP@0.5: **bee 0.88 · butterfly 0.81 · bug 0.53 · beetle 0.45 · fly 0.42**.
Bee and butterfly are strong; fly/beetle/bug are the current weak point (small,
camouflaged insects on flowers) — the Act-2 improvement round targets them
(iNaturalist augmentation of the weak classes, class rebalancing, `imgsz` 800,
mixup + copy-paste; all wired into `prepare_detect.py` / `train.py`).

Detection over the test videos produced **55 flower rows / 186 counted visits**
across 5 insect types. Per-video and combined CSVs are in `test_video_result/`
(`ALL_visits.csv`, `ALL_timeline.csv`).

Data stage: **2,526** Insecta classes, **151,545** labelled images, a clean
**70 / 15 / 15** split, zero corrupt images, zero cross-split leakage.

## Run it in one snippet (weights ship in the repo)

The trained weights are committed, so a teammate/server can run **without training
or downloading datasets**. Open **`notebooks/04_detection_pipeline.ipynb`** and run
**only the last cell (§5 ⚡ ONE-SHOT TEST)** — it loads the weights, prints both
detectors' mAP, runs a test video, and writes the CSVs. `full_notebooks/04_…` is a
self-contained (no `import src`) version for a clean machine.

CLI equivalent:

```bash
python -m src.cv_engine.video_detect \
    --video data/raw/Test_Video/clip.mp4 \
    --flower-weights data/interim/cv_runs/flower_det2_yolo26m/weights/best.pt \
    --insect-weights data/interim/cv_runs/insect_multidet_yolo26m/weights/best.pt \
    --save-video
```

## Repository structure

```
src/cv_engine/
├── prepare_detect.py    # build YOLO detection sets from data/raw (flower + 5-class insect)
├── video_detect.py      # flower box+ID + insect box+ID+type (BoT-SORT) + per-flower counts + CSVs
├── train.py             # YOLO26 fine-tuning (imgsz, mixup/copy-paste, auto-resume friendly)
└── visit_counter.py     # FlowerTracker + shared helpers
notebooks/04_detection_pipeline.ipynb        # import-src; §5 = one-shot test
full_notebooks/04_detection_pipeline_full.ipynb  # self-contained (no import src)
data/interim/cv_runs/{flower_det2,insect_multidet}_yolo26m/weights/best.pt   # committed
test_video_result/ALL_visits.csv, ALL_timeline.csv                          # committed team CSVs
```

## Training datasets (Act-2, to retrain on the server)

Committed weights let you run inference immediately. To **retrain**, download these
into `data/raw/` (git-ignored) and run `prepare_detect` then `train` — see
`data/raw/DATASETS_TO_DOWNLOAD.md`:

| Dataset | Use | Source |
|---|---|---|
| Bee Detection in the Wild (Kaggle) | bee (video frames) | kaggle.com/datasets/birdy654/bee-detection-in-the-wild |
| Roboflow bee COCO sets | bee | Roboflow Universe |
| Roboflow flower COCO sets | flower | Roboflow Universe |
| Flower-visits (Zenodo, Ștefan 2025) | insect types + flower ROI | nature.com/articles/s41598-025-16140-z |
| iNaturalist 2021 | rare-class augmentation | github.com/visipedia/inat_comp/tree/master/2021 |

```bash
python -m src.cv_engine.prepare_detect both          # build flower + insect datasets (+ iNat aug)
python -m src.cv_engine.train --data data/interim/flower_det2/data.yaml \
    --name flower_det2_yolo26m --model yolo26m.pt --epochs 100 --imgsz 640 --batch 8
python -m src.cv_engine.train --data data/interim/insect_multidet/data.yaml \
    --name insect_multidet_yolo26m --model yolo26m.pt --epochs 70 --imgsz 800 --batch 4 \
    --mixup 0.1 --copy-paste 0.1
```

## Reproducibility

Everything is seeded (`SEED = 42` in `src/config.py`) and deterministic; paths
resolve relative to the repo root, so there is nothing machine-specific to
configure. Training forces the `fork` start method (Python 3.14 fix) and is
Windows-safe.

## Team

| Member | Role |
|--------|------|
| **Asif Habilov** | Team lead — planning, research direction, ML/CV engineering, QA |
| **Raul Ibrahimov** | Data research & ML engineering — dataset curation, model training |
| **Narmin Dirayeva** | LLM & ML engineering — reporting, model development |
| **Khaver** | Data & LLM — collection, annotation, quality |
