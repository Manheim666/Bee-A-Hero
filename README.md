# Bee-A-Hero

A computer-vision system that watches pomegranate flowers, detects and tracks the
insects that visit them, tells pollinators from non-pollinators, and turns those
visits into a quantitative pollination signal. It starts from object detection and
tracking and builds up toward linking insect activity to pollination and yield.

The project runs in two stages:

1. **Data stage** — turn the raw iNaturalist archive into a clean, balanced,
   fully-labelled dataset (the foundation for the models).
2. **CV stage** — train a flower detector and a two-stage insect
   detector/classifier, then track insects through video and count how many
   times each flower is visited.

---

## What it does

Given a video of flowers, the pipeline answers three questions:

- **Where are the flowers?** — a single-class YOLO26 detector marks each flower
  and gives it a stable ID (`flower_1`, `flower_2`, …).
- **What is visiting them?** — insects are detected and tracked frame-to-frame
  with BoT-SORT, then each track is classified as **pollinator** or
  **non-pollinator** by an iNaturalist-pretrained classifier.
- **How much?** — every time a tracked insect enters a flower's region it counts
  as a visit, producing a per-flower tally:

  ```
  flower_id   total   pollinator   non_pollinator
  flower_1      2          2              0
  flower_2      3          2              1
  ```

## Architecture

```
                ┌─────────────────────┐
 video frame ──▶│ Flower detector      │─▶ flower ROIs (+ IDs)
                │ YOLO26 (1 class)     │
                └─────────────────────┘
                ┌─────────────────────┐        ┌───────────────────────┐
 video frame ──▶│ Insect detector      │─track─▶│ Pollinator classifier │
                │ YOLO26 (1 class)     │        │ iNat21-pretrained     │
                │ + BoT-SORT tracking  │        └───────────┬───────────┘
                └─────────────────────┘                    │
                            └───────────── visit logic ─────┘
                                   (insect enters flower ROI = 1 visit)
                                            │
                                            ▼
                              per-flower visit dataframe + annotated video
```

Detection and classification are deliberately split: single-class detection is
easy and scores a high mAP, while the harder pollinator-vs-other decision is
handled by a classifier that was pretrained on iNaturalist — the same domain as
our data — so it starts from strong insect features.

## Results

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Flower detector (YOLO26n) | flower detection | mAP@0.5 | **0.917** |
| Insect detector (YOLO26n) | insect detection | mAP@0.5 | **0.900** |
| Insect classifier (ConvNeXt-L, iNat21) | pollinator / non-pollinator | balanced acc | **0.978** |

Visit counting was run on the test videos; per-flower tallies, the metrics
summary and a sample annotated frame are in `docs/results/cv/`. Example output —
one clip logged `flower_1: 16 visits (14 pollinator, 2 non-pollinator)`.

Data stage: **2,526** Insecta classes, **151,545** labelled images, a clean
**70 / 15 / 15** train/val/test split, zero corrupt images, zero cross-split
leakage. Figures and reports are in `docs/results/`.

## Repository structure

```
Bee-A-Hero/
├── data/                       # datasets + generated artifacts (git-ignored, see below)
│   ├── raw/                    # source data: iNaturist/, Flower/, BEE_coco/, Test_Video/
│   ├── interim/                # manifests, labels, EDA figures, weights, cv runs
│   └── processed/
├── src/
│   ├── config.py               # central paths, seed, split ratios, taxonomy targets
│   ├── data_pipeline/          # data stage
│   │   ├── inaturalist_prep.py #   Insecta filter · dedup · 70/15/15 split
│   │   ├── label_tools.py      #   label regeneration + integrity validation
│   │   └── eda.py              #   EDA / quality primitives
│   └── cv_engine/              # CV stage
│       ├── prepare_flower.py   #   Flower classification → YOLO detection (GrabCut)
│       ├── prepare_insect.py   #   iNat + BEE.v8i → detector + classifier datasets
│       ├── train.py            #   YOLO26 fine-tuning
│       ├── train_classifier.py #   iNat21-pretrained pollinator classifier
│       ├── visit_counter.py    #   tracking + flower-visit counting
│       └── weights.py          #   publish/download trained weights (HF Hub)
├── notebooks/
│   ├── 00_data_ready.ipynb     # data-stage gate (inspect → clean → validate → report)
│   └── 01_eda.ipynb            # exploratory data analysis
├── scripts/
│   ├── setup_env.sh            # data-stage environment
│   ├── setup_cv.sh             # CV-stage environment (torch / ultralytics / timm)
│   ├── run_pipeline.sh         # run the data stage end-to-end
│   └── run_cv.sh               # run the CV stage end-to-end
├── docs/results/               # result snapshot (figures + report JSONs)
└── tests/
```

## Getting the datasets

The datasets are large and are **not** stored in git, so a fresh clone has an
empty `data/` tree. Download each and place it as shown:

| Dataset | Place under | Source |
|---------|-------------|--------|
| iNaturalist 2021 | `data/raw/iNaturist/{train_mini,val,public_test}/` (+ their `.json`) | https://github.com/visipedia/inat_comp/tree/master/2021 |
| Flower Classification | `data/raw/Flower/{Training,Validation,Testing} Data/<class>/` | Kaggle "Flower Classification" |
| BEE.v8i (bee boxes) | `data/raw/BEE_coco/{train,valid,test}/` (COCO export) | Roboflow Universe "BEE" |
| Test videos | `data/raw/Test_Video/*.mp4` | your own footage / test clips |

Every stage checks for its inputs and prints what is missing, so you can run with
whatever you have (e.g. the insect step still works without `BEE_coco`).

## Setup

```bash
bash scripts/setup_env.sh     # creates .venv, installs the data-stage dependencies
bash scripts/setup_cv.sh      # adds the CV-stage dependencies (needs an NVIDIA GPU to train)
source .venv/bin/activate
```

## Running

**Data stage** — clean, split and validate the dataset:

```bash
bash scripts/run_pipeline.sh
```

**CV stage** — train the detectors + classifier and count visits on the test
videos (idempotent: finished stages are skipped on re-run):

```bash
bash scripts/run_cv.sh
```

**Pretrained weights** — skip training and run inference directly:

```bash
python -m src.cv_engine.weights --download            # pull trained weights from the Hub
python -m src.cv_engine.visit_counter --video data/raw/Test_Video/clip.mp4 \
    --flower-weights   data/interim/weights/flower_yolo26.pt \
    --insect-weights   data/interim/weights/insect_yolo26.pt \
    --classifier-weights data/interim/weights/insect_classifier.pt --save-video
```

## Reproducibility

Everything is seeded (`SEED = 42` in `src/config.py`) and deterministic: the same
raw data yields the same removed images, splits, labels, and video subset on any
machine. Paths are resolved relative to the repository root, so there is nothing
machine-specific to configure.

## Team

| Member | Role |
|--------|------|
| **Asif Habilov** | Team lead — planning, research direction, ML/CV engineering, QA |
| **Raul Ibrahimov** | Data research & ML engineering — dataset curation, model training |
| **Narmin Dirayeva** | LLM & ML engineering — reporting, model development |
| **Khaver** | Data & LLM — collection, annotation, quality |
```
