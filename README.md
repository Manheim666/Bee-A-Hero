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

## Results (trained on this machine, RTX 3050 6 GB)

The detectors were retrained (**v2 / Act-2**) and now beat the **v1 / Act-1** baselines on
every metric. The detection + tracking + **landing** pipeline (**v3 / Act-3**) regenerates the
test-video CSVs and annotated videos from the best weights — **v3 is not a new model, it is the
results produced by the best detectors + the honeybee subclassifier.** Only the best weights per
detector ship in the repo (v2); v1 is kept locally for reference.

### Detectors — v1 vs v2 (v2 shipped)

| Detector | Ver | mAP@0.5 | mAP@0.5:0.95 | recall | key change |
|---|---|---|---|---|---|
| Flower (YOLO26m, 1 cls) | v1 | 0.776 | 0.506 | — | imgsz 640 |
| **Flower** | **v2** ✅ | **0.815** | **0.537** | — | imgsz 768, longer schedule |
| Insect (YOLO26m, 5 cls) | v1 | 0.618 | 0.403 | 0.565 | imgsz 640 |
| **Insect** | **v2** ✅ | **0.683** | **0.476** | **0.623** | imgsz 768 + mixup/copy-paste |

**What changed v1 → v2:** `imgsz` 640 → 768, **mixup + copy-paste** augmentation, longer training.
The Act-1 weak point — small, camouflaged fly/beetle/bug and low insect recall — improved:
insect **recall 0.565 → 0.623** and **localization (mAP@0.5:0.95) +0.07** on both detectors, so the
boxes in the annotated videos are noticeably tighter. (v1 per-class AP@0.5 for reference:
bee 0.88 · butterfly 0.81 · bug 0.53 · beetle 0.45 · fly 0.42.)

### Honeybee subclassifier (v2-era)

Binary honeybee (*Apis*) vs other-bee, run on `bee` crops inside `video_detect`. Honeybees are
weighted **10×** in `pollination_score` (they pollinate far more than other bees). iNaturalist
data is thin (168 *Apis* training images), so this is **provisional: F1 0.523** (recall ~0.75,
precision ~0.38 → it over-calls honeybee). Treat `is_honeybee` and the honeybee share of
`pollination_score` as approximate until more *Apis* data is added.

### v3 — landing results over the 20 test videos

233 landing episodes → **52 real landings** (dwell ≥ 2 s). By insect type: **honeybee 20 · fly 14 ·
beetle 10 · butterfly 4 · bee 3 · bug 1**. **69 flowers** tracked, total **pollination_score 1838**.
Zero *inferred* (undetected-flower) landings — flower v2 recall caught every flower that saw a real
landing. Per-video + combined tables in `test_video_result/` (`ALL_landings.csv`,
`ALL_flower_summary.csv`) feed the ML + LLM phases.

Data stage: **2,526** Insecta classes, **151,545** labelled images, a clean
**70 / 15 / 15** split, zero corrupt images, zero cross-split leakage.

## Run it in one snippet (weights ship in the repo)

The trained weights are committed, so a teammate/server can run **without training
or downloading datasets**. Open **`notebooks/04_detection_pipeline.ipynb`** and run
**only the last cell (§5 ⚡ ONE-SHOT TEST)** — it loads the weights, prints both
detectors' mAP, runs a test video, and writes the CSVs. `full_notebooks/04_…` is a
self-contained (no `import src`) version for a clean machine.

CLI equivalent:

**One command runs the whole `data/raw/Test_Video/` folder** through the best
weights and writes everything to `test_video_result/` — no notebook, no training,
no dataset download. Point `--video` at the folder (or a single clip):

```bash
python -m src.cv_engine.video_detect \
    --video data/raw/Test_Video \
    --flower-weights data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt \
    --insect-weights data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt \
    --honeybee-weights data/interim/cv_runs/honeybee_clf/best.pt \
    --save-video
```

For each video it writes, to `test_video_result/`:
- `<video>_landings.csv` — one row per landing episode (enter/exit/dwell, type,
  `is_honeybee`, `is_real_landing`≥2s, `flower_detected` detected|inferred, `pollination_weight`)
- `<video>_flower_summary.csv` — per-flower counts, dwell, `pollination_score`
- `<video>_annotated.mp4` — bbox video (flower + per-insect boxes/IDs/type + live counts)

and aggregates all videos into `ALL_landings.csv` + `ALL_flower_summary.csv` for the ML/LLM phase.

## Repository structure

```
src/cv_engine/
├── prepare_detect.py    # build YOLO detection sets from data/raw (flower + 5-class insect)
├── video_detect.py      # flower+insect boxes/IDs/type (BoT-SORT) + landing episodes + CSVs
├── train.py             # YOLO26 fine-tuning (imgsz, mixup/copy-paste, auto-resume friendly)
├── honeybee_clf.py      # honeybee-vs-other-bee subclassifier (run on bee crops in video_detect)
└── visit_counter.py     # FlowerTracker + shared helpers
notebooks/04_detection_pipeline.ipynb        # import-src; §5 = one-shot test
full_notebooks/04_detection_pipeline_full.ipynb  # self-contained (no import src)
data/interim/cv_runs/{flower_det2,insect_multidet}_v2_yolo26m/weights/best.pt   # committed
data/interim/cv_runs/honeybee_clf/best.pt                                    # committed
test_video_result/ALL_landings.csv, ALL_flower_summary.csv                   # committed team CSVs
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
