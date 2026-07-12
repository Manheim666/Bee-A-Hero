# Bee-A-Hero

**A pollination-monitoring system that turns orchard video into a quantitative pollination
signal.** Cameras watch flowers; the pipeline detects and tracks the insects that visit them,
measures how long each insect lands, and converts those landings — stage by stage — into a
per-flower, per-species visit record, a fruit-set estimate, and a yield estimate, finally
narrated as a plain-language report.

```
Camera video
   │
   ▼
[ DATA ]  clean, balanced, labelled training sets            notebooks/00_data_ready · 01_eda
   │                                                          src/data_pipeline/
   ▼
[ CV ]    flower + insect detection · BoT-SORT tracking       notebooks/02_cv
          landing episodes → per-flower / per-species CSVs     src/cv_engine/         ✅ shipped
   │
   ▼
[ ML ]    visits → effective dose → fruit-set dose–response    notebooks/03_ml
          → yield band, with uncertainty                       src/ml_models/         ✅ integrated
   │
   ▼
[ LLM ]   grounded, farmer-friendly pollination report         notebooks/04_llm
                                                                src/llm_reporting/     ✅ integrated
```

The stages are wired end-to-end: the CV tracker writes `test_video_result/csv/ALL_*.csv`; the ML
stage (`python -m src.ml_models.train`) applies the committed fruit-set curve
(`models/dose_response_v11.json`) to those landings and writes a yield band; the LLM stage
(`python -m src.llm_reporting.generate`) turns the grounded CV + ML numbers into a grower report
(Claude `claude-opus-4-8` when a key is present, else a deterministic offline template).

The project is organised as a **four-stage pipeline on four branches** that merge into `main`:

| Stage | Branch | Notebook slot | Source package | Status |
|---|---|---|---|---|
| **Data** | `data` | `00_data_ready`, `01_eda` | `src/data_pipeline/` | ✅ built |
| **CV** | `cv` | `02_cv` | `src/cv_engine/` | ✅ trained + shipped weights |
| **ML** | `ml` | `03_ml` | `src/ml_models/` | ✅ integrated onto `main` (fruit-set curve + yield band) |
| **LLM** | `llm` | `04_llm` | `src/llm_reporting/` | ✅ integrated onto `main` (grounded report, Claude + offline) |

`main` is the integration branch and holds the canonical structure; each team fills its own
notebook slot and source package, then merges. The slots not owned by a stage are kept as empty
placeholders so every branch shares one layout and merges without conflict.

---

## 1. Data stage — from raw archive to a clean dataset

Turns the raw iNaturalist archive (plus targeted bee/flower detection sets) into a
fully-labelled, balanced, leak-free training corpus.

- **Result:** 2,526 Insecta classes, **151,545** labelled images, a clean **70 / 15 / 15**
  split, zero corrupt images, zero cross-split leakage.
- **Where:** `src/data_pipeline/` (`inaturalist_prep.py`, `eda.py`, `label_tools.py`,
  `flower/`), notebooks `00_data_ready` + `01_eda`, and `full_notebooks/00…`, `01…`
  (self-contained variants).
- **Retraining datasets:** see `data/raw/DATASETS_TO_DOWNLOAD.md` and §6 below.

## 2. CV stage — detection, tracking, landing counting ✅

Given a video, the CV stage answers three questions:

- **Where are the flowers?** A single-class YOLO26 detector marks each flower and gives it a
  stable ID (`flower_1`, `flower_2`, …).
- **What is visiting them?** A five-class YOLO26 insect detector (`bee, fly, beetle, bug,
  butterfly`) plus **BoT-SORT** tracking gives each insect its own ID and type (majority-voted
  over the track). A **honeybee subclassifier** further splits `bee` into honeybee (*Apis*) vs.
  other bee, because honeybees pollinate far more.
- **How much?** A **landing episode** is a contiguous span of an insect on a flower. Episodes
  with dwell **≥ 2 s** count as *real landings* (a fly-through is not a feeding visit). Each
  landing carries enter/exit time, dwell length, type, `is_honeybee`, and a per-species
  `pollination_weight`.

The insect **type is a detection class** (not a separate crop classifier), so localization and
typing are decided together on the full-resolution box — this removed the bee/fly confusion of
an earlier design. An earlier instance-**segmentation** approach (SAM-bootstrapped masks) was
tried and dropped: on hard scenes it masked the flower instead of the insect; clean bounding
boxes are more robust for counting.

### 2.1 Results (trained on this machine, RTX 3050 6 GB)

Detectors were retrained (**v2 / Act-2**) and beat the **v1 / Act-1** baselines on every metric.
The landing pipeline (**v3 / Act-3**) regenerates the test-video CSVs and annotated videos from
the best weights — **v3 is not a new model, it is the results produced by the best detectors +
the honeybee subclassifier.** Only the best weights per detector ship (v2); v1 is kept locally.

| Detector | Ver | mAP@0.5 | mAP@0.5:0.95 | recall | key change |
|---|---|---|---|---|---|
| Flower (YOLO26m, 1 cls) | v1 | 0.776 | 0.506 | 0.683 | imgsz 640 |
| **Flower** | **v2** ✅ | **0.808** | **0.537** | **0.738** | imgsz 768, longer schedule |
| Insect (YOLO26m, 5 cls) | v1 | 0.618 | 0.404 | 0.581 | imgsz 640 |
| **Insect** | **v2** ✅ | **0.669** | **0.476** | **0.623** | imgsz 768 + mixup/copy-paste |

*(Best-checkpoint validation metrics, argmax mAP@0.5:0.95 — the `best.pt` that ships and the
figure `notebooks/02_cv.ipynb` prints.)* **v1 → v2:** `imgsz` 640 → 768, **mixup + copy-paste**
augmentation, longer training. Small, camouflaged fly/beetle/bug and low insect recall were the
Act-1 weak point; insect **recall 0.581 → 0.623** and **localization (mAP@0.5:0.95) up on both**
(+0.03 flower, +0.07 insect), so boxes in the annotated videos are noticeably tighter.

**Honeybee subclassifier (v2-era).** Binary honeybee (*Apis*) vs. other-bee on `bee` crops.
iNaturalist *Apis* data is thin (168 training images), so this is **provisional: F1 0.523**
(recall ~0.75, precision ~0.38 → it over-calls honeybee). Treat `is_honeybee` and the honeybee
share of `pollination_score` as approximate until more *Apis* data is added.

**Landing results over the 20 test videos** (476 s total, 24–60 fps, 1280×720 → 2732×1440;
19 of 20 clips saw a real visit). 155 landing episodes → **31 real landings** (dwell ≥ 2 s).
By type: **honeybee 21 · bee 4 · butterfly 3 · fly 2 · bug 1**. **60 flowers** tracked, total
**pollination_score 1756.3**, with **2** *inferred* (undetected-flower) landings. These counts are
after the tracking-robustness pass (cumulative type voting + box smoothing + `MAX_INSECT_FRAME_FRAC`
size gate, no retraining). Combined tables in `test_video_result/csv/` (`ALL_landings.csv`,
`ALL_flower_summary.csv`) feed the ML + LLM stages.

### 2.2 Run it in one snippet (weights ship in the repo — best model only)

The best weights are committed, so a teammate/server runs **without training or downloading
datasets**. Point `--video` at a folder (batch) or a single clip:

```bash
python -m src.cv_engine.video_detect \
    --video data/raw/Test_Video \
    --flower-weights   data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt \
    --insect-weights   data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt \
    --honeybee-weights data/interim/cv_runs/honeybee_clf/best.pt \
    --save-video
```

Notebook equivalent: open **`notebooks/02_cv.ipynb`** and run **only the last cell (§5 ⚡
ONE-SHOT TEST)** — it loads the best weights, prints both detectors' mAP, runs a test video, and
writes the CSVs. `full_notebooks/02_cv_full.ipynb` is a self-contained (no `import src`) version
for a clean machine. Both use the **best model only**.

For each video it writes, grouped under `test_video_result/`:
- `csv/<video>_landings.csv` — one row per landing episode (enter/exit/dwell, type, `is_honeybee`,
  `is_real_landing` ≥ 2 s, `flower_detected` detected|inferred, `pollination_weight`)
- `csv/<video>_flower_summary.csv` — per-flower counts, dwell, `pollination_score`
- `videos/<video>_annotated.mp4` — bbox video (flower + per-insect boxes/IDs/type + live counts)

and aggregates all videos into `csv/ALL_landings.csv` + `csv/ALL_flower_summary.csv` for the ML/LLM
stages. Useful knobs: `--conf 0.2` (insect sensitivity), `--flower-conf 0.15`, `--target-fps 24`.

## 3. ML stage — visits → fruit set → yield 🟡

The CV CSVs are the **input**; fruit set and yield are the **output**. The modeling approach
(designed, scaffolding in `src/ml_models/`):

- **Effective dose, not a flat count.** Raw counts are an imperfect proxy — species, dwell time,
  tracking reliability and weather all matter. Each flower's dose is a weighted sum
  `D = Σ w_species · φ(dwell) · reliability · weather_gate`, correcting the **attenuation bias**
  that imperfect tracking introduces (it systematically *understates* pollinator value).
- **Environmental gates.** Visitation rate is modelled as `λ = λ_max · f_T · f_W · f_VPD · f_H`
  — temperature, wind, humidity (via vapour-pressure deficit), and hour-of-day gates — so counts
  from a cold, windy hour are comparable to a warm, calm one, and weather is treated as the
  confounder it is (it moves both visits *and* fruit set).
- **Saturating dose–response.** Fruit set is a bounded proportion following a decelerating,
  saturating curve — a Hill / logistic form
  `FruitSet(D) = P_self + (P_max − P_self) · D^n/(k^n + D^n)`, fitted as a **binomial GLMM** with
  plant/orchard/date random effects and a flower-type (bisexual vs. functionally male) structural
  component. Small-sample, no-anchor regime → **Bayesian fit with cross-crop priors** and
  **uncertainty propagated into every yield number** (no point estimate without an interval).

Full derivation, formulas, pomegranate biology, and the `src/ml_models/` build order live in the
ML-team modeling reference (`Bee_a_Hero_Unified_ML_Research`). **Status:** no fitted pomegranate
curve yet — the blocker is data collection (fruit-set labels, flower-type labels, cross-time
flower identity), not tooling. The honest interim deliverable is the modeling code validated on
simulated data.

## 4. LLM stage — grounded reporting 🟡

`src/llm_reporting/` turns model output into a plain-language report under a strict **grounding
contract**: use only fields present in the structured input; never compute or invent a number;
omit missing topics; never speculate on causes (weather, pests, species effectiveness) unless
that exact field is present; always label model estimates as estimates. The model *uses* weather
as a fitted feature; the report may only *restate* weather the pipeline measured. **Status:**
prompt + schema designed; scaffolding.

---

## 5. Repository structure

```
notebooks/                       # shared slot scheme (all branches); §5 of 02_cv = one-shot test
├── 00_data_ready.ipynb          # data prep
├── 01_eda.ipynb                 # exploratory analysis
├── 02_cv.ipynb                  # CV — detection + tracking + landings   (import src)
├── 03_ml.ipynb                  # ML slot
└── 04_llm.ipynb                 # LLM slot
full_notebooks/                  # self-contained variants (no import src) for a clean machine
├── 00_data_ready_full.ipynb
├── 01_eda_full.ipynb
└── 02_cv_full.ipynb
src/
├── data_pipeline/               # iNaturalist prep, EDA, label tools, flower dataset builders
├── cv_engine/
│   ├── prepare_detect.py        # build YOLO detection sets (flower + 5-class insect)
│   ├── train.py                 # YOLO26 fine-tuning (imgsz, mixup/copy-paste)
│   ├── video_detect.py          # flower+insect boxes/IDs/type (BoT-SORT) + landing episodes + CSVs
│   ├── honeybee_clf.py          # honeybee-vs-other-bee subclassifier (run on bee crops)
│   └── visit_counter.py         # FlowerTracker + shared helpers
├── ml_models/                   # ML slot (scaffolding)
└── llm_reporting/               # LLM slot (scaffolding)
data/interim/cv_runs/{flower_det2,insect_multidet}_v2_yolo26m/weights/best.pt   # committed (best)
data/interim/cv_runs/honeybee_clf/best.pt                                       # committed (best)
test_video_result/csv/ALL_landings.csv, ALL_flower_summary.csv                  # committed team CSVs
test_video_result/videos/<video>_annotated.mp4                                  # annotated videos (local)
```

## 6. Retraining the detectors (Act-2, on the server)

Committed weights let you run inference immediately. To **retrain**, download these into
`data/raw/` (git-ignored) and run `prepare_detect` then `train` — see
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
    --name flower_det2_v2_yolo26m --model yolo26m.pt --epochs 100 --imgsz 768 --batch 8
python -m src.cv_engine.train --data data/interim/insect_multidet/data.yaml \
    --name insect_multidet_v2_yolo26m --model yolo26m.pt --epochs 70 --imgsz 768 --batch 4 \
    --mixup 0.1 --copy-paste 0.1
python -m src.cv_engine.honeybee_clf --model efficientnet_b0 --epochs 30 --batch 16
```

## 7. Reproducibility

Everything is seeded (`SEED = 42` in `src/config.py`) and deterministic; paths resolve relative
to the repo root, so there is nothing machine-specific to configure. Training forces the `fork`
start method (Python 3.14 fix) and is Windows-safe.

## 8. Team

| Member | Role |
|--------|------|
| **Asif Habilov** | Team lead — planning, research direction, ML/CV engineering, QA |
| **Raul Ibrahimov** | Data research & ML engineering — dataset curation, model training |
| **Narmin Dirayeva** | LLM & ML engineering — reporting, model development |
| **Khaver** | Data & LLM — collection, annotation, quality |
