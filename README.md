# BEE_HERo

End-to-end **dataset cleaning + EDA + CV-readiness pipeline** for an iNaturalist-style
insect image dataset, with a focus on retaining all **Insecta** species and tagging the
**bee** families as a subset.

> **Dataset reality:** this is an **image-classification** dataset (per-species folders,
> taxonomy-encoded names), **not** object detection. There are **no bounding boxes**, so
> detection-only steps are executed as their classification equivalents and bbox-specific
> items are marked N/A.

## Status
✅ **Data-ready pipeline rebuilt & verified (2026-07-01).** Clean, script-driven,
reproducible (seed=42), fully **non-destructive** (removed images are *moved* to
`data/_backup/removed/`, never deleted). Entry point: **`notebooks/00_data_ready.ipynb`**
→ EDA in **`notebooks/01_eda.ipynb`**. Reusable logic in
`src/data_pipeline/{inaturalist_prep,label_tools,eda}.py`; methodology in
`docs/DATA_PIPELINE.md` and `docs/eda_best_practices.md`.

Quick start:
```bash
bash scripts/setup_env.sh                     # create .venv + install pinned deps
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/00_data_ready.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
```

## Dataset
- **Source:** iNaturalist (`train_mini.tar.gz`, `val.tar.gz`, `public_test.tar.gz`).
- **Location:** all raw data (archives + extracted `train_mini/`, `val/`, `public_test/`) lives under `data/raw/iNaturist/`.
- **Folder naming:** `ID_Kingdom_Phylum_Class_Order_Family_Genus_species`.
- **Classes (`nc`):** **2526** unique Insecta species.
- **Orders represented:** 17.

**Model-ready split** — the labeled images (`train_mini` + `val`, pooled) are exact-duplicate
de-duplicated (md5) and stratified per species into a **70/15/15** train/val/test split
(largest-remainder, no truncation). The split is a **manifest** — images are never moved
between split folders, so there is no duplication or cross-split leakage:

| Split | Images | Share |
|-------|--------|-------|
| `train` | 106,077 | 70% |
| `val` | 22,734 | 15% |
| `test` | 22,734 | 15% |
| **Total (labeled)** | **151,545** | 100% |

> **Before splitting:** `train_mini` (126,300 imgs) + `val` (25,260) = **151,560** labeled Insecta
> images across **2526** species; **15** exact duplicates removed during dedup → **151,545**.
>
> **`public_test` (500,000 flat, *unlabeled* images) is NOT part of this split.** It has no labels
> (`annotations: 0`), so it can't be trained/evaluated on — it's kept aside for
> inference / leaderboard submission only.

- Non-Insecta folders removed (moved to backup): **230 per split** (13,800 images).
- Corrupt images: **0** (full PIL verify over all 151,545 images).
- Split assignment/manifest: `data/interim/manifests/split_manifest.csv`; clean per-split
  COCO labels: `data/interim/labels/{train,val,test}.json`.

## Bee subset
Tagged where `Order == Hymenoptera` and `Family` ∈ {Andrenidae, Apidae, Colletidae,
Halictidae, Megachilidae, Melittidae, Stenotritidae}. The `is_bee` flag is recorded per
image in the manifests.

## Pipeline phases
1. **Semantic filtering & class cleansing** — keep a species folder iff taxonomic `Class == Insecta`; tag bee families.
2. **Annotation alignment & validation** — labels derived from parent species folder; integrity-verify all images; write manifests. (Bbox coordinate validation: N/A.)
3. **Exploratory data analysis** — class/order distributions, resolution & aspect-ratio analysis, sample grid.
4. **CV readiness & augmentation strategy** — split consistency check + augmentation blueprint + 6 GB-VRAM training recipe.
5. **Folder integrity & path mapping** — original hierarchy preserved in place; archives untouched; configs written.
6. **Quality evaluation** — completeness, class balance, and cross-split leakage scan.

## Quality summary (Phase 6)
- **Completeness:** 100.0% of retained images are label-aligned.
- **Class balance:** 60 images per class (min = max = 60), imbalance ratio **1.0**, Gini **0.0**
  → standard CrossEntropy with light class weights is sufficient.
- **Split overlap:** every one of the 2526 classes appears in all of train/val/test.
- **Leakage:** 0 cross-split duplicate paths; 0 perceptual near-duplicate groups (sampled).

## Training recipe (6 GB VRAM)
- Image size 224, batch 32 (AMP can push to 48–64), `num_workers=4–6`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor=2`.
- **Augmentation (Albumentations):** RandomResizedCrop(224), HorizontalFlip, ShiftScaleRotate,
  ColorJitter/HueSaturationValue + RandomBrightnessContrast, CoarseDropout, plus batch-level
  **MixUp/CutMix**. Always `Normalize(ImageNet stats)`. (Mosaic is detection-oriented — skip.)
- Use `WeightedRandomSampler` only if imbalance becomes severe (not needed here).

## Layout
```
Bee-A-Hero/
├── RUN_ME.py                    # one-click entrypoint (repo root)
├── data/
│   └── raw/
│       └── iNaturist/           # ALL raw data lives here:
│           ├── train_mini/      #   per-species folders (filtered to Insecta)
│           ├── val/             #   per-species folders (filtered to Insecta)
│           ├── public_test/     #   500k flat, unlabeled .jpgs
│           └── *.tar.gz         #   original archives — DO NOT TOUCH
├── data.yaml                    # nc=2526 + ordered class names (path: -> data/raw/iNaturist)
├── dataset_config.json
├── src/
│   ├── data_pipeline/
│   │   ├── pipeline.py          # clean / filter / EDA / manifest pipeline
│   │   ├── reproduce_bee_hero.py# extract → label → split orchestrator
│   │   └── resplit_option3.py   # perceptual-dedup + stratified split
│   └── ml_models/
│       └── bee_hero_dataset.py  # PyTorch Dataset/DataLoader, augmentation, MixUp/CutMix
├── notebooks/
│   └── bee_hero_dataready.ipynb
├── docs/                        # continue.md, SETUP_README.txt
└── _pipeline/                   # data artifacts (read/written by the code above)
    ├── REPORT.md                # full phase-by-phase report
    ├── manifest_all.csv         # [split, path, class_id, folder, order, family, genus, species, is_bee]
    ├── manifest_train_mini.csv
    ├── manifest_val.csv
    ├── phase4_split_check.json
    ├── phase6_quality.json
    ├── splits/                  # train/val/test lists + split_assignments.csv + phash cache
    └── eda/                     # dist_by_order, class_size_hist, resolution_scatter,
                                 # aspect_ratio_hist, sample_grid (.png) + summary csv/json
```

## How to re-run (current pipeline)
```bash
bash scripts/setup_env.sh && source .venv/bin/activate

# option A — one command end-to-end
bash scripts/run_pipeline.sh

# option B — step by step
python -m src.data_pipeline.inaturalist_prep --apply      # filter + dedup + 70/15/15
python -m src.data_pipeline.label_tools                   # regenerate + validate labels
jupyter nbconvert --to notebook --execute notebooks/00_data_ready.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
```
Every step is **idempotent** and paths are repo-root-relative (`src/config.py`). Removed
images are moved to `data/_backup/removed/` (never deleted); raw label JSONs are backed up
under `data/_backup/original_labels/`. See `docs/DATA_PIPELINE.md` for full detail.

> **Legacy note.** The earlier `RUN_ME.py` / `_pipeline/` / `data.yaml` workflow (80/10/10,
> Windows paths) is superseded by the above; `_pipeline/` outputs are now git-ignored and
> archived in the backup.

## Daily Activities
- **2026-06-30 — Raul/Data —** Reorganized the repo so all raw data (archives + extracted
  `train_mini/`, `val/`, `public_test/`) lives under `data/raw/iNaturist/`. Repointed every
  script (`pipeline.py`, `reproduce_bee_hero.py`, `resplit_option3.py`, `bee_hero_dataset.py`,
  `RUN_ME.py`) to read splits/archives from there; code stays in `src/`, artifacts in
  `_pipeline/`. Updated `data.yaml` `path:`, `dataset_config.json` `root`, and `.gitignore`.
  Existing manifests/splits remain valid (paths resolve into the new data dir) — verified the
  loader builds **2526** classes / train-val-test = 121,226 / 15,157 / 15,155.
