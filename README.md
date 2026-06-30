# BEE_HERo

End-to-end **dataset cleaning + EDA + CV-readiness pipeline** for an iNaturalist-style
insect image dataset, with a focus on retaining all **Insecta** species and tagging the
**bee** families as a subset.

> **Dataset reality:** this is an **image-classification** dataset (per-species folders,
> taxonomy-encoded names), **not** object detection. There are **no bounding boxes**, so
> detection-only steps are executed as their classification equivalents and bbox-specific
> items are marked N/A.

## Status
✅ **Pipeline COMPLETED_OK** (last run 2026-06-29). Archives left untouched — everything is reversible.

## Dataset
- **Source:** iNaturalist (`train_mini.tar.gz`, `val.tar.gz`, `public_test.tar.gz`).
- **Location:** all raw data (archives + extracted `train_mini/`, `val/`, `public_test/`) lives under `data/raw/iNaturist/`.
- **Folder naming:** `ID_Kingdom_Phylum_Class_Order_Family_Genus_species`.
- **Classes (`nc`):** **2526** unique Insecta species.
- **Orders represented:** 17.

| Split | Folders kept | Images kept | Bee images | Notes |
|-------|-------------|-------------|------------|-------|
| `train_mini` | 2526 | 126,300 | 3,100 | filtered to Insecta |
| `val` | 2526 | 25,260 | 620 | filtered to Insecta |
| `public_test` | — | 500,000 | — | flat/**unlabeled**, left intact (profiled only) |
| **Total (labeled)** | **5,052** | **151,560** | **3,720** | |

- Folders audited: **19,825** → kept **5,052**, removed **14,773** non-insect.
- Corrupt images removed: **0** (every retained image opened + verified with PIL).

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
- **Split overlap:** 0 species only-in-train, 0 only-in-val (perfectly aligned).
- ⚠️ **Leakage:** 1 cross-split duplicate group found (perceptual pHash). Review before training.

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

## How to re-run
```bash
cd "/c/Users/narim/Desktop/Bee-A-Hero"
rm -f _pipeline/STATUS.txt _pipeline/pipeline.log
python src/data_pipeline/pipeline.py > _pipeline/pipeline_console.log 2>&1 &
# poll _pipeline/STATUS.txt until it reads COMPLETED_OK (~40–70 min)
```
The pipeline is idempotent: it re-walks everything, re-removes non-insect folders, keeps
insects, and rebuilds manifests. The `.tar.gz` archives are never modified.

## Daily Activities
- **2026-06-30 — Raul/Data —** Reorganized the repo so all raw data (archives + extracted
  `train_mini/`, `val/`, `public_test/`) lives under `data/raw/iNaturist/`. Repointed every
  script (`pipeline.py`, `reproduce_bee_hero.py`, `resplit_option3.py`, `bee_hero_dataset.py`,
  `RUN_ME.py`) to read splits/archives from there; code stays in `src/`, artifacts in
  `_pipeline/`. Updated `data.yaml` `path:`, `dataset_config.json` `root`, and `.gitignore`.
  Existing manifests/splits remain valid (paths resolve into the new data dir) — verified the
  loader builds **2526** classes / train-val-test = 121,226 / 15,157 / 15,155.
