# BEE_HERo Dataset Pipeline Report

_Generated 2026-06-29T06:54:27_

## ☀️ Morning Summary (TL;DR)

The pipeline finished, monitored in an active session (no auto-shutdown — left running for you). Here is what happened:

- **Done:** extracted all 3 archives, filtered to insects/bees, built manifests, ran full EDA + quality eval, wrote configs.
- **Folders audited:** 19825 (across train_mini + val) → kept **5052** species folders = **2526 unique species** × 2 splits; removed **14773** non-insect folders.
- **Images retained:** **151560** insect images (of which **3720** are bees); 0 corrupt removed.
- **Classes for training (nc):** **2526** (see `data.yaml` / `dataset_config.json`).
- **Class balance:** imbalance ratio **1.0**, Gini **0.0** → Mild imbalance: standard CrossEntropy with light class weights is sufficient.
- **Data leakage:** WARNING: 1 cross-split duplicate groups found.
- **Completeness:** 100.0% of retained images are label-aligned.
- **Your `.tar.gz` archives were left untouched** — everything is reversible.
- **Files to look at:** this `REPORT.md`, `manifest_all.csv`, the `eda/` plots (`sample_grid.png` first), `phase6_quality.json`, and `run_all.log`/`pipeline.log` for the full trace.

Full phase-by-phase detail below.

---

**Dataset reality:** iNaturalist-style image *classification* (per-species folders, taxonomy-encoded names). No bounding boxes exist, so detection-only steps are executed as their classification equivalents and bbox-specific items are marked N/A.

## Phase 1 - Semantic Filtering & Class Cleansing
Criteria: keep a species folder iff its taxonomic **Class == Insecta**; tag bees where Order==Hymenoptera and Family in ['Andrenidae', 'Apidae', 'Colletidae', 'Halictidae', 'Megachilidae', 'Melittidae', 'Stenotritidae'].

- Folders audited: **19825**
- Folders kept (Insecta): **5052**
- Folders removed (non-insect): **14773**
- Images retained: **151560** (bee images: **3720**)
- Corrupt images removed: **0**

Per-split:

| split | audited folders | kept | removed | kept images | bees |
|---|---|---|---|---|---|
| train_mini | 9825 | 2526 | 7299 | 126300 | 3100 |
| val | 10000 | 2526 | 7474 | 25260 | 620 |

`public_test`: 500000 images, flat/unlabeled, left intact (cannot be class-filtered without labels).

## Phase 2 - Annotation Alignment & Validation
No bbox annotations exist. Classification-equivalent actions performed:
- Label = parent species folder (taxonomy-derived). Manifests written to `_pipeline/manifest_*.csv` with [split, path, class_id, folder, order, family, genus, species, is_bee].
- Integrity validation: every retained image opened+verified with PIL; 0 corrupt files removed.
- Orphans: by construction every retained image lives under exactly one class folder, so there are no orphan label/image mismatches. Empty folders removed.
- Final synchronized labeled images: **151560**.
- Coordinate validation (out-of-bounds / inverted / zero-area boxes): **N/A** (no boxes).

## Phase 3 - Exploratory Data Analysis
- Retained insect species (classes): **2526**, orders: **17**, total labeled images: **151560**.
- Resolution width (min/med/max): [172, 500, 500], height: [160, 385, 500] (sampled 6000).
- Color modes: {'RGB': 6000}.
- Plots in `_pipeline/eda/`: `dist_by_order.png`, `class_size_hist.png`, `resolution_scatter.png`, `aspect_ratio_hist.png`, `sample_grid.png`.
- Bounding-box size / aspect / anchor analysis: **N/A** (classification dataset). Object-scale concerns are instead addressed by input-resolution choice + augmentation.

## Phase 4 - CV Readiness & Augmentation Strategy
- Split check: train species=2526 (126300 imgs), val species=2526 (25260 imgs); species only-in-train=0, only-in-val=0.
- **Augmentation blueprint (Albumentations recommended):** RandomResizedCrop(224), HorizontalFlip, ShiftScaleRotate, ColorJitter/HueSaturationValue + RandomBrightnessContrast (simulate natural lighting), CoarseDropout, plus **MixUp/CutMix** at the batch level. Mosaic is detection-oriented and optional/low-value for classification. Always Normalize(ImageNet stats).
- **6GB VRAM prep:** img size 224, batch 32 (use AMP/torch.cuda.amp to push to 48-64), `num_workers=4-6`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`. WeightedRandomSampler if imbalance is severe.

## Phase 5 - Folder Integrity & Path Mapping
- Original folder hierarchy preserved; filtering done in-place. Original `.tar.gz` archives untouched.
- Wrote `data.yaml` and `dataset_config.json` at repo root with **nc=2526** ordered insect class names.

## Phase 6 - Quality Evaluation
- **Completeness:** 100.0% of retained images are perfectly label-aligned.
- **Class balance:** 2526 classes, per-class min=60, max=60, imbalance ratio=1.0, Gini=0.0.
  - Recommendation: Mild imbalance: standard CrossEntropy with light class weights is sufficient.
- **Leakage:** WARNING: 1 cross-split duplicate groups found (perceptual pHash, up to 15000 imgs/split; 1 cross-split duplicate groups).

Artifacts: `_pipeline/REPORT.md`, `manifest_*.csv`, `eda/`, `phase4_split_check.json`, `phase6_quality.json`, `pipeline.log`.
