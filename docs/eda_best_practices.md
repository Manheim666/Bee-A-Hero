# EDA & Data-Quality Best Practices (Bee-A-Hero iNaturalist prep)

Research notes backing the checks implemented in `src/data_pipeline/eda.py` and
`notebooks/01_eda.ipynb`. Goal: go beyond the prior `_pipeline/eda` outputs
(order distribution, class-size histogram, resolution scatter, aspect histogram,
sample grid) and make the dataset as clean and reliable as possible.

## What the sources say

| Practice | Why it matters here | Source |
|---|---|---|
| **Audit for blur / under-over-exposure / odd sizes / (near) duplicates** | These are the "common real-world issues" that silently degrade classifiers; fine-grained insect ID is especially sensitive to blur obscuring diagnostic features. | CleanVision (cleanlab) |
| **Purge near-duplicates across train/test** | Even curated academic sets (CIFAR) contain near-dup leakage that inflates test accuracy; must be caught with perceptual hashing and kept split-safe. | "Do We Train on Test Data? Purging CIFAR of Near-Duplicates", arXiv:1902.00423 |
| **Hash-based dedup + leakage detection scales** | Perceptual/embedding hashing is the standard scalable method for large image sets. | arXiv:2304.02296 |
| **Data quality has measurable model impact** | A data-centric pass (resolution, exposure, mislabels) improves models more than architecture tweaks at this scale. | arXiv:2509.24420 |
| **Report class imbalance quantitatively** | Imbalance ratio / Gini drive the loss choice downstream (weighted CE, label smoothing, over/under-sampling). | Paperspace "Class Imbalance in Image Datasets"; class-imbalance survey |
| **Inspect resolution/aspect/color distributions before modeling** | Input-size and normalization choices key off these; grayscale contaminants hurt color-reliant features. | PyData EDA-for-vision; general CV EDA guidance |

## Check checklist (implemented)

**Overview** — #images, #classes, class frequencies, split distribution, sizes.
**Image stats** — resolution (w/h) distribution, aspect ratios, brightness,
contrast, color mode / grayscale detection.
**Quality** — corrupted/unreadable (full scan), blank (low-variance), blurry
(variance-of-Laplacian), near-duplicate groups (perceptual pHash), exact
duplicates (md5, handled in Phase 4).
**Annotation** — missing labels, labels without images, invalid category ids,
empty/duplicate annotations, cross-split leakage (all in `label_tools.validate`).
**Imbalance** — per-class counts, imbalance ratio, Gini, long-tail view.
**Bee subset** — bee vs non-bee balance per split (project-specific).
**Visualizations** — class & order histograms, resolution histogram + scatter,
aspect boxplot, brightness/blur distributions, split pie charts, per-order
heatmap, deterministic sample grid.

## Reused & improved from prior work
- `_pipeline/eda` (teammate) established order/class/resolution/aspect/sample-grid
  plots — reproduced and extended with brightness, contrast, **blur**, grayscale,
  blank, near-duplicate, split-distribution, and bee-subset analyses.
- Desktop references (`insect-detect-*`, Roboflow `*.coco` bee sets) inform the
  downstream detection stage; this EDA stays classification-focused.

## Sources
- [CleanVision — Audit your Image Data](https://cleanlab.ai/blog/cleanvision/)
- [Do We Train on Test Data? Purging CIFAR of Near-Duplicates (arXiv:1902.00423)](https://arxiv.org/pdf/1902.00423)
- [Efficient Deduplication and Leakage Detection in Large Scale Image Datasets (arXiv:2304.02296)](https://arxiv.org/html/2304.02296v2)
- [A Data-Centric Perspective on Image Data Quality (arXiv:2509.24420)](https://arxiv.org/html/2509.24420v1)
- [Class Imbalance in Image Datasets (Paperspace)](https://blog.paperspace.com/class-imbalance-in-image-datasets/)
- [EDA & Visualization for Image Challenges (PyData Eindhoven 2019)](https://pydata.org/eindhoven2019/schedule/presentation/15/)
