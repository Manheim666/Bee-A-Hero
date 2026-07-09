# Data-prep results snapshot (for review)

Lightweight snapshot of the iNaturalist data-preparation outputs so reviewers can
see the results **without downloading the 14 GB image dataset**. Regenerate any
time with `bash scripts/run_pipeline.sh` (seeded → identical output).

## Figures (`01_eda.ipynb`)
| File | Shows |
|---|---|
| `class_and_order_distribution.png` | per-class image counts + images per taxonomic order |
| `split_bee_order_heatmap.png` | 70/15/15 split pie, bee vs non-bee per split, order×split heatmap |
| `image_statistics.png` | resolution histograms + scatter, aspect boxplot, brightness & blur |
| `quality_mode_contrast.png` | colour-mode counts, contrast distribution |
| `sample_grid.png` | random sample images across train/val/test |

## Report JSONs
| File | Contents |
|---|---|
| `split_summary.json` | 2526 classes · 151,545 imgs · 70/15/15 = 106,077/22,734/22,734 · public_test → 22,734 |
| `validation_report.json` | Phase-5 integrity checklist (all pass, 0 orphans/dups/leakage) |
| `data_ready_report.json` | full `00_data_ready` gate → `DATA_READY: true` |
| `eda_summary.json` | overview + imbalance (Gini≈0) + image-stat + quality summary |

> Snapshot only — the live artifacts live (git-ignored) under `data/interim/`.
