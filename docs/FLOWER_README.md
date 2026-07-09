# Flower dataset — rebuild from the zip

The Flower images are **not** in git (too large). Anyone with the Kaggle
*Flower Classification* zip can regenerate everything locally with one command.
Outputs use relative paths and are `.gitignored`.

## 1. Provide the source
Place the zip at:
```
data/raw/Flower/archive (1).zip
```

## 2. Build
```bash
python src/data_pipeline/flower/build_flower.py
```
This runs two steps:
1. `merge_flowers.py` — merges Flower V1 (5 classes) + V2 (10 classes) into one
   labeled set, deduping filename collisions across versions.
2. `make_detection_dataset.py` — auto-generates YOLO detection labels (OpenCV
   GrabCut foreground boxes) from the merged classification images.

## 3. Outputs
| Path | Contents |
|------|----------|
| `data/processed/flower/classification/` | `Training/Validation/Testing Data/<Class>/` — 10 classes (Aster, Daisy, Iris, Lavender, Lily, Marigold, Orchid, Poppy, Rose, Sunflower) |
| `data/processed/flower/yolo/` | `images/` + `labels/` (YOLO txt), `data.yaml`, `classes.txt`, `labels.csv` (image→class), `annotations.csv` (image→bbox) |

Train a detector:
```bash
yolo detect train data="data/processed/flower/yolo/data.yaml" model=yolo11n.pt epochs=50
```

## Notes
- Counts: classification = 30,813 images (train 20,000 / val 7,500 / test 3,313);
  YOLO images and labels match 1:1 per split.
- The source is a *classification* dataset, so the detection boxes are
  **auto-generated**, not human-annotated. ~32% fall back to a full-image box on
  busy/low-contrast scenes. Fine to prototype with; spot-check before treating as
  ground truth.
- The build is re-runnable and idempotent; outputs are regenerated under `data/`.
