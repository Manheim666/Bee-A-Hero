# Training datasets to download (detection approach — bbox, no segmentation)

Goal: detect each **flower** (bbox), detect each **bee/insect** (bbox + track ID),
count distinct insects landing per flower. Weak point = detectors on **video**, so
we add real, video-domain, box-labelled data and retrain the detectors.

Download each into the folder shown under `data/raw/`, then tell me — I retrain.

## 1. Bee detector on VIDEO (highest priority)
**Bee Detection in the Wild** — 9,664 images extracted from **video streams**,
YOLO bounding boxes, MIT licence. Species incl. *Bombus* + *Apis mellifera*.
Splits: train 6722 / val 1915 / test 997. Best match for video generalisation.
- https://www.kaggle.com/datasets/birdy654/bee-detection-in-the-wild
- Put in: `data/raw/bee_wild/`  (Kaggle → Download; needs a free Kaggle account)

## 2. Multi-insect TYPE detector (fixes bee/fly/ant confusion)
**Honeybee pollinator (multi-class)** — classes: bee, butterfly, ant, bombus, fly,
hoverfly, ladybug, pollinator/non_pollinator. Learning types as *detection classes*
avoids the separate-classifier confusion entirely.
- https://universe.roboflow.com/search?q=class:pollenbee
- Also: https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb (909 imgs, worker/drone/queen/pollen)
- Also: https://universe.roboflow.com/honey-bee-project/honey-bee-project (962 imgs)
- Put in: `data/raw/pollinator_multiclass/`  (Roboflow → Export → YOLOv8 → download zip)

## 3. Flower detector (separate box per flower on video)
**Flower detection** — multiple-flower bounding boxes, exports YOLOv8/v11.
- https://universe.roboflow.com/flower-42dyl/flower-detection-hiutj/dataset/3
- Browse more: https://universe.roboflow.com/search?q=class:flower
- Put in: `data/raw/flower_det_rf/`  (Roboflow → Export → YOLOv8 → download zip)

## 4. (Optional) Flower-visitor time-lapse, 6 insect taxa with boxes
Bees/wasps, flies, beetles, ants, spiders, bugs — annotated across image sequences.
Good extra context for insect-on-flower + types.
- Paper: https://www.nature.com/articles/s41598-025-16140-z (follow its Data Availability link)

---

## Fastest path (no manual downloads)
Get a **free Roboflow API key** (roboflow.com → Settings → API key) and put it in a
`.env` at the repo root as `ROBOFLOW_API_KEY=xxxx`. Then I pull datasets 2 & 3
automatically via the `roboflow` pip package (already installed). Kaggle set (1)
still needs a manual download (or a `~/.kaggle/kaggle.json`).

## After data is in place
I retrain two YOLO26 detectors on the combined real + existing data:
- **flower** (single class) and **insect/bee** (multi-class types),
then run the existing detection + BoT-SORT tracking + per-flower counting pipeline
(`src/cv_engine/visit_counter.py`) — bbox per flower, bbox + ID per bee, distinct
count per flower. No segmentation.
