BEE_HERo - ONE-CLICK SETUP
================================================================================

HOW TO USE (3 steps)
--------------------------------------------------------------------------------
1. Unzip this file into any folder.

2. Copy your THREE archives into the SAME folder, named EXACTLY:
       train_mini.tar.gz
       val.tar.gz
       public_test.tar.gz
   (If your "test" archive has another name, rename it to public_test.tar.gz)

   The folder should now look like:
       RUN_ME.py
       reproduce_bee_hero.py
       bee_hero_dataset.py
       bee_hero_dataready.ipynb
       train_mini.tar.gz
       val.tar.gz
       public_test.tar.gz

3. Open a terminal in that folder and run:
       python RUN_ME.py

   That's it. It will automatically:
     - install any missing Python packages
     - extract the 3 archives
     - label + filter to Insecta (and tag the bee families)
     - build the manifests
     - de-duplicate + make a leakage-safe 80/10/10 train/val/test split
     - write data.yaml / dataset_config.json / class_index.json
     - verify the data is training-ready (loads real batches, runs one
       ResNet-18 training step) and save an augmented preview image
   and finish with:  ===  BOOM - DATA IS 100% READY  ===


REQUIREMENTS
--------------------------------------------------------------------------------
- Python 3.8 or newer.
- About 130 GB free disk space (extracting public_test = ~500k images).
  If you do NOT need the unlabeled public_test set, you can leave that one
  archive out - the labeled train/val/test still build fine.
- No GPU needed to PREPARE the data. (A GPU is only needed later to TRAIN.)


HANDY OPTIONS
--------------------------------------------------------------------------------
  python RUN_ME.py --check        only check the environment, build nothing
  python RUN_ME.py --no-install   skip pip install (packages already present)
  python RUN_ME.py --no-extract   archives are already extracted to folders
  python RUN_ME.py --purge        also delete non-insect folders from disk


WHAT YOU GET (outputs)
--------------------------------------------------------------------------------
  train_mini/  val/  public_test/        extracted image folders
  _pipeline/manifest_*.csv               per-image labels (split,path,class,...)
  _pipeline/splits/train.txt|val.txt|test.txt   the leakage-safe split lists
  _pipeline/splits/split_assignments.csv path -> class + split (with phash)
  _pipeline/class_index.json             class_id -> 0..2525 index
  data.yaml  dataset_config.json         dataset config (nc=2526)
  _pipeline/eda/augmented_preview.png    visual check of the augmentation


TO TRAIN LATER
--------------------------------------------------------------------------------
  from bee_hero_dataset import build_dataloaders, mixup_cutmix, SoftTargetCrossEntropy
  train_dl, val_dl, test_dl, class_to_idx = build_dataloaders(batch_size=32)
  # then write your epoch loop, or open bee_hero_dataready.ipynb to explore.
