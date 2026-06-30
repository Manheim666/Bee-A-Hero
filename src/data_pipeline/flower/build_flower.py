#!/usr/bin/env python3
"""
build_flower.py  -  ONE command to rebuild the whole Flower dataset from the zip.

Prerequisite (the only thing a teammate must provide):
    data/raw/Flower/archive (1).zip      <- the Kaggle "Flower Classification" zip

Run:
    python src/data_pipeline/flower/build_flower.py

It produces, with relative paths (works on any machine / any clone):
    data/processed/flower/classification/   <- 10-class images in Train/Val/Test folders
    data/processed/flower/yolo/             <- object-detection-ready (YOLO format)
        images/{train,val,test}/  labels/{train,val,test}/
        data.yaml  classes.txt  labels.csv  annotations.csv

Nothing here is committed to git (the data is .gitignored); only this script is.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # .../Bee-A-Hero
ZIP = REPO / "data" / "raw" / "Flower" / "archive (1).zip"
CLASSIFICATION = REPO / "data" / "processed" / "flower" / "classification"
YOLO = REPO / "data" / "processed" / "flower" / "yolo"


def run(label, cmd):
    print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70)
    print("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([sys.executable, *map(str, cmd)])
    if r.returncode != 0:
        sys.exit(f"!! step failed: {label} (exit {r.returncode})")


def main():
    if not ZIP.exists():
        sys.exit(f"!! source zip not found:\n   {ZIP}\n"
                 f"   Put the Kaggle 'Flower Classification' zip there, then re-run.")

    # 1) merge V1+V2 -> unified labeled classification dataset
    run("STEP 1/2  merge V1+V2 -> classification",
        [HERE / "merge_flowers.py", "--zip", ZIP, "--out", CLASSIFICATION])

    # 2) auto-generate YOLO detection labels from the classification set
    run("STEP 2/2  build YOLO object-detection dataset",
        [HERE / "make_detection_dataset.py", "--src", CLASSIFICATION, "--out", YOLO])

    print("\n" + "=" * 70)
    print("FLOWER DATA READY")
    print("=" * 70)
    print(f"  classification : {CLASSIFICATION}")
    print(f"  detection(YOLO): {YOLO}")
    print(f"  train YOLO with: yolo detect train data=\"{YOLO / 'data.yaml'}\" model=yolo11n.pt")


if __name__ == "__main__":
    main()
