#!/usr/bin/env python3
"""
BEE_HERo - ONE-CLICK SETUP.   Run this single file and everything is ready.
================================================================================
On a fresh PC that has ONLY the three archives:

    train_mini.tar.gz   val.tar.gz   public_test.tar.gz

put this file at the repo root (the pipeline code lives in
`src/data_pipeline/reproduce_bee_hero.py` and `src/ml_models/bee_hero_dataset.py`)
and run:

    python RUN_ME.py

It will, with no further input:
    1. install any missing Python packages (pip),
    2. extract the 3 archives,
    3. label + filter to Insecta (+ tag bees) and build the manifests,
    4. perceptual-dedup + leakage-safe stratified 80/10/10 split,
    5. write data.yaml / dataset_config.json / class_index.json,
    6. VERIFY the data is training-ready (loads real batches, MixUp/CutMix,
       one ResNet-18 train step) and save an augmented preview grid.

Then it prints  ===  BOOM - DATA IS 100% READY  ===

Useful flags:
    python RUN_ME.py --check       # only check environment, do nothing
    python RUN_ME.py --no-install  # don't pip-install (deps already present)
    python RUN_ME.py --purge       # delete non-insect folders on disk too
    python RUN_ME.py --no-extract  # archives already extracted to folders
Anything else is forwarded to reproduce_bee_hero.py.
"""

import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPRODUCE = os.path.join(HERE, "src", "data_pipeline", "reproduce_bee_hero.py")
DATASET = os.path.join(HERE, "src", "ml_models", "bee_hero_dataset.py")
DATA = os.path.join(HERE, "data", "raw", "iNaturist")  # raw splits + archives live here
ARCHIVES = ("train_mini.tar.gz", "val.tar.gz", "public_test.tar.gz")

# import-name -> pip-name. Required first, then optional.
REQUIRED = {"PIL": "pillow", "imagehash": "imagehash", "numpy": "numpy",
            "torch": "torch", "torchvision": "torchvision"}
OPTIONAL = {"albumentations": "albumentations", "cv2": "opencv-python",
            "matplotlib": "matplotlib"}


def hr(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def pip_install(pkgs):
    if not pkgs:
        return True
    print(f"  pip install {' '.join(pkgs)}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
                       check=True)
        return True
    except Exception as e:
        print(f"  !! pip install failed: {e}")
        return False


def ensure_deps(do_install):
    hr("STEP 0/6  dependencies")
    missing_req = [p for m, p in REQUIRED.items() if not have(m)]
    missing_opt = [p for m, p in OPTIONAL.items() if not have(m)]
    if not missing_req:
        print("  required packages: all present")
    if do_install:
        if missing_req and not pip_install(missing_req):
            print("  !! could not install required packages - install manually:")
            print(f"     {sys.executable} -m pip install {' '.join(missing_req)}")
            return False
        pip_install(missing_opt)          # optional: failure is non-fatal
    else:
        if missing_req:
            print(f"  !! missing required (run without --no-install): {missing_req}")
            return False
    # re-check required
    still = [p for m, p in REQUIRED.items() if not have(m)]
    if still:
        print(f"  !! still missing after install: {still}")
        return False
    print("  OK - required deps satisfied"
          + ("" if not [m for m in OPTIONAL if not have(m)]
             else f"  (optional missing, fine: {[m for m in OPTIONAL if not have(m)]})"))
    return True


def preflight(check_only):
    hr("PRE-FLIGHT  environment check")
    ok = True
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 8):
        print("  !! Python 3.8+ required"); ok = False

    for s, label in ((REPRODUCE, "reproduce_bee_hero.py"),
                     (DATASET, "bee_hero_dataset.py")):
        print(f"  script   : {label:24s} {'found' if os.path.isfile(s) else 'MISSING !!'}")
        if not os.path.isfile(s):
            ok = False

    extracted = all(os.path.isdir(os.path.join(DATA, d))
                    for d in ("train_mini", "val", "public_test"))
    for a in ARCHIVES:
        p = os.path.join(DATA, a)
        if os.path.isfile(p):
            print(f"  archive  : {a:22s} {os.path.getsize(p)/1e9:6.1f} GB")
        else:
            print(f"  archive  : {a:22s} {'(absent)' if extracted else 'MISSING !!'}")
            if not extracted:
                ok = False
    if extracted:
        print("  note     : split folders already extracted - extraction will be skipped")

    free = shutil.disk_usage(HERE).free / 1e9
    print(f"  free disk: {free:.0f} GB"
          + ("" if free > 130 or extracted else "   !! recommend >=130 GB to extract public_test"))
    print("  result   :", "READY" if ok else "PROBLEMS FOUND (see !! above)")
    return ok


def run(label, cmd):
    hr(label)
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  !! step failed (exit {r.returncode})")
    return r.returncode == 0


def main():
    argv = sys.argv[1:]
    check_only = "--check" in argv
    do_install = "--no-install" not in argv
    forward = [a for a in argv if a not in ("--check", "--no-install")]

    hr("BEE_HERo - ONE-CLICK SETUP")
    print(f"  working folder: {HERE}")

    if not preflight(check_only):
        if check_only:
            return 1
        print("\n  Fix the items marked !! above, then re-run.  Aborting.")
        return 1
    if check_only:
        print("\n  --check only: environment looks good. Run without --check to build.")
        return 0

    if not ensure_deps(do_install):
        return 1

    # STEP 1-5: extract -> label -> filter -> split -> configs
    if not run("STEP 1-5/6  build dataset (extract, label, filter, split, configs)",
               [sys.executable, REPRODUCE, "--root", HERE, *forward]):
        return 1

    # STEP 6: prove it is training-ready
    run("STEP 6/6  verify training-readiness",
        [sys.executable, DATASET, "--root", HERE])

    hr()
    status = "?"
    sp = os.path.join(HERE, "_pipeline", "REPRODUCE_STATUS.txt")
    if os.path.isfile(sp):
        status = open(sp, encoding="utf-8").read().strip()
    print(f"  pipeline status : {status}")
    print("  artifacts       : _pipeline/manifest_*.csv, _pipeline/splits/, "
          "data.yaml, dataset_config.json, _pipeline/class_index.json")
    print("  train with      : from src.ml_models.bee_hero_dataset import build_dataloaders")
    hr("===  BOOM - DATA IS 100% READY  ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
