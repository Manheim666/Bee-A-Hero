#!/usr/bin/env python3
"""
BEE_HERo - RUN WHEN THE FOLDERS ARE ALREADY EXTRACTED
================================================================================
Use this file if you have ALREADY extracted the three archives into folders and
do NOT have / don't want to re-extract the .tar.gz files.

It needs these three folders (the result of extracting the archives):

    train_mini/    val/    public_test/

You can drop them anywhere sensible - this script will locate them and link them
into the place the pipeline expects (data/raw/iNaturist/). Then just run:

    python RUN_EXTRACTED.py

It will, with no further input:
    0. install the few Python packages the build needs (pillow, imagehash, numpy),
    1. find the extracted train_mini/ val/ public_test/ folders,
    2. link/move them into data/raw/iNaturist/ if they aren't already there,
    3. label + filter to Insecta, dedup, leakage-safe 80/10/10 split, write configs,
    4. (optional) verify training-readiness if torch is installed.

No archives required. Original folders are never modified or deleted.

Flags:
    python RUN_EXTRACTED.py --check        # only locate folders + check, do nothing
    python RUN_EXTRACTED.py --no-install   # skip pip (deps already present)
    python RUN_EXTRACTED.py --copy         # copy folders into place instead of linking
    python RUN_EXTRACTED.py --data PATH    # I'll look for the splits under PATH
"""

import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPRODUCE = os.path.join(HERE, "src", "data_pipeline", "reproduce_bee_hero.py")
DATASET = os.path.join(HERE, "src", "ml_models", "bee_hero_dataset.py")
CANON = os.path.join(HERE, "data", "raw", "iNaturist")   # where the pipeline looks
SPLITS = ("train_mini", "val", "public_test")            # public_test is optional
REQUIRED_SPLITS = ("train_mini", "val")                  # build needs these two

# import-name -> pip-name. These are all the BUILD needs.
BUILD_DEPS = {"PIL": "pillow", "imagehash": "imagehash", "numpy": "numpy"}
# Only the optional STEP "verify" needs these; failure to install is non-fatal.
VERIFY_DEPS = {"torch": "torch", "torchvision": "torchvision"}


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


def is_split_dir(path, name):
    """True if <path>/<name> exists and looks non-empty."""
    d = os.path.join(path, name)
    try:
        return os.path.isdir(d) and any(os.scandir(d))
    except Exception:
        return False


def find_splits_root(extra):
    """Return the first directory that holds train_mini/ and val/, searching the
    most likely places. `extra` is an optional user-supplied --data path."""
    candidates = []
    if extra:
        candidates.append(os.path.abspath(extra))
    candidates += [
        CANON,                                   # already in the right place
        HERE,                                    # repo root
        os.path.join(HERE, "data"),
        os.path.join(HERE, "data", "raw"),
        os.path.dirname(HERE),                   # one level up (e.g. Desktop)
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.path.expanduser("~"), "Downloads"),
        os.getcwd(),
    ]
    seen = set()
    for c in candidates:
        c = os.path.abspath(c)
        if c in seen or not os.path.isdir(c):
            continue
        seen.add(c)
        if all(is_split_dir(c, s) for s in REQUIRED_SPLITS):
            return c
    return None


def link_dir(src, dst, do_copy):
    """Put the folder `src` at `dst` (junction/symlink, or copy with --copy).
    Returns a short word describing what happened."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return "in place"
    if os.path.isdir(dst) and any(os.scandir(dst)):
        return "already present"
    if os.path.lexists(dst):
        try:
            os.remove(dst)            # stale empty link/file
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)

    if do_copy:
        shutil.copytree(src, dst)
        return "copied"

    # Prefer a no-admin directory junction on Windows; symlink elsewhere.
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return "junction"
        # fall through to copy if junction is refused
        print(f"     (junction failed, copying instead: {r.stderr.strip()})")
        shutil.copytree(src, dst)
        return "copied"
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except Exception:
        shutil.copytree(src, dst)
        return "copied"


def main():
    argv = sys.argv[1:]
    check_only = "--check" in argv
    do_install = "--no-install" not in argv
    do_copy = "--copy" in argv
    extra = None
    if "--data" in argv:
        i = argv.index("--data")
        if i + 1 < len(argv):
            extra = argv[i + 1]

    hr("BEE_HERo - RUN (folders already extracted)")
    print(f"  repo folder : {HERE}")

    # ----- locate the extracted split folders -------------------------------
    src_root = find_splits_root(extra)
    if not src_root:
        hr("PROBLEM: could not find the extracted folders")
        print("  I looked for 'train_mini' and 'val' folders but didn't find them.")
        print("  Put the extracted folders here:")
        print(f"     {CANON}")
        print("  ...so the layout is:")
        print(f"     {os.path.join(CANON, 'train_mini', '...')}")
        print(f"     {os.path.join(CANON, 'val',        '...')}")
        print(f"     {os.path.join(CANON, 'public_test','...')}")
        print("  Or tell me where they are:  python RUN_EXTRACTED.py --data <FOLDER>")
        return 1
    print(f"  found splits: {src_root}")

    os.makedirs(CANON, exist_ok=True)
    placed = {}
    for s in SPLITS:
        if is_split_dir(src_root, s):
            placed[s] = link_dir(os.path.join(src_root, s),
                                 os.path.join(CANON, s), do_copy) if not check_only \
                                 else "would link"
        else:
            placed[s] = "MISSING" if s in REQUIRED_SPLITS else "absent (ok)"
    for s in SPLITS:
        print(f"  {s:14s}: {placed[s]}")
    if any(placed[s] == "MISSING" for s in REQUIRED_SPLITS):
        print("\n  !! train_mini/ and val/ are required. Aborting.")
        return 1

    # ----- environment ------------------------------------------------------
    print(f"\n  python      : {sys.version.split()[0]}  ({sys.executable})")
    for s, label in ((REPRODUCE, "reproduce_bee_hero.py"),
                     (DATASET, "bee_hero_dataset.py")):
        print(f"  script      : {label:24s} {'found' if os.path.isfile(s) else 'MISSING !!'}")
        if not os.path.isfile(s):
            print("  !! pipeline script missing - is this the full repo? Aborting.")
            return 1
    free = shutil.disk_usage(HERE).free / 1e9
    print(f"  free disk   : {free:.0f} GB")

    if check_only:
        hr("--check only: folders located, environment looks good.")
        print("  Run without --check to build.")
        return 0

    # ----- dependencies -----------------------------------------------------
    hr("STEP 0  dependencies (build)")
    missing = [p for m, p in BUILD_DEPS.items() if not have(m)]
    if not missing:
        print("  build deps: all present")
    elif do_install:
        if not pip_install(missing):
            print("  !! install these manually, then re-run:")
            print(f"     {sys.executable} -m pip install {' '.join(missing)}")
            return 1
    else:
        print(f"  !! missing build deps (drop --no-install): {missing}")
        return 1

    # ----- build (extract is skipped: folders are already there) ------------
    hr("STEP 1-5  build dataset (label, filter, dedup, split, configs)")
    cmd = [sys.executable, REPRODUCE, "--root", HERE, "--no-extract"]
    print("  $ " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        print("  !! build step failed - see _pipeline/reproduce.log")
        return 1

    # ----- optional verify --------------------------------------------------
    hr("STEP 6  verify training-readiness (optional - needs torch)")
    if any(not have(m) for m in VERIFY_DEPS):
        if do_install:
            pip_install([p for m, p in VERIFY_DEPS.items() if not have(m)])
    if all(have(m) for m in VERIFY_DEPS):
        print("  $ " + " ".join([sys.executable, DATASET, "--root", HERE]))
        subprocess.run([sys.executable, DATASET, "--root", HERE])
    else:
        print("  torch/torchvision not installed -> skipping verify (build is still done).")

    # ----- status -----------------------------------------------------------
    hr()
    status = "?"
    sp = os.path.join(HERE, "_pipeline", "REPRODUCE_STATUS.txt")
    if os.path.isfile(sp):
        status = open(sp, encoding="utf-8").read().strip()
    print(f"  pipeline status : {status}")
    print("  artifacts       : _pipeline/manifest_*.csv, _pipeline/splits/, "
          "data.yaml, dataset_config.json")
    hr("===  BUILD DONE  ===" if status == "COMPLETED_OK"
       else "===  FINISHED WITH PROBLEMS (see status above)  ===")
    return 0 if status == "COMPLETED_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
