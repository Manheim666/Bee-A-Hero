"""Reusable EDA & image-quality primitives for the Bee-A-Hero dataset.

These functions are imported by BOTH ``notebooks/00_data_ready.ipynb``
(integrity / quality gate) and ``notebooks/01_eda.ipynb`` (visual analysis), so
the heavy logic lives here once and the notebooks stay presentation-only.

Everything reads the Phase-4 manifest (``data/interim/manifests/split_manifest.csv``)
as the single source of truth for which images exist and their labels.
"""
from __future__ import annotations

import multiprocessing
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src import config as C

# Large iNaturalist frames are legitimate; disable the decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

# Python 3.14 defaults to the "forkserver" start method, which re-imports the
# entry module in each worker and breaks under stdin/notebook execution. "fork"
# is safe here (workers only do PIL + numpy) and needs no re-import.
_MP = multiprocessing.get_context("fork")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def load_manifest_df():
    """Return the Phase-4 split manifest as a DataFrame."""
    import pandas as pd
    return pd.read_csv(C.MANIFEST_DIR / "split_manifest.csv")


def manifest_paths(df=None) -> list[str]:
    df = load_manifest_df() if df is None else df
    return df["path"].tolist()


# --------------------------------------------------------------------------- #
# Integrity — corrupted / unreadable images
# --------------------------------------------------------------------------- #
def _check_one(path_str: str):
    try:
        with Image.open(path_str) as im:
            im.verify()               # verify header + payload without full decode
        return None
    except Exception as e:            # unreadable / truncated / corrupt
        return (path_str, repr(e))


def scan_corrupted(paths, workers: int | None = None, chunk: int = 64) -> list[tuple[str, str]]:
    """Return ``[(path, error)]`` for every image that fails to open/verify."""
    paths = [str(p) for p in paths]
    bad: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP) as ex:
        for res in ex.map(_check_one, paths, chunksize=chunk):
            if res is not None:
                bad.append(res)
    return bad


# --------------------------------------------------------------------------- #
# Per-image statistics (sampled — full set is 151k images)
# --------------------------------------------------------------------------- #
def _meta_one(path_str: str):
    try:
        with Image.open(path_str) as im:
            w, h = im.size
            mode = im.mode
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            is_gray = mode in ("L", "1")
            if mode == "RGB":
                rgb = np.asarray(im, dtype=np.int16)
                if rgb.ndim == 3 and rgb.shape[2] >= 3:
                    dg = np.abs(rgb[..., 0] - rgb[..., 1]).mean()
                    db = np.abs(rgb[..., 1] - rgb[..., 2]).mean()
                    is_gray = bool(dg < 2 and db < 2)
            blank = bool(contrast < 1.0)
        return (path_str, w, h, mode, brightness, contrast, int(is_gray), int(blank))
    except Exception:
        return (path_str, None, None, None, None, None, None, None)


def sample_image_stats(paths, sample: int | None = 6000, seed: int = C.SEED,
                       workers: int | None = None):
    """Compute width/height/aspect/mode/brightness/contrast/grayscale/blank.

    Sampled by default for speed; pass ``sample=None`` to scan everything.
    """
    import pandas as pd
    paths = [str(p) for p in paths]
    rng = random.Random(seed)
    if sample and len(paths) > sample:
        paths = rng.sample(paths, sample)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP) as ex:
        for r in ex.map(_meta_one, paths, chunksize=32):
            rows.append(r)
    df = pd.DataFrame(rows, columns=["path", "width", "height", "mode",
                                     "brightness", "contrast", "is_gray", "blank"])
    df["aspect"] = df["width"] / df["height"]
    return df


# --------------------------------------------------------------------------- #
# Near-duplicate detection (perceptual hash)
# --------------------------------------------------------------------------- #
def find_near_duplicates(paths, sample: int | None = 8000, seed: int = C.SEED):
    """Group images sharing an identical perceptual hash (resize/re-encode dups).

    Returns a list of path-groups (each length >= 2). Sampled for tractability.
    """
    import imagehash
    rng = random.Random(seed)
    paths = [str(p) for p in paths]
    if sample and len(paths) > sample:
        paths = rng.sample(paths, sample)
    buckets: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        try:
            with Image.open(p) as im:
                buckets[str(imagehash.phash(im))].append(p)
        except Exception:
            continue
    return [ps for ps in buckets.values() if len(ps) > 1]


# --------------------------------------------------------------------------- #
# Aggregations for plots
# --------------------------------------------------------------------------- #
def class_distribution(df=None):
    """Per-class image counts (DataFrame indexed by class_name)."""
    df = load_manifest_df() if df is None else df
    return df.groupby("class_name").size().sort_values(ascending=False)


def split_class_matrix(df=None):
    """Rows = class, columns = split, values = image counts."""
    df = load_manifest_df() if df is None else df
    return df.pivot_table(index="class_name", columns="split",
                          values="path", aggfunc="count", fill_value=0)


def order_distribution(df=None):
    """Image counts per taxonomic order."""
    df = load_manifest_df() if df is None else df
    return df.groupby("order").size().sort_values(ascending=False)


def imbalance_metrics(counts) -> dict:
    """Imbalance ratio + Gini over a series/array of per-class counts."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    ratio = float(x.max() / x.min()) if x.min() > 0 else float("inf")
    if x.sum() == 0:
        gini = 0.0
    else:
        cum = np.cumsum(x)
        gini = float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
    return {"num_classes": n, "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "imbalance_ratio": ratio, "gini": round(gini, 4)}


def sample_grid_paths(df=None, per_split: int = 8, seed: int = C.SEED) -> list[str]:
    """Deterministic sample of image paths for a preview grid."""
    df = load_manifest_df() if df is None else df
    rng = random.Random(seed)
    out: list[str] = []
    for split in ("train", "val", "test"):
        pool = df[df["split"] == split]["path"].tolist()
        out += rng.sample(pool, min(per_split, len(pool)))
    return out
