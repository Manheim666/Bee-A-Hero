"""Central configuration for the Bee-A-Hero data pipeline.

All paths are anchored to the repository root (this file lives in ``src/``),
so the pipeline is portable across machines and does not depend on any
absolute/Windows path. Import this module rather than hard-coding paths.
"""
from __future__ import annotations

from pathlib import Path

# --- Repository layout -------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
BACKUP_DIR: Path = DATA_DIR / "_backup"

INAT_DIR: Path = RAW_DIR / "iNaturist"
FLOWER_DIR: Path = RAW_DIR / "Flower"
# Roboflow bee-detection COCO export (BEE.v8i). Place it here on any machine
# (git-ignored); overridable via the BEE_COCO_DIR env var for other layouts.
BEE_COCO_DIR: Path = Path(__import__("os").environ.get("BEE_COCO_DIR", RAW_DIR / "BEE_coco"))
# Videos to run the visit-counter on.
TEST_VIDEO_DIR: Path = RAW_DIR / "Test_Video"

# Published trained-weights repo on the Hugging Face Hub (for teammates to pull).
HF_WEIGHTS_REPO: str = "Manheim/bee-a-hero-cv"

# iNaturalist splits present on disk. ``public_test`` is unlabeled
# (annotations: 0) and is inference-only — it is NOT part of the labeled split.
INAT_LABELED_SPLITS: tuple[str, ...] = ("train_mini", "val")
INAT_UNLABELED_SPLIT: str = "public_test"

# Generated-artifact locations (git-ignored; see .gitignore data/interim/*).
MANIFEST_DIR: Path = INTERIM_DIR / "manifests"
EDA_DIR: Path = INTERIM_DIR / "eda"
REMOVED_DIR: Path = BACKUP_DIR / "removed"

# --- Reproducibility ---------------------------------------------------------
SEED: int = 42

# --- Split configuration -----------------------------------------------------
# Train/val/test proportions carved from the pooled labeled Insecta images.
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# --- Taxonomy targets --------------------------------------------------------
# Keep only folders whose taxonomic Class == Insecta.
TARGET_CLASS: str = "Insecta"

# Bee families (Hymenoptera) tagged as a subset of interest for the project.
BEE_FAMILIES: frozenset[str] = frozenset({
    "Andrenidae", "Apidae", "Colletidae", "Halictidae",
    "Megachilidae", "Melittidae", "Stenotritidae",
})

# Valid image extensions.
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
