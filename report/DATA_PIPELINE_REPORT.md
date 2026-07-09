# Data Pipeline — Audit Report

_Scope: `src/data_pipeline/`, `notebooks/00_data_ready.ipynb`, `notebooks/01_eda.ipynb`._

## Verdict

**Healthy and production-ready.** Full integrity check passes with zero issues.

```
num_classes            2526 (Insecta species)
images                 151,545
split                  70 / 15 / 15  (106,077 / 22,734 / 22,734)
images_without_labels  0
labels_without_images  0
duplicated_labels      0
invalid_category_ids   0
corrupted_labels       0
cross_split_leakage    0
all_checks_pass        True
```

## Architecture (from the knowledge graph)

Active pipeline (3 focused modules):

| Module | Role |
|---|---|
| `inaturalist_prep.py` | Insecta filter → md5 dedup → per-class 70/15/15 split → public_test downsample. Non-destructive (moves to backup). |
| `label_tools.py` | Regenerate COCO labels to surviving images + full integrity validation. |
| `eda.py` | Reusable EDA / quality primitives (shared by both notebooks). |

Config centralised in `src/config.py` (repo-root-relative paths, `SEED=42`).

## Strengths

- **Reproducible** — single seed, deterministic ordering; same input → same output on any machine.
- **Non-destructive** — removed images moved to backup, never deleted; every removal logged.
- **Portable** — no absolute/OS-specific paths; missing-data guard prints acquisition instructions.
- **Validated** — the 6-point integrity checklist gates the whole stage (`00_data_ready` asserts `DATA_READY`).
- **Species labels present** — the manifest already carries per-species `class_id` (0–2525) and taxonomy fields, so downstream species classification needs **no data change**.

## Insufficiencies found + status

| # | Issue | Severity | Action |
|---|---|---|---|
| 1 | Superseded teammate scripts (`pipeline.py`, `reproduce_bee_hero.py`, `resplit_option3.py`) still live in `src/data_pipeline/` — duplicate/confusing. | low | **Recommend** relocating to `scripts/legacy/` (kept for now to avoid touching teammate code). |
| 2 | No automated test asserting the integrity invariants (only the notebook gate). | low | **Recommend** a small `tests/test_data.py` calling `label_tools.validate`. |
| 3 | `is_bee` flag is coarse (pollinator vs not). CV will move to full species — flag retained only as a convenience column. | info | Handled in CV stage. |

## Convenience / quality upgrades available

- One-command run already exists: `bash scripts/run_pipeline.sh` (idempotent).
- Result snapshot for reviewers already published under `docs/results/`.

## Recommendation

Data stage is **complete and does not block CV**. The two low-severity items are
housekeeping, not correctness. Proceed to the CV improvements (species-level
classification, tighter insect boxes, higher mAP@0.5:0.95, lower-fps tracking).
