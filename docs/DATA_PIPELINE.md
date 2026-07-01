# Data Preparation Pipeline — What / Why / How

Authoritative record of the iNaturalist data-preparation work on the `data`
branch. Everything is script-driven, reproducible (`seed=42`), and
**non-destructive** (nothing is deleted; "removed" images are *moved* to
`data/_backup/removed/`).

## Components

| File | Role |
|---|---|
| `src/config.py` | Repo-root-relative paths, seed, split ratios, taxonomy targets, bee families. |
| `src/data_pipeline/inaturalist_prep.py` | Insecta filter → exact md5 dedup → per-class 70/15/15 split (manifest). |
| `src/data_pipeline/label_tools.py` | Regenerate labels to surviving images; Phase-5 integrity validation. |
| `src/data_pipeline/eda.py` | Reusable stats / quality / near-dup primitives (shared by both notebooks). |
| `notebooks/00_data_ready.ipynb` | Orchestrates the whole gate; emits a PASS/FAIL report. |
| `notebooks/01_eda.ipynb` | Comprehensive EDA + visualizations. |
| `scripts/setup_env.sh`, `scripts/run_pipeline.sh` | Reproducible env + one-command run. |

Generated artifacts (all git-ignored) land under `data/interim/` (`manifests/`,
`labels/`, `eda/`, `reports/`); safety copies under `data/_backup/`.

## What was found

1. **Unrelated git histories** between `main` and `data` (no merge-base) — plain
   merge refuses. Handled by rebuilding cleanly on `data`; **merge to `main` is
   deferred** until reviewed and accepted.
2. **~170 MB of regenerable artifacts** (`_pipeline/` CSVs, logs, PNGs) committed
   to git, polluting the architecture.
3. **Half-filtered dataset:** `train_mini` and `val` each still held **230
   non-Insecta** species folders (crabs, spiders, isopods) — the earlier manual
   filtering was incomplete.
4. **Labels out of sync with disk:** original COCO JSONs referenced the *full*
   dataset (`train_mini.json` 500k images / 10k categories, `val.json` 100k) while
   disk held far fewer — a massive orphan-label mismatch.
5. **Wrong target ratio:** prior split was 80/10/10; the task requires 70/15/15.
6. **Broken paths:** hardcoded Windows paths (`C:\Users\narim\...`) in configs/docs.
7. **`public_test` is unlabeled** (`annotations: 0`) — cannot serve as a labeled
   test split.

## What was fixed and why

| Fix | Why |
|---|---|
| Untracked `_pipeline/` (kept on disk, git-ignored) | Regenerable outputs don't belong in git; restores clean architecture. |
| Phase-3 backup (labels + tree snapshot + legacy tar + git tag `pre-dataprep`) | Nothing is lost before any mutation; md5-verified. |
| Removed **230 non-Insecta folders/split** (13,800 imgs) → backup, logged | Task requires an Insecta-only classification set. |
| Exact md5 dedup: **15** duplicates removed → backup, logged | Duplicates cause train/test leakage and inflate metrics. |
| Per-class **70/15/15** manifest split (largest-remainder) | Required ratio; largest-remainder keeps all images (no arbitrary truncation). Uniform 60/class → 42/9/9. |
| Manifest-based logical split (images stay in place) | Satisfies "no moving/duplicating between splits"; zero leakage by construction. |
| Regenerated labels: filtered originals + clean per-split COCO | Labels now reference **only** surviving images; passes full integrity checklist. |
| Repo-root-relative paths via `src/config.py` | Portable; fixes the Windows-path breakage. |
| Carved test from labeled pool; left `public_test` untouched | Test must be labeled + relabelable — `public_test` cannot be. |

## Results (verified)

- **2526** Insecta classes · **151,545** images · **70/15/15** =
  106,077 / 22,734 / 22,734 · **3,720** bee images · 11.64 GB.
- Balance: min 59 / max 60 per class, imbalance ratio **1.02**, Gini **≈0**.
- Integrity: **0** missing, **0** duplicate labels, **0** invalid category ids,
  **0** corrupted images (full scan), **0** cross-split leakage, **0** near-dup
  groups (sampled). `00_data_ready` gate → **DATA_READY = True**.

## Remaining issues / recommendations

- **Near-duplicate scan is sampled** (8k) for runtime. For a publishable claim,
  run `eda.find_near_duplicates(paths, sample=None)` over the full set on a
  machine with time to spare; extend to a Hamming-distance threshold if needed.
- **`grayscale` / `blurry` flags are advisory** (sampled, percentile-relative).
  They are reported, not removed — decide per downstream model whether to prune.
- **Merge to `main` is intentionally not done.** After review, create a PR from
  `data`; note the unrelated-history caveat (`--allow-unrelated-histories` or a
  file-level port).
- **Downstream (not in scope):** with a balanced set, standard CrossEntropy
  (optionally light label smoothing) suffices; `is_bee` enables a bee-subset
  evaluation. `public_test` is available for inference-only.
- Legacy teammate scripts (`pipeline.py`, `resplit_option3.py`, `RUN_ME.py`,
  `flower/`) are retained but superseded by the modules above for iNaturalist.
