# Pipeline Walkthrough — definitions & explanation of every step

Plain-language reference for the iNaturalist data-preparation pipeline. Explains
**what** each notebook step and each Python function does, and **why**. Read
alongside `docs/DATA_PIPELINE.md` (the what/why/results summary).

## Glossary (key terms used below)

| Term | Definition |
|---|---|
| **Manifest** | A CSV listing every kept image with its label and split assignment. The single source of truth — the split is *logical* (a table), so images are never physically shuffled between folders. |
| **COCO-classification JSON** | The iNaturalist label format: `images` (id, file, size), `categories` (taxonomy), `annotations` (image_id → category_id). Image-level labels, **no bounding boxes**. |
| **Insecta filter** | Keep only species whose taxonomic *Class* is `Insecta` (drops crabs, spiders, isopods, etc.). |
| **Exact duplicate** | Two files with identical bytes (same md5 hash). One is redundant. |
| **Near-duplicate** | Visually near-identical images (resize/re-encode) — detected by a *perceptual hash* (pHash), not byte equality. |
| **Stratified split** | Split each class independently so every class keeps the same train/val/test proportions. |
| **Largest-remainder** | A rounding rule that allocates whole images to train/val/test so the counts sum exactly to the class total with no image dropped. |
| **Cross-split leakage** | The same (or near-same) image landing in two splits — inflates test scores. Prevented here because each image path is assigned to exactly one split. |
| **Blur (variance of Laplacian)** | A sharpness score: the Laplacian highlights edges; low variance = few edges = blurry. |
| **Idempotent** | Running a step twice gives the same result and does no extra harm. |

---

## `src/config.py` — central configuration
Defines everything path- and policy-related in one place so nothing is
hard-coded. `REPO_ROOT` is derived from the file location (`parents[1]`), which
makes every path **repo-relative and portable** (fixes the old Windows-path
breakage). Key values: `SEED=42` (reproducibility), `SPLIT_RATIOS` (70/15/15),
`TARGET_CLASS="Insecta"`, `BEE_FAMILIES` (7 bee families), and the directory
constants (`INAT_DIR`, `MANIFEST_DIR`, `BACKUP_DIR`, `REMOVED_DIR`, …).

## `src/data_pipeline/inaturalist_prep.py` — filter → dedup → split

| Function | What it does | Why |
|---|---|---|
| `parse_taxon(folder_name)` | Splits `ID_Kingdom_..._species` into taxonomy fields. | The folder name *is* the label; this decodes it. |
| `is_insecta` / `is_bee` | Class == Insecta; Order == Hymenoptera & family ∈ bee families. | Defines what to keep and what to tag as a bee. |
| `iter_class_dirs` / `_images_in` | Deterministic (sorted) iteration of folders/images. | Determinism → reproducible runs. |
| `move_to_backup(path, reason)` | Moves a file under `data/_backup/removed/` mirroring its path; returns a log row. | The **only** mutating primitive — nothing is deleted, everything reversible + logged. |
| `filter_non_insecta` | Moves every non-Insecta folder's images to backup. | Produces an Insecta-only set (drops 230 folders/split). |
| `build_pool` | One record per surviving Insecta image (pooled from train_mini + val). | The labeled pool to split. |
| `_md5` + `dedup_exact` | Hashes each image (cached by size+mtime); moves exact duplicates to backup, keeping the first. | Removes redundant/leak-causing copies (15 found). |
| `_largest_remainder(n, ratios)` | Allocates `n` images to train/val/test summing exactly to `n`. | Hits 70/15/15 per class **without dropping images**. |
| `assign_splits` | Per class: sort → seeded shuffle → allocate → tag `split` + contiguous `class_id`. | Stratified, reproducible, leakage-free assignment. |
| `write_outputs` | Writes `split_manifest.csv`, `removed_log.csv`, `class_index.json`, `split_summary.json`. | Persists the labels + full audit trail. |
| `run(apply, skip_dedup)` | Orchestrates all of the above. `apply=False` = dry-run (plan only). | One entry point; safe preview before mutating. |

## `src/data_pipeline/label_tools.py` — relabel & validate

| Function | What it does | Why |
|---|---|---|
| `filter_original(split_json, survivors)` | Drops images/annotations/categories from the original COCO JSON that no longer exist. | Original JSONs referenced the full pre-filter dataset — this realigns them. |
| `build_clean_labels(manifest, …)` | Writes clean `train/val/test.json` (contiguous category ids, dims from originals). | Model-ready labels referencing only surviving images. |
| `validate(manifest)` | Runs the Phase-5 checklist: missing files, labels-without-images, duplicate labels, invalid category ids, corrupted labels, cross-split leakage. | Proves "labels perfectly match the dataset, no orphans." |
| `run(validate_only)` | Orchestrates filtering + clean-label build + validation; writes `validation_report.json`. | One entry point; `--validate-only` re-checks without rewriting. |

## `src/data_pipeline/eda.py` — reusable stats/quality primitives
Imported by **both** notebooks (no duplicated code). Uses a `fork`
multiprocessing context (Python 3.14 defaults to `forkserver`, which breaks
under notebooks).

- `load_manifest_df` / `manifest_paths` — read the manifest.
- `scan_corrupted(paths)` — opens+verifies every image in parallel; returns unreadable ones.
- `_laplacian_var` / `_meta_one` / `sample_image_stats` — per-image width/height/aspect/mode/brightness/contrast/**blur**/grayscale/blank.
- `find_near_duplicates(paths)` — groups images sharing a perceptual hash.
- `class_distribution` / `order_distribution` / `split_class_matrix` — aggregations for plots.
- `imbalance_metrics(counts)` — imbalance ratio + Gini coefficient.
- `sample_grid_paths` — deterministic image sample for preview grids.

---

## `notebooks/00_data_ready.ipynb` — the readiness gate (step by step)
1. **Environment** — locate repo root, make `src` importable.
2. **Inspection** — count folders/images per split; confirm non-Insecta removed; note `public_test` is unlabeled.
3. **Backup verification** — confirm `data/_backup/` exists and original labels match their md5 **before** any mutation.
4. **Balancing** — run `inaturalist_prep.run(apply=True)` (idempotent) → 70/15/15.
5. **Label regeneration** — run `label_tools.run()` → clean labels + validation.
6. **Integrity** — missing files, duplicate labels, corrupted images (full scan), near-duplicates, directory consistency.
7. **Final report** — aggregate every check into a PASS/FAIL gate, write `data_ready_report.json`, and **`assert DATA_READY`** so the notebook fails loudly if anything is wrong.

## `notebooks/01_eda.ipynb` — comprehensive EDA (step by step)
1. **Setup** — imports, headless matplotlib, load manifest.
2. **Overview** — image/class/order/family counts, split distribution, disk size.
3. **Class & order distribution** — per-class histogram, order bar chart, imbalance metrics.
4. **Split, bee subset, order heatmap** — split pie, bee vs non-bee bars, order×split heatmap, per-class coverage.
5. **Image statistics (sampled)** — resolution histograms + scatter, aspect boxplot, brightness & blur histograms.
6. **Quality analysis** — grayscale, blank, blurry, exposure outliers, near-duplicates; colour-mode & contrast plots.
7. **Sample images** — deterministic random grid across splits.
8. **Summary** — write `eda_summary.json`; figures saved to `data/interim/eda/`.

---

## Where outputs go (all git-ignored)
```
data/interim/manifests/   split_manifest.csv, removed_log.csv, class_index.json,
                          split_summary.json, validation_report.json, md5_cache.csv
data/interim/labels/      train/val/test.json, inat_{train_mini,val}_filtered.json
data/interim/eda/         *.png, eda_summary.json
data/interim/reports/     data_ready_report.json
data/_backup/             original_labels/, removed/, file_manifests/, legacy_pipeline.tar.gz
```
