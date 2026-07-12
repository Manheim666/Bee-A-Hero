# Legacy scripts (archived)

Pre-70/15/15 artifacts, kept for reference only. **Superseded** by the current
data-preparation pipeline (`src/data_pipeline/{inaturalist_prep,label_tools,eda}.py`,
`notebooks/00_data_ready.ipynb`, `scripts/run_pipeline.sh`).

| File | Was | Note |
|---|---|---|
| `RUN_ME.py` | one-click runner for the old pipeline | references `_pipeline/` (now git-ignored) and Windows paths |
| `RUN_EXTRACTED.py` | runner for already-extracted folders | same |
| `data.yaml` | 80/10/10 split config | stale split + absolute path; current split is `data/interim/manifests/split_manifest.csv` |
| `dataset_config.json` | 80/10/10 split metadata | superseded by `data/interim/manifests/split_summary.json` |

Moved here so the repository root matches the canonical project structure
(`data/ src/ notebooks/ scripts/ tests/ docs/` + standard root files).
