# BEE_HERo — Continue Prompt (resume tomorrow)

**To resume:** open Claude Code in `C:\Users\narim\Desktop\BEE_HERo` and say:
> "Read continue.md and continue the BEE_HERo pipeline."

---

## What this project is
End-to-end dataset cleaning + EDA + readiness pipeline for an **insect/bee** dataset.
The data in `C:\Users\narim\Desktop\BEE_HERo` is the **iNaturalist** dataset (3 archives:
`train_mini.tar.gz`, `val.tar.gz`, `public_test.tar.gz`).

## Key reality (already confirmed)
- It is an **image-classification** dataset, **NOT detection** — folders are per-species,
  named `ID_Kingdom_Phylum_Class_Order_Family_Genus_species`. **No bounding boxes exist.**
- So the pipeline runs as a **classification** adaptation; bbox-only steps are marked N/A.

## Decisions locked with the user (do NOT re-ask)
1. **Adapt to classification** pipeline (faithful to the 6-phase task).
2. **Keep ALL `Insecta`** species folders; additionally **tag bee families** as a subset
   (Apidae, Andrenidae, Halictidae, Megachilidae, Colletidae, Melittidae, Stenotritidae).
3. **All 3 splits processed.** `public_test` is **flat/unlabeled** (just numbered .jpgs) →
   left intact, profiled only (cannot be class-filtered, no labels).
4. **Never touch the `.tar.gz` archives** (keep everything reversible).
5. **20 GB disk floor** — abort (no shutdown) if free space would drop below it.
6. Overnight shutdown was wanted, BUT see lesson below. If running while user is awake,
   **do not auto-shutdown** unless they ask.

## LESSON LEARNED (important)
The overnight run **failed**: detached `nohup ... &` background jobs were **killed when the
chat session ended**. Extraction finished but the analysis pipeline never ran and the PC did
not shut down. **Do NOT detach. Run the pipeline in the active session and monitor it.**

## Current state (frozen at stop)
- Extraction COMPLETE: `train_mini` (partially filtered: 10000 → ~9825 folders, 2526 Insecta
  kept, ~7299 non-insect still to delete), `val` (10000, untouched), `public_test` (500000, intact).
- Archives intact. ~257 GB free.
- Pipeline script exists and is tested/compiles: `_pipeline/pipeline.py` (+ `_pipeline/run_all.sh`
  orchestrator — IGNORE run_all.sh for the manual resume; it was only for the detached overnight run).
- Partial folder deletion in train_mini is harmless — the pipeline re-walks everything and is
  effectively idempotent (re-deletes remaining non-insect, keeps insect, rebuilds manifests).

## How to resume (exact steps)
1. Confirm state: `df -h .`, count folders in train_mini/val/public_test, confirm archives present.
2. Run the pipeline **monitored, NOT detached**:
   ```bash
   cd "/c/Users/narim/Desktop/BEE_HERo"
   rm -f _pipeline/STATUS.txt _pipeline/pipeline.log
   python _pipeline/pipeline.py > _pipeline/pipeline_console.log 2>&1 &
   ```
   then poll `_pipeline/pipeline.log` and `_pipeline/STATUS.txt` until STATUS = `COMPLETED_OK`.
   (Run time ≈ 40–70 min: folder deletion + image verify + perceptual-hash leakage scan.)
3. The pipeline does Phases 1–6: filter to Insecta (+tag bees), integrity-verify images,
   build `manifest_*.csv`, write `data.yaml` + `dataset_config.json`, EDA plots in `_pipeline/eda/`,
   quality eval (`phase6_quality.json`), and `_pipeline/REPORT.md` (with a Morning Summary at top).
4. When STATUS = `COMPLETED_OK`, summarize `REPORT.md` to the user. **Ask before shutting down.**

## Expected outputs (in `_pipeline/`)
`REPORT.md`, `manifest_all.csv`, `manifest_train_mini.csv`, `manifest_val.csv`,
`eda/` (dist_by_order.png, class_size_hist.png, resolution_scatter.png, aspect_ratio_hist.png,
sample_grid.png, *.csv, eda_summary.json), `phase4_split_check.json`, `phase6_quality.json`,
plus `data.yaml` and `dataset_config.json` at the repo root.

## Pending CLAUDE.md task (do at the very end)
Append a Daily Activities entry to `C:\Users\narim\Desktop\README.md` (ARIAN project rule) —
note: this BEE_HERo work is separate from ARIAN; confirm with user whether to log it there.
