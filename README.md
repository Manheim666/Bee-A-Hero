# Bee-A-Hero — ML Stage

**Pollination → fruit-set → yield.** This branch holds the machine-learning / modeling stage:
it turns the computer-vision pipeline's per-flower **pollinator-visit** records into a
**fruit-set** and **yield** estimate, with uncertainty.

The full modeling stack lives in `src/ml_models/` and is demonstrated end-to-end in one notebook,
`notebooks/03_ml.ipynb`.

---

## The model

Fruit set is modeled as a **saturating dose–response** on the effective pollinator dose `V`:

```
FruitSet(V) = F0 + (Fmax − F0) · (1 − e^(−k·V))
```

- **F0** — fruit set with no insect visits (partial self-fertility)
- **Fmax** — maximum attainable fruit set (ceiling)
- **k** — how fast it saturates (diminishing returns)

The pipeline (all in `src/ml_models/`):

| Module | Role |
|---|---|
| `visit_dataset.py` | load visits, apply the **three qualifying gates** (dwell / velocity / fraction-on), build the per-flower effective dose `V` |
| `dose_response.py` | fit `FruitSet(V)` with bootstrap 95% CIs and AIC/BIC |
| `glmm.py` | binomial **GLMM** with orchard/year random intercepts (statsmodels) |
| `bayesian.py` | Bayesian curve with a **cross-crop prior** + prior-sensitivity check |
| `uncertainty.py` | fruit-set prediction intervals + Monte-Carlo **yield** propagation |
| `train.py` | CLI orchestrator (`python -m src.ml_models.train`) |

Design rationale, formulas, and pomegranate biology are in `docs/ML_MODELING_RESEARCH.md`.

---

## Data: `dataset_training_v11.csv` (CV-schema aligned)

No real per-flower **pomegranate** fruit-set labels exist yet, so the notebook trains on a
**synthetic** frame that **mirrors the CV pipeline's output** (`video_detect.py` →
`ALL_flower_summary.csv`): per-flower landing counts, the CV insect categories
(`n_honeybee, n_bee, n_fly, n_beetle, n_bug, n_butterfly`), `pollination_score`, weather, and a
synthetic `fruit_set_label` drawn from a known dose-response curve.

Because the columns and species **match the CV output 1:1**, a real CV run feeds straight into
the pipeline — no schema translation. The effective dose `V` is the CV `pollination_score`.

> **Data files are git-ignored** (large / regenerable). The notebook expects
> `data/processed/dataset_training_v11.csv` and `data/processed/croppol_field.csv`
> plus `test_video_result/ALL_landings.csv`. A ready-to-run bundle with the data is shared
> separately (`bee-ml-share.zip`).

---

## Results

Fit on v11, the pipeline **recovers the known asymptotes** almost exactly:

| crop | F0 (fit / true) | Fmax (fit / true, 95% CI) |
|---|---|---|
| cucumber | 0.055 / 0.05 | 0.787 / 0.78 ([0.77, 0.80]) |
| pomegranate | 0.447 / 0.45 | 0.687 / 0.68 ([0.67, 0.70]) |

- **GLMM:** significant positive dose effect (more visits → more fruit set) after orchard/year random effects.
- **Bayesian:** ceiling recovered; small prior-sensitivity shift (the data, not the prior, set Fmax).
- **Model test (notebook §7):** predicts sensible, saturating fruit set + yield for any input dose.
- **Real-data test (notebook §8):** the saturating curve explains **R² = 0.74** of real CropPol visitation→yield.

> **CV alignment:** v11's species and columns are the **exact CV taxonomy** the tracker emits, so
> the ML stage integrates with the CV output directly — no `nq_*`/`w_*` translation needed.

---

## Run it

```bash
# one-shot pipeline (prints params + yield, writes the git-ignored models/yield_report.json;
# the committed models/dose_response_v11.json curve is read-only)
python -m src.ml_models.train --dataset data/processed/dataset_training_v11.csv

# apply-only: reuse the committed v11 curve on the CV landings (no re-fit, no statsmodels;
# also auto-selected when statsmodels is not installed)
python -m src.ml_models.train --apply-only

# or open the notebook and Run All (portable Python 3 kernel, no path edits needed)
notebooks/03_ml.ipynb
```

Deps (fit): `numpy pandas scipy scikit-learn statsmodels matplotlib seaborn jupyter`.
Apply-only needs only `numpy pandas scipy`.

---

## Status & honest limitations

- **ML pipeline: complete and validated** — recovers the known curve, passes behavioral and
  real-data (CropPol) tests, runs end-to-end, portable across machines.
- **Not yet a real pomegranate model** — it is validated on synthetic and real-calibrated
  (proxy-crop) data. A true pomegranate fit needs **local per-flower fruit-set labels** and
  cross-time flower identity, which are field-data-collection tasks, not modeling gaps. The code
  is already wired to accept them (`visit_dataset.join_fruit_set_labels`, and the
  `anchor_f0` / `fmax_anchor` controls in `dose_response.fit_dose_response`).
