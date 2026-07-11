# Bee-A-Hero — ML Stage

**Pollination → fruit-set → yield.** This branch holds the machine-learning / modeling stage:
it turns the computer-vision pipeline's per-flower **pollinator-visit** records into a
**fruit-set** and **yield** estimate, with uncertainty.

The full modeling stack lives in `src/ml_models/` and is demonstrated end-to-end in one notebook,
`notebooks/03_ml_dose_response.ipynb`.

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
| `bee_hero_dataset.py` | load visits, apply the **three qualifying gates** (dwell / velocity / fraction-on), build the per-flower effective dose `V` |
| `dose_response.py` | fit `FruitSet(V)` with bootstrap 95% CIs and AIC/BIC |
| `glmm.py` | binomial **GLMM** with orchard/year random intercepts (statsmodels) |
| `bayesian.py` | Bayesian curve with a **cross-crop prior** + prior-sensitivity check |
| `uncertainty.py` | fruit-set prediction intervals + Monte-Carlo **yield** propagation |
| `train.py` | CLI orchestrator (`python -m src.ml_models.train`) |

Design rationale, formulas, and pomegranate biology are in `docs/ML_MODELING_RESEARCH.md`.

---

## Data: the "synthetic-from-real" approach

No real per-flower **pomegranate** fruit-set labels exist yet, so the notebook trains on a
**real-calibrated** dataset: synthetic feature rows, but fruit-set labels drawn from
dose–response curves **fitted to real CropPol field data** (Allen-Perkins et al. 2022):

- cucumber ← *Cucurbita pepo* (squash), real fit **R² = 0.74**
- pomegranate ← *Brassica napus* (rapeseed), real fit **R² = 0.39**

So the outcomes are grounded in real crop-pollination biology, not invented.

> **Data files are git-ignored** (large / regenerable). The notebook expects
> `data/processed/dataset_training_realcalibrated.csv` and `data/processed/croppol_field.csv`
> plus `test_video_result/ALL_landings.csv`. A ready-to-run bundle with the data is shared
> separately (`bee-ml-share.zip`).

---

## Results

Fit on the real-calibrated data, the pipeline **recovers the real-derived asymptotes**:

| crop | F0 (fit / true) | Fmax (fit / true, 95% CI) |
|---|---|---|
| cucumber | 0.01 / 0.00 | 0.69 / 0.71 ([0.69, 0.70]) |
| pomegranate | 0.15 / 0.14 | 0.58 / 0.57 ([0.57, 0.58]) |

- **GLMM:** significant positive dose effect (more visits → more fruit set) after orchard/year random effects.
- **Bayesian:** ceiling recovered; prior-sensitivity shift ≈ 0.001 (the data, not the prior, set Fmax).
- **Model test (notebook §7):** predicts sensible, saturating fruit set + yield for any input dose.
- **Real-data test (notebook §8):** the saturating curve explains **R² = 0.74** of real CropPol visitation→yield.
- **Model optimization (notebook §9):** adding **weather covariates** + cross-validated
  **hyperparameter tuning** raises predictive 5-fold CV ROC-AUC from **0.615 → 0.733 (+0.118)**;
  the weather covariates provide the bulk of the lift (temperature/wind/humidity affect pollen
  viability, research doc §5/§13).

---

## Run it

```bash
# one-shot pipeline (prints params + yield, writes models/dose_response_realcalibrated.json)
python -m src.ml_models.train --dataset data/processed/dataset_training_realcalibrated.csv

# or open the notebook and Run All (portable Python 3 kernel, no path edits needed)
notebooks/03_ml_dose_response.ipynb
```

Deps: `numpy pandas scipy scikit-learn statsmodels matplotlib seaborn jupyter`.

---

## Status & honest limitations

- **ML pipeline: complete and validated** — recovers the known curve, passes behavioral and
  real-data (CropPol) tests, runs end-to-end, portable across machines.
- **Not yet a real pomegranate model** — it is validated on synthetic and real-calibrated
  (proxy-crop) data. A true pomegranate fit needs **local per-flower fruit-set labels** and
  cross-time flower identity, which are field-data-collection tasks, not modeling gaps. The code
  is already wired to accept them (`bee_hero_dataset.join_fruit_set_labels`, and the
  `anchor_f0` / `fmax_anchor` controls in `dose_response.fit_dose_response`).
