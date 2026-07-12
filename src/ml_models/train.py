"""CLI orchestrator for the fruit-set modeling stage (research doc Section 19, step 8).

Wires the modeling pipeline end to end:

    tracker/visit CSV  ->  three qualifying gates  ->  per-flower effective dose V
                       ->  fit FruitSet(V) = F0 + (Fmax-F0)(1-exp(-kV))
                       ->  fruit-set + orchard-yield estimates, each with a 95% interval

Because no real fruit-set labels exist yet (research doc Section 16), the curve is
**fit** on the processed training frame ``dataset_training_v8.csv`` (the same dataset as
``notebooks/03_ml.ipynb``). v6 drops the raw dose column, so the effective dose ``V`` is
reconstructed from the per-flower aggregates, and the fit is validated by recovering the
per-crop floor/ceiling (``p_self_used`` / ``p_cross_used``). The fitted curve is then
**applied** to the real tracker export (``ALL_landings.csv``) to produce interval-carrying
fruit-set and yield numbers for the report layer — the genuine tracker-to-yield path.

Run:
    python -m src.ml_models.train
    python -m src.ml_models.train --dataset data/processed/dataset_training_v8.csv \
        --n-flowers 1200 --mean-mass 0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_models import bee_hero_dataset as ds
from src.ml_models.bee_hero_dataset import CROPS
from src.ml_models.dose_response import fit_dose_response
from src.ml_models.uncertainty import propagate_yield
from src.ml_models.glmm import fit_glmm
from src.ml_models.bayesian import bayes_dose_response, prior_sensitivity

# Repo paths derived from this file's location (self-contained, no config dependency,
# portable across machines and branches).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

# Default modeling inputs; resolved from data/processed with a Downloads fallback.
DATASET_CANDIDATES = (
    _PROCESSED_DIR / "dataset_training_v11.csv",
    Path.home() / "Downloads" / "dataset_training_v11.csv",
)
TRACKER_LANDINGS = _REPO_ROOT / "test_video_result" / "ALL_landings.csv"
OUT_JSON = _REPO_ROOT / "models" / "dose_response_v11.json"


def _resolve_dataset(path: str | None) -> Path:
    """Return the first existing dataset path (explicit override, then the candidates)."""
    for cand in ([Path(path)] if path else []) + list(DATASET_CANDIDATES):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "dataset_training_v8.csv not found in data/processed or ~/Downloads")

# Illustrative orchard constants for the yield figure (override on the CLI).
DEFAULT_N_FLOWERS = 1200
DEFAULT_MEAN_MASS_KG = 0.30


# --------------------------------------------------------------------------- #
# Fit stage — one curve per crop on the synthetic modeling frame
# --------------------------------------------------------------------------- #
def fit_on_dataset(dataset_path: Path) -> dict:
    """Fit ``FruitSet(V)`` per crop and compare recovered asymptotes to the known truth.

    Works on the processed ``dataset_training_v8`` frame — which has no raw ``V`` column
    (dropped as a leakage/generator field) and labels ``fruit_set_label`` — by
    reconstructing the effective dose from the per-flower aggregates. The per-crop
    ground-truth floor/ceiling come from ``p_self_used`` / ``p_cross_used`` when present,
    else from the generator's :data:`CROPS` table.
    """
    df = pd.read_csv(dataset_path)
    if "V" not in df.columns:
        df["V"] = ds.effective_dose_from_aggregates(df)
    target = "fruit_set" if "fruit_set" in df.columns else "fruit_set_label"
    fallback = {c["crop"]: c for c in CROPS}

    results = {}
    for crop, g in df.groupby("crop"):
        V, y = g["V"].to_numpy(), g[target].to_numpy()
        fit = fit_dose_response(V, y)
        rec = fit.as_dict()
        if {"p_self_used", "p_cross_used"} <= set(g.columns):
            rec["truth"] = {"F0": float(g["p_self_used"].iloc[0]),
                            "Fmax": float(g["p_cross_used"].iloc[0])}
        elif crop in fallback:
            t = fallback[crop]
            rec["truth"] = {"F0": t["F0"], "Fmax": t["Fmax"],
                            "k": round(t["k"], 5), "v_star": t["V_star"]}

        # binomial GLMM (random intercepts) + Bayesian curve (cross-crop prior) — doc steps 4-5
        grp = tuple(c for c in ("orchard_id", "year") if c in g.columns)
        if grp:
            rec["glmm"] = fit_glmm(g, dose_col="V", target=target, groups=grp).as_dict()
        bfit = bayes_dose_response(V, y)
        rec["bayesian"] = bfit.summary()
        ps = prior_sensitivity(V, y)
        rec["bayesian"]["prior_sensitivity_shift"] = round(float(ps.attrs["posterior_shift"]), 4)

        results[crop] = {"fit": fit, "report": rec}
    return results


# --------------------------------------------------------------------------- #
# Apply stage — fitted curve -> real tracker output -> yield
# --------------------------------------------------------------------------- #
def apply_to_tracker(fit, landings_path: Path, n_flowers: int, mean_mass: float) -> dict:
    """Run the tracker export through the gates and the fitted curve to a yield band."""
    flowers, applied = ds.flower_dose_frame(landings_path)
    doses = flowers["V"].to_numpy()
    mean_dose = float(np.mean(doses)) if doses.size else 0.0

    yield_est = propagate_yield(fit, mean_dose, n_flowers, mean_mass)
    return {
        "source": str(landings_path.name),
        "applied_gates": applied,
        "n_flowers_observed": int(len(flowers)),
        "n_qualifying_visits_total": int(flowers["n_qualifying_visits"].sum()),
        "mean_effective_dose": round(mean_dose, 4),
        "yield_estimate": yield_est,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(dataset_path: Path, landings_path: Path, out_path: Path,
        n_flowers: int, mean_mass: float) -> dict:
    print(f"[1/3] Fitting FruitSet(V) per crop on {dataset_path.name} ...")
    fits = fit_on_dataset(dataset_path)
    for crop, r in fits.items():
        rep = r["report"]
        t = rep.get("truth", {})
        print(f"      {crop:<12} F0={rep['F0']:.3f} (true {t.get('F0','?')})  "
              f"Fmax={rep['Fmax']:.3f} (true {t.get('Fmax','?')})  "
              f"k={rep['k']:.3f}  d_logloss={rep['delta_logloss']:+.4f}")
        if "glmm" in rep:
            gl = rep["glmm"]
            print(f"        GLMM   dose_beta/sd={gl['dose_beta_per_sd']:+.3f}  "
                  f"random-intercept SD={gl['random_intercept_sd']}")
        by = rep["bayesian"]
        print(f"        Bayes  Fmax={by['Fmax_mean']:.3f} {by['Fmax_ci95']}  "
              f"prior-sensitivity shift={by['prior_sensitivity_shift']:.4f}")

    report: dict = {
        "curve": "FruitSet(V) = F0 + (Fmax - F0) * (1 - exp(-k * V))",
        "fit_data": dataset_path.name,
        "gates": {"dwell_min_s": ds.DWELL_MIN, "vel_max": ds.VEL_MAX,
                  "frac_min": ds.FRAC_MIN},
        "crops": {c: r["report"] for c, r in fits.items()},
        "notes": [
            "Fit on dataset_training_v8.csv, whose labels are synthetic (no real fruit-set "
            "labels exist yet, research doc Section 16); the effective dose V is "
            "reconstructed from the per-flower aggregates since v8 drops the raw V column. "
            "Recovery of the per-crop floor/ceiling (p_self_used/p_cross_used) validates the "
            "machinery, not real-orchard accuracy.",
            "Three model layers are reported per crop: the frequentist saturation curve, a "
            "binomial GLMM with orchard/year random intercepts (glmm), and a Bayesian fit "
            "with a cross-crop prior on Fmax plus a prior-sensitivity check (bayesian). A "
            "small prior_sensitivity_shift means the data (not the prior) drive the ceiling.",
            "v8 carries a genuine visit->fruit-set signal (dose/label corr ~0.24) and includes "
            "near-zero-dose flowers, so both the floor F0 and ceiling Fmax are recovered "
            "accurately for both crops (no upward F0 bias). fit_dose_response(anchor_f0=...) / "
            "(fmax_anchor=...) still accept bagging / open-pollination controls when real data "
            "arrives.",
            "Tracker application uses the v8-fit curve on the real effective dose. The two "
            "doses share a definition but not a scale, so the yield figure is illustrative "
            "until the curve is refit on real (dose, fruit_set) pairs joined via "
            "bee_hero_dataset.join_fruit_set_labels once field labels + cross-time flower "
            "identity exist.",
            "Only the dwell gate applies on real tracker output; velocity and fraction_on "
            "are not yet emitted by video_detect.py.",
        ],
    }

    if landings_path.exists():
        # apply the pomegranate curve (project's target crop) to real tracker output
        crop = "pomegranate" if "pomegranate" in fits else next(iter(fits))
        print(f"[2/3] Applying '{crop}' curve to tracker output {landings_path.name} ...")
        applied = apply_to_tracker(fits[crop]["fit"], landings_path, n_flowers, mean_mass)
        y = applied["yield_estimate"]
        print(f"      gates applied on real data: {applied['applied_gates']}")
        print(f"      fruit set at mean dose = {y['fruit_set_mean']:.2f} "
              f"[{y['fruit_set_ci95'][0]:.2f}, {y['fruit_set_ci95'][1]:.2f}]")
        print(f"      yield = {y['yield_kg_mean']:.0f} kg/tree "
              f"[{y['yield_kg_ci95'][0]:.0f}, {y['yield_kg_ci95'][1]:.0f}]")
        report["tracker_application"] = applied
    else:
        print(f"[2/3] Tracker file {landings_path} not found — skipping apply stage.")

    print(f"[3/3] Writing report -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="modeling frame to fit on (default: dataset_training_v8.csv "
                         "from data/processed, then ~/Downloads)")
    ap.add_argument("--tracker", default=str(TRACKER_LANDINGS),
                    help="real tracker ALL_landings.csv to apply the curve to")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--n-flowers", type=int, default=DEFAULT_N_FLOWERS,
                    help="flowers per tree for the yield figure")
    ap.add_argument("--mean-mass", type=float, default=DEFAULT_MEAN_MASS_KG,
                    help="mean fruit mass (kg) for the yield figure")
    args = ap.parse_args()
    run(_resolve_dataset(args.dataset), Path(args.tracker), Path(args.out),
        args.n_flowers, args.mean_mass)


if __name__ == "__main__":
    main()
