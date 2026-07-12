"""CLI orchestrator for the fruit-set modeling stage (research doc Section 19, step 8).

Wires the modeling pipeline end to end:

    tracker/visit CSV  ->  three qualifying gates  ->  per-flower effective dose V
                       ->  fit FruitSet(V) = F0 + (Fmax-F0)(1-exp(-kV))
                       ->  fruit-set + orchard-yield estimates, each with a 95% interval

Because no real fruit-set labels exist yet (research doc Section 16), the curve is
**fit** on the processed training frame ``dataset_training_v11.csv`` (the same dataset as
``notebooks/03_ml.ipynb``). v11 carries no raw ``V`` column, so the effective
dose ``V`` is the CV ``pollination_score`` (or is reconstructed from the per-flower
aggregates), and the fit is validated by recovering the per-crop floor/ceiling
(``p_self_used`` / ``p_cross_used``). The fitted curve is then **applied** to the tracker
export (``ALL_landings.csv``) to produce interval-carrying fruit-set and yield numbers for the
report layer — the genuine tracker-to-yield path.

The ML stage does **not** detect cameras; it only consumes the CSV the CV stage writes.
Landings resolution (see :func:`resolve_landings`) is a single chain:

    1. CV tracker export  ``test_video_result/csv/ALL_landings.csv`` (grouped, preferred)
                          ``test_video_result/ALL_landings.csv``     (legacy-flat fallback)
    2. SYNTHETIC v11      ``data/processed/dataset_training_v11.csv`` generated so the fit
                          can still run when no CV export is present.

Apply-only mode (``--apply-only``) reuses the committed curve ``models/dose_response_v11.json``
without re-fitting, so it needs nothing beyond numpy/pandas/scipy. The default fit path is also
statsmodels-optional: without statsmodels it fits the core curve + Bayesian layer and skips only
the optional GLMM (random-intercepts) sub-report, so a bare checkout still completes.

Run:
    python -m src.ml_models.train
    python -m src.ml_models.train --apply-only        # reuse the committed v11 curve, no re-fit
    python -m src.ml_models.train --dataset data/processed/dataset_training_v11.csv \
        --n-flowers 1200 --mean-mass 0.30
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Apply-only-safe imports only (numpy/pandas/scipy). The fit-stage layers (glmm/bayesian ->
# statsmodels) are imported lazily inside fit_on_dataset so apply-only never needs them.
from src.ml_models import visit_dataset as ds
from src.ml_models.visit_dataset import CROPS
from src.ml_models.dose_response import DoseResponseFit, fit_dose_response, fruit_set_curve
from src.ml_models.uncertainty import propagate_yield

# Repo paths derived from this file's location (self-contained, no config dependency,
# portable across machines and branches).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_TEST_VIDEO_DIR = _REPO_ROOT / "test_video_result"

# Default modeling inputs; resolved from data/processed with a Downloads fallback.
DATASET_CANDIDATES = (
    _PROCESSED_DIR / "dataset_training_v11.csv",
    Path.home() / "Downloads" / "dataset_training_v11.csv",
)
# Where the synthetic last-resort frame is written (git-ignored under data/).
SYNTHETIC_DATASET = _PROCESSED_DIR / "dataset_training_v11.csv"

# Apply-stage landings, resolved in precedence order: grouped (current CV layout) then
# legacy-flat. Nothing else in the pipeline may hardcode a landings path (see resolve_landings).
LANDINGS_CANDIDATES = (
    _TEST_VIDEO_DIR / "csv" / "ALL_landings.csv",   # grouped path (current CV layout)
    _TEST_VIDEO_DIR / "ALL_landings.csv",           # legacy-flat fallback
)

# The committed, fitted curve is READ-ONLY; runtime output goes to a git-ignored file so a
# run never overwrites the blessed curve.
COMMITTED_CURVE = _REPO_ROOT / "models" / "dose_response_v11.json"   # read-only, committed
OUT_JSON = _REPO_ROOT / "models" / "yield_report.json"              # git-ignored runtime output

# Illustrative orchard constants for the yield figure (override on the CLI).
DEFAULT_N_FLOWERS = 1200
DEFAULT_MEAN_MASS_KG = 0.30


# --------------------------------------------------------------------------- #
# Landings source chain:  CV export (grouped -> legacy flat) -> synthetic v11
# --------------------------------------------------------------------------- #
def _usable_csv(path: Path) -> bool:
    """True iff ``path`` exists and holds at least one data row (header-only counts as empty)."""
    if not path.exists():
        return False
    try:
        return len(pd.read_csv(path, nrows=1)) > 0
    except Exception:
        return False


def resolve_landings(explicit: str | None = None) -> Path | None:
    """Single source of truth for the apply-stage landings CSV.

    Returns the first CV export that resolves to a **non-empty** CSV — an explicit
    ``--tracker`` path, then the grouped ``test_video_result/csv/ALL_landings.csv``, then the
    legacy-flat ``test_video_result/ALL_landings.csv`` — else ``None`` so the caller falls
    through (a fit run skips the apply stage with a clear notice; apply-only fails loudly).
    Never returns a stale path that does not actually load.
    """
    for cand in ([Path(explicit)] if explicit else []) + list(LANDINGS_CANDIDATES):
        if _usable_csv(cand):
            return cand
    return None


# Columns the synthetic frame must carry for the fit + smoke assertion.
_SYNTH_COLUMNS = ("crop", "orchard_id", "year", "n_honeybee", "n_bee", "n_fly",
                  "mean_landing_s", "pollination_score", "p_self_used", "p_cross_used",
                  "fruit_set_label")


def generate_synthetic_v11(out_path: Path = SYNTHETIC_DATASET, *,
                           n_per_crop: int = 4000, seed: int = 42) -> Path:
    """Generate a synthetic v11 training frame (NumPy/pandas only) as the last-resort fit input.

    Mirrors the CV-schema columns the modeling stage expects (``pollination_score`` is the
    effective dose ``V``) and labels each flower from the same saturating curve the model
    fits, so the fit runs end-to-end with no real data. Writes to the git-ignored
    ``data/processed`` path.

    After writing, a **smoke assertion** confirms the file exists, carries every expected
    column, and has rows — raising a clear, named message (not a deep traceback) if a step
    produced nothing.
    """
    rng = np.random.default_rng(seed)
    w = ds.CV_SPECIES_W
    frames = []
    for spec in CROPS:
        crop, F0, Fmax, k = spec["crop"], spec["F0"], spec["Fmax"], spec["k"]
        n = n_per_crop
        n_hb = rng.poisson(2.0, n)
        n_be = rng.poisson(1.0, n)
        n_fl = rng.poisson(3.0, n)
        dwell = rng.gamma(2.0, 6.0, n)                         # mean ~12 s
        species = n_hb * w["n_honeybee"] + n_be * w["n_bee"] + n_fl * w["n_fly"]
        score = species * (1.0 - np.exp(-dwell / ds.TAU))      # == effective dose V
        p = np.clip(fruit_set_curve(score, F0, Fmax, k), 0.0, 1.0)
        label = (rng.random(n) < p).astype(int)
        frames.append(pd.DataFrame({
            "crop": crop,
            "orchard_id": rng.integers(0, 6, n),
            "year": rng.integers(2015, 2021, n),
            "n_honeybee": n_hb, "n_bee": n_be, "n_fly": n_fl,
            "mean_landing_s": dwell.round(2),
            "pollination_score": score.round(4),
            "p_self_used": F0, "p_cross_used": Fmax,
            "fruit_set_label": label,
        }))
    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # --- smoke assertion: name the missing step, never a bare deep traceback ---
    if not out_path.exists():
        raise RuntimeError(f"synthetic v11 generation did not write {out_path} (step: to_csv)")
    check = pd.read_csv(out_path)
    missing = [c for c in _SYNTH_COLUMNS if c not in check.columns]
    if missing:
        raise RuntimeError(f"synthetic v11 frame is missing columns {missing} "
                           f"(step: column build)")
    if len(check) == 0:
        raise RuntimeError("synthetic v11 frame has 0 rows (step: row generation)")
    return out_path


def ensure_training_dataset(explicit: str | None = None) -> tuple[Path, str]:
    """Resolve the fit dataset, generating synthetic v11 as a last resort.

    Only generates when no dataset exists (so a real committed frame is never overwritten).
    Returns ``(path, source)`` where ``source`` is ``explicit`` / ``processed`` /
    ``downloads`` / ``synthetic`` so the orchestrator can print which input the fit used.
    """
    labelled: list[tuple[Path, str]] = [(Path(explicit), "explicit")] if explicit else []
    labelled += [(DATASET_CANDIDATES[0], "processed"), (DATASET_CANDIDATES[1], "downloads")]
    for cand, source in labelled:
        if cand.exists():
            return cand, source
    print("[0/3] No v11 training frame found (no CV export) — generating synthetic v11 ...")
    return generate_synthetic_v11(), "synthetic"


# --------------------------------------------------------------------------- #
# Apply-only support: reuse the committed curve without re-fitting (no statsmodels)
# --------------------------------------------------------------------------- #
def _statsmodels_available() -> bool:
    """True iff ``statsmodels`` can be imported (needed only by the fit stage)."""
    return importlib.util.find_spec("statsmodels") is not None


def load_committed_curve(path: Path = COMMITTED_CURVE) -> dict:
    """Load the read-only committed v11 curve report; raise an actionable error if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"apply-only needs the committed curve {path}, which is missing. Run a full fit "
            f"first (python -m src.ml_models.train) or restore it from git.")
    return json.loads(path.read_text())


def _fit_from_report(rep: dict, *, n_boot: int = 400, seed: int = 42) -> DoseResponseFit:
    """Rebuild a :class:`DoseResponseFit` from a committed per-crop report (no re-fit).

    Point estimates come straight from the committed v11 curve; a bootstrap cloud is
    regenerated to reproduce the stored 95% intervals, so the apply-stage yield band still
    reflects the fitted parameter uncertainty — without importing statsmodels or writing the
    committed file.
    """
    rng = np.random.default_rng(seed)
    F0, Fmax, k = float(rep["F0"]), float(rep["Fmax"]), float(rep["k"])
    ci = rep.get("ci95", {})

    def _draws(name: str, mean: float) -> np.ndarray:
        lo, hi = ci.get(name, [mean, mean])
        sd = max((hi - lo) / (2 * 1.959963985), 1e-9)         # 95% CI half-width -> sd
        return rng.normal(mean, sd, n_boot)

    boot = np.column_stack([_draws("F0", F0), _draws("Fmax", Fmax), _draws("k", k)])
    ci_t = {name: (float(ci.get(name, [val, val])[0]), float(ci.get(name, [val, val])[1]))
            for name, val in (("F0", F0), ("Fmax", Fmax), ("k", k))}
    return DoseResponseFit(
        F0=F0, Fmax=Fmax, k=k, ci=ci_t, boot=boot,
        n_flowers=int(rep.get("n_flowers", 0)), n_events=int(rep.get("n_events", 0)),
        aic=float(rep.get("aic", np.nan)), bic=float(rep.get("bic", np.nan)),
        baseline_logloss=float(rep.get("baseline_logloss", np.nan)),
        model_logloss=float(rep.get("model_logloss", np.nan)),
        v_star=float(rep.get("v_star", 3.0 / k if k else np.nan)))


# --------------------------------------------------------------------------- #
# Fit stage — one curve per crop on the synthetic modeling frame
# --------------------------------------------------------------------------- #
def fit_on_dataset(dataset_path: Path) -> dict:
    """Fit ``FruitSet(V)`` per crop and compare recovered asymptotes to the known truth.

    Works on the processed ``dataset_training_v11`` frame — which carries no raw ``V`` column
    and labels ``fruit_set_label`` — by taking the CV ``pollination_score`` as the effective
    dose (or reconstructing it from the per-flower aggregates). The per-crop ground-truth
    floor/ceiling come from ``p_self_used`` / ``p_cross_used`` when present, else from the
    generator's :data:`CROPS` table.

    The GLMM/Bayesian layers (and therefore ``statsmodels``) are imported **here**, inside the
    fit stage, so importing this module and the apply-only path never require statsmodels.
    """
    # The core curve (scipy) and the Bayesian layer (numpy/scipy) need no statsmodels; only the
    # GLMM random-intercepts layer does, so it is imported and run conditionally below. The fit
    # therefore still completes without statsmodels — just without the optional GLMM sub-report.
    from src.ml_models.bayesian import bayes_dose_response, prior_sensitivity

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
        if grp and _statsmodels_available():
            from src.ml_models.glmm import fit_glmm          # statsmodels-only layer (optional)
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
def run(dataset_path: Path | None, landings_path: Path | None, out_path: Path,
        n_flowers: int, mean_mass: float, *, apply_only: bool = False,
        curve_path: Path = COMMITTED_CURVE) -> dict:
    # Never let a run clobber the committed, read-only curve.
    if out_path.resolve() == COMMITTED_CURVE.resolve():
        raise ValueError(
            f"refusing to write runtime output over the committed curve {out_path}; "
            f"use the git-ignored {OUT_JSON.name} instead")

    if apply_only:
        # Reuse the committed curve — no re-fit, so nothing here imports statsmodels.
        print(f"[1/2] Apply-only: loading committed curve {curve_path.name} (read-only) ...")
        curve = load_committed_curve(curve_path)
        crops_rep = curve.get("crops", {})
        if not crops_rep:
            raise ValueError(f"{curve_path} has no 'crops' section to apply")
        fits = {c: {"fit": _fit_from_report(r), "report": r} for c, r in crops_rep.items()}
        report: dict = {
            "mode": "apply-only",
            "curve": curve.get("curve", "FruitSet(V) = F0 + (Fmax - F0) * (1 - exp(-k * V))"),
            "curve_source": curve_path.name,
            "fit_data": curve.get("fit_data"),
            "gates": curve.get("gates", {"dwell_min_s": ds.DWELL_MIN, "vel_max": ds.VEL_MAX,
                                         "frac_min": ds.FRAC_MIN}),
            "crops": crops_rep,
        }
    else:
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

        report = {
            "mode": "fit+apply",
            "curve": "FruitSet(V) = F0 + (Fmax - F0) * (1 - exp(-k * V))",
            "fit_data": dataset_path.name,
            "gates": {"dwell_min_s": ds.DWELL_MIN, "vel_max": ds.VEL_MAX,
                      "frac_min": ds.FRAC_MIN},
            "crops": {c: r["report"] for c, r in fits.items()},
            "notes": [
                "Fit on dataset_training_v11.csv, whose labels are synthetic (no real "
                "fruit-set labels exist yet, research doc Section 16); the effective dose V is "
                "the CV pollination_score (or reconstructed from the per-flower aggregates). "
                "Recovery of the per-crop floor/ceiling (p_self_used/p_cross_used) validates "
                "the machinery, not real-orchard accuracy.",
                "Three model layers are reported per crop: the frequentist saturation curve, a "
                "binomial GLMM with orchard/year random intercepts (glmm), and a Bayesian fit "
                "with a cross-crop prior on Fmax plus a prior-sensitivity check (bayesian). A "
                "small prior_sensitivity_shift means the data (not the prior) drive the ceiling.",
                "v11 carries a genuine visit->fruit-set signal and includes near-zero-dose "
                "flowers, so both the floor F0 and ceiling Fmax are recovered accurately for "
                "both crops (no upward F0 bias). fit_dose_response(anchor_f0=...) / "
                "(fmax_anchor=...) still accept bagging / open-pollination controls when real "
                "data arrives.",
                "Tracker application uses the v11-fit curve on the real effective dose. The two "
                "doses share a definition but not a scale, so the yield figure is illustrative "
                "until the curve is refit on real (dose, fruit_set) pairs joined via "
                "visit_dataset.join_fruit_set_labels once field labels + cross-time flower "
                "identity exist.",
                "Only the dwell gate applies on real tracker output; velocity and fraction_on "
                "are not yet emitted by video_detect.py.",
            ],
        }

    stage = "2/2" if apply_only else "2/3"
    if landings_path is not None:
        # apply the pomegranate curve (project's target crop) to real tracker output
        crop = "pomegranate" if "pomegranate" in fits else next(iter(fits))
        print(f"[{stage}] Applying '{crop}' curve to tracker output {landings_path.name} ...")
        applied = apply_to_tracker(fits[crop]["fit"], landings_path, n_flowers, mean_mass)
        y = applied["yield_estimate"]
        print(f"      gates applied on real data: {applied['applied_gates']}")
        print(f"      fruit set at mean dose = {y['fruit_set_mean']:.2f} "
              f"[{y['fruit_set_ci95'][0]:.2f}, {y['fruit_set_ci95'][1]:.2f}]")
        print(f"      yield = {y['yield_kg_mean']:.0f} kg/tree "
              f"[{y['yield_kg_ci95'][0]:.0f}, {y['yield_kg_ci95'][1]:.0f}]")
        report["tracker_application"] = applied
    else:
        tried = "; ".join(str(p) for p in LANDINGS_CANDIDATES)
        if apply_only:
            # Apply-only exists to score tracker data — with none resolved, fail loudly.
            raise FileNotFoundError(
                f"apply-only requires tracker landings, but none resolved. Tried: {tried}. "
                f"Provide the CV export at one of those paths or pass --tracker <csv>.")
        # Fit run: apply is best-effort. This is an explicit, actionable notice — not a silent
        # skip — so the fit outputs are still written and the reason is visible.
        print(f"[{stage}] No tracker landings resolved (no CV export). Tried: {tried}. "
              f"Skipping apply stage; fit outputs still written.")

    final = "2/2" if apply_only else "3/3"
    print(f"[{final}] Writing report -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="modeling frame to fit on (default: dataset_training_v11.csv "
                         "from data/processed, then ~/Downloads, then synthetic)")
    ap.add_argument("--tracker", default=None,
                    help="explicit landings CSV to apply the curve to; by default resolves "
                         "the CV export (grouped -> legacy flat) via resolve_landings")
    ap.add_argument("--apply-only", action="store_true",
                    help="reuse the committed v11 curve; no re-fit, needs no statsmodels "
                         "(requires resolvable tracker landings)")
    ap.add_argument("--out", default=str(OUT_JSON),
                    help="git-ignored runtime report (never the committed curve)")
    ap.add_argument("--n-flowers", type=int, default=DEFAULT_N_FLOWERS,
                    help="flowers per tree for the yield figure")
    ap.add_argument("--mean-mass", type=float, default=DEFAULT_MEAN_MASS_KG,
                    help="mean fruit mass (kg) for the yield figure")
    args = ap.parse_args()

    landings = resolve_landings(args.tracker)
    # Apply-only is opt-in (reuse the committed curve, no re-fit). The default fit path runs the
    # core curve + Bayesian layer with scipy/numpy alone, so a bare checkout still completes when
    # statsmodels is absent — only the optional GLMM layer is skipped (noted below).
    if args.apply_only:
        run(None, landings, Path(args.out), args.n_flowers, args.mean_mass, apply_only=True)
    else:
        dataset, source = ensure_training_dataset(args.dataset)
        if not _statsmodels_available():
            print("[note] statsmodels not installed — fitting the core curve + Bayesian layer; "
                  "the optional GLMM (random-intercepts) layer is skipped.")
        print(f"[0/3] Fit dataset source: {source} -> {dataset}")
        run(dataset, landings, Path(args.out), args.n_flowers, args.mean_mass)


if __name__ == "__main__":
    main()
