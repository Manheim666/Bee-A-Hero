"""Frequentist visit-to-fruit-set dose-response curve (Section 6.2 of the research doc).

Fits the saturating ("diminishing-returns") fruit-set curve

    FruitSet(V) = F0 + (Fmax - F0) * (1 - exp(-k * V))

to per-flower dose ``V`` and binary ``fruit_set`` outcomes, where

  * ``F0``   is fruit set at zero insect visits (partial self-fertility, P_self),
  * ``Fmax`` is the upper asymptote (maximum attainable fruit set, P_max),
  * ``k``    is the rate constant; ``k = 3 / V*`` puts ~95% of the rise by dose ``V*``.

This is the same functional form ``generate_bee_data.py`` uses to synthesise labels, so
on synthetic data the fit is validated by recovering the known ``(F0, Fmax, k)``.

Because the label is a single Bernoulli draw per flower, the curve is fit to **binomial
proportions** over dose bins (weighted by bin count), and parameter uncertainty is
obtained by **bootstrap over flowers** (cluster-level resampling is a future refinement,
Section 3.2). Fit quality is reported as binomial AIC/BIC at the per-flower level and
relative to a predict-the-mean baseline (Section 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SEED = 42


# --------------------------------------------------------------------------- #
# The curve
# --------------------------------------------------------------------------- #
def fruit_set_curve(V, F0: float, Fmax: float, k: float):
    """Saturating fruit-set probability at dose ``V``: ``F0 + (Fmax-F0)(1-exp(-kV))``."""
    V = np.asarray(V, dtype=float)
    return F0 + (Fmax - F0) * (1.0 - np.exp(-k * V))


def k_from_v_star(v_star: float) -> float:
    """Rate constant that reaches ~95% of ``Fmax`` by dose ``v_star`` (``k = 3 / V*``)."""
    return 3.0 / float(v_star)


# --------------------------------------------------------------------------- #
# Fit result container
# --------------------------------------------------------------------------- #
@dataclass
class DoseResponseFit:
    """A fitted curve plus its uncertainty and goodness-of-fit."""
    F0: float
    Fmax: float
    k: float
    ci: dict[str, tuple[float, float]]        # 95% bootstrap interval per parameter
    boot: np.ndarray                          # (n_boot, 3) bootstrap parameter draws
    n_flowers: int
    n_events: int                             # rarer-class count (min of set / no-set)
    aic: float
    bic: float
    baseline_logloss: float                   # predict-the-mean reference
    model_logloss: float
    v_star: float = field(default=np.nan)     # dose reaching 95% of Fmax (3/k)

    def predict(self, V):
        """Fruit-set probability at dose ``V`` using the point estimate."""
        return fruit_set_curve(V, self.F0, self.Fmax, self.k)

    def as_dict(self) -> dict:
        return {
            "F0": round(self.F0, 4), "Fmax": round(self.Fmax, 4), "k": round(self.k, 5),
            "v_star": round(self.v_star, 3),
            "ci95": {p: [round(lo, 4), round(hi, 4)] for p, (lo, hi) in self.ci.items()},
            "n_flowers": self.n_flowers, "n_events": self.n_events,
            "aic": round(self.aic, 2), "bic": round(self.bic, 2),
            "baseline_logloss": round(self.baseline_logloss, 4),
            "model_logloss": round(self.model_logloss, 4),
            "delta_logloss": round(self.model_logloss - self.baseline_logloss, 4),
        }


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _binned(V: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Bin flowers by dose and return per-bin dose, fruit-set rate and count."""
    b = pd.qcut(V, min(n_bins, np.unique(V).size), duplicates="drop")
    g = pd.DataFrame({"V": V, "y": y, "b": b}).groupby("b", observed=True)
    return pd.DataFrame({"V": g["V"].mean(), "rate": g["y"].mean(),
                         "n": g["y"].size()}).reset_index(drop=True)


def _fit_once(V: np.ndarray, y: np.ndarray, n_bins: int,
              f0_anchor: float | None, fmax_anchor: float | None = None) -> np.ndarray:
    """One weighted least-squares fit to binned proportions -> (F0, Fmax, k).

    ``f0_anchor`` pins ``F0`` (a bagged-flower control) and ``fmax_anchor`` pins ``Fmax``
    (an open-pollination control); either or both may be given, otherwise the parameter is
    free. Bins are weighted by count, so dense bins dominate.
    """
    g = _binned(V, y, n_bins)
    lo_f0 = 0.0 if f0_anchor is None else max(0.0, f0_anchor - 1e-6)
    hi_f0 = 1.0 if f0_anchor is None else min(1.0, f0_anchor + 1e-6)
    lo_fm = 0.0 if fmax_anchor is None else max(0.0, fmax_anchor - 1e-6)
    hi_fm = 1.0 if fmax_anchor is None else min(1.0, fmax_anchor + 1e-6)
    p0 = [float(np.clip(g.rate.min(), lo_f0, hi_f0)),
          float(np.clip(g.rate.max() + 0.05, lo_fm, hi_fm)), 0.3]
    popt, _ = curve_fit(
        fruit_set_curve, g.V.to_numpy(), g.rate.to_numpy(), p0=p0,
        bounds=([lo_f0, lo_fm, 1e-3], [hi_f0, hi_fm, 10.0]),
        sigma=1.0 / np.sqrt(g.n.to_numpy()), absolute_sigma=False, maxfev=40000)
    return popt


def _binomial_logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean per-flower binomial log-loss (lower is better)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_dose_response(V, fruit_set, *, n_bins: int = 20, n_boot: int = 400,
                      anchor_f0: bool = False, fmax_anchor: float | None = None,
                      seed: int = SEED) -> DoseResponseFit:
    """Fit ``FruitSet(V)`` to per-flower dose and binary outcome, with bootstrap CIs.

    Parameters
    ----------
    V, fruit_set
        Per-flower effective dose and 0/1 fruit-set label (equal length).
    n_bins
        Number of dose quantile bins used to form binomial proportions for the fit.
    n_boot
        Bootstrap resamples (over flowers) for the 95% parameter intervals.
    anchor_f0
        If True, pin the lower asymptote ``F0`` to the empirical fruit-set rate of the
        **near-zero-dose** flowers (``V <= 0.5``), a stand-in for a bagged-flower control.
        Only meaningful when such flowers exist; off by default because a purely
        observational dataset rarely contains true zero-visit flowers, and anchoring on
        the lowest *observed* (non-zero) dose biases ``F0`` upward (Section 4.2).
    fmax_anchor
        If given, pin the upper asymptote ``Fmax`` to this value — the fruit set of
        open-pollination control flowers (Section 4.2). Rescues the ceiling when the
        observed dose never reaches saturation; ``None`` leaves ``Fmax`` free.
    """
    V = np.asarray(V, dtype=float)
    y = np.asarray(fruit_set, dtype=float)
    if V.size != y.size:
        raise ValueError("V and fruit_set must have the same length")
    rng = np.random.default_rng(seed)

    f0_anchor = None
    if anchor_f0:
        zero = V <= 0.5                               # true near-zero-dose flowers only
        if zero.sum() >= 10:
            f0_anchor = float(y[zero].mean())

    popt = _fit_once(V, y, n_bins, f0_anchor, fmax_anchor)

    # bootstrap over flowers
    draws = []
    idx = np.arange(V.size)
    for _ in range(n_boot):
        s = rng.choice(idx, V.size, replace=True)
        try:
            draws.append(_fit_once(V[s], y[s], n_bins, f0_anchor, fmax_anchor))
        except Exception:
            continue
    draws = np.array(draws) if draws else popt[None, :]
    ci = {name: (float(np.percentile(draws[:, j], 2.5)),
                 float(np.percentile(draws[:, j], 97.5)))
          for j, name in enumerate(("F0", "Fmax", "k"))}

    # goodness of fit at the per-flower binomial level
    p_model = fruit_set_curve(V, *popt)
    model_ll = _binomial_logloss(y, p_model)
    base_ll = _binomial_logloss(y, np.full_like(y, y.mean()))
    n = V.size
    n_events = int(min(y.sum(), n - y.sum()))
    nll = model_ll * n                                   # total negative log-likelihood
    aic = 2 * 3 + 2 * nll
    bic = 3 * np.log(n) + 2 * nll

    return DoseResponseFit(
        F0=float(popt[0]), Fmax=float(popt[1]), k=float(popt[2]), ci=ci, boot=draws,
        n_flowers=n, n_events=n_events, aic=aic, bic=bic,
        baseline_logloss=base_ll, model_logloss=model_ll,
        v_star=float(3.0 / popt[2]) if popt[2] > 0 else np.nan)
