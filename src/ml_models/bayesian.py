"""Bayesian dose-response fit with a cross-crop prior (research doc Section 4).

Pomegranate has no controlled-visit anchor and a small effective sample, so a purely
data-driven fit cannot identify the upper asymptote when saturating doses are never observed
(exactly what we see on v6: ``Fmax`` runs to the bound). The Bayesian answer is to let a curve
shape we *do* trust from other crops inform the fit without dictating it:

    p(theta | data) ~ Binomial-likelihood(data | theta) * prior(theta),

with a **weakly-informative prior on Fmax centred on a cross-crop value** and a mild prior on
the rate. The floor ``F0`` is anchored empirically (bagged-flower stand-in) when near-zero-dose
flowers exist. Sampling is a compact Metropolis sampler in NumPy, so it runs in the existing
environment with no PyMC dependency; swap in PyMC/Bambi for random effects + measurement-error
layers when real labels arrive — the likelihood and priors are identical.

A **prior-sensitivity check** (refit under a vague and a tighter prior, report how far the
posterior moves) is provided and is mandatory before any Bayesian number is quoted: a
conclusion that changes with the prior is reporting the prior, not the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml_models.dose_response import fruit_set_curve

SEED = 42


@dataclass
class BayesFit:
    """Posterior summary for the saturation-curve parameters."""
    F0: float
    posterior: pd.DataFrame                    # columns Fmax, k (F0 fixed by anchor)
    prior_fmax: tuple[float, float]            # (mu, sd) used for Fmax

    def summary(self) -> dict:
        q = lambda s, a: float(np.percentile(self.posterior[s], a))
        return {
            "F0_anchor": round(self.F0, 4),
            "Fmax_mean": round(float(self.posterior.Fmax.mean()), 4),
            "Fmax_ci95": [round(q("Fmax", 2.5), 4), round(q("Fmax", 97.5), 4)],
            "k_mean": round(float(self.posterior.k.mean()), 4),
            "k_ci95": [round(q("k", 2.5), 4), round(q("k", 97.5), 4)],
            "prior_Fmax": {"mu": self.prior_fmax[0], "sd": self.prior_fmax[1]},
            "n_draws": len(self.posterior),
        }


def _binned(V: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    b = pd.qcut(V, min(n_bins, np.unique(V).size), duplicates="drop")
    g = pd.DataFrame({"V": V, "y": y, "b": b}).groupby("b", observed=True)
    return pd.DataFrame({"V": g["V"].mean(), "k": g["y"].sum(), "n": g["y"].size()}).reset_index(drop=True)


def bayes_dose_response(V, fruit_set, *, prior_fmax_mu: float = 0.72, prior_fmax_sd: float = 0.10,
                        prior_k_mu: float = 0.3, prior_k_sd: float = 0.5, n_bins: int = 20,
                        n_iter: int = 24000, burn: int = 6000, thin: int = 5,
                        seed: int = SEED) -> BayesFit:
    """Sample the posterior of ``(Fmax, k)`` for the saturation curve given a cross-crop prior.

    ``F0`` is anchored to the empirical fruit-set rate of near-zero-dose flowers (``V <= 0.5``)
    if enough exist, else to the lowest-dose bin. The likelihood treats each dose bin as
    binomial. The prior on ``Fmax`` is Normal(``prior_fmax_mu``, ``prior_fmax_sd``) — the
    cross-crop shape prior that rescues an unidentified ceiling.
    """
    V = np.asarray(V, float)
    y = np.asarray(fruit_set, float)
    rng = np.random.default_rng(seed)

    zero = V <= 0.5
    F0 = float(y[zero].mean()) if zero.sum() >= 10 else float(y[V <= np.quantile(V, 0.05)].mean())
    g = _binned(V, y, n_bins)
    Vb, kb, nb = g.V.to_numpy(), g.k.to_numpy(), g.n.to_numpy()

    def log_post(theta):
        Fmax, logk = theta
        if not (F0 < Fmax < 0.999 and -6.0 < logk < 4.0):
            return -np.inf
        k = np.exp(logk)
        p = np.clip(fruit_set_curve(Vb, F0, Fmax, k), 1e-9, 1 - 1e-9)
        ll = np.sum(kb * np.log(p) + (nb - kb) * np.log(1 - p))
        lp = -0.5 * ((Fmax - prior_fmax_mu) / prior_fmax_sd) ** 2
        lp += -0.5 * ((k - prior_k_mu) / prior_k_sd) ** 2
        return ll + lp

    theta = np.array([min(0.9, max(F0 + 0.05, prior_fmax_mu)), np.log(prior_k_mu)])
    lp = log_post(theta)
    step = np.array([0.03, 0.20])
    draws = []
    for i in range(n_iter):
        cand = theta + step * rng.standard_normal(2)
        lpc = log_post(cand)
        if np.log(rng.random()) < lpc - lp:
            theta, lp = cand, lpc
        if i >= burn and (i - burn) % thin == 0:
            draws.append((theta[0], np.exp(theta[1])))
    post = pd.DataFrame(draws, columns=["Fmax", "k"])
    return BayesFit(F0=F0, posterior=post, prior_fmax=(prior_fmax_mu, prior_fmax_sd))


def prior_sensitivity(V, fruit_set, *, mu: float = 0.72, seed: int = SEED) -> pd.DataFrame:
    """Refit under a vague and a tighter Fmax prior; report how far the posterior moves.

    A small shift means the data drive the ceiling; a large shift means the prior does. This
    check is mandatory before quoting any Bayesian asymptote (research doc Section 4.2).
    """
    rows = []
    for label, sd in [("vague (sd=0.25)", 0.25), ("tight (sd=0.05)", 0.05)]:
        fit = bayes_dose_response(V, fruit_set, prior_fmax_mu=mu, prior_fmax_sd=sd, seed=seed)
        s = fit.summary()
        rows.append({"prior": label, "Fmax_mean": s["Fmax_mean"], "Fmax_ci95": s["Fmax_ci95"]})
    out = pd.DataFrame(rows)
    out.attrs["posterior_shift"] = abs(out.Fmax_mean.iloc[0] - out.Fmax_mean.iloc[1])
    return out
