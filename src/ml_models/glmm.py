"""Binomial GLMM with random intercepts (research doc Section 2.3).

Flowers on the same plant/orchard and in the same clip/date are not independent — they share
genotype, water status, camera, and weather — so an ordinary binomial GLM understates the
standard errors (pseudoreplication). This module fits

    fruitset_i ~ Binomial(1, p_i),
    logit(p_i) = b0 + b1 * dose_i + u_orchard(i) + u_date(i),
    u_. ~ Normal(0, sigma^2_.),

with the grouping factors as **random intercepts**, using statsmodels' variational-Bayes
binomial mixed GLM. The variance components ``sigma^2_.`` *are* the non-independence,
estimated rather than assumed away; the dose slope ``b1`` is the visit effect adjusted for it.

The dose is standardised before fitting for numerical stability, so ``b1`` is the change in
log-odds per one standard deviation of dose. This is the doc's GLMM step (build order step 4),
validated here on the synthetic ``dataset_training_v6`` frame; on real data the same call adds
plant/date random effects once cross-time flower identity exists.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

SEED = 42


@dataclass
class GLMMFit:
    """Fitted binomial GLMM: dose effect plus random-intercept variance components."""
    dose_beta: float                          # log-odds per 1 SD of dose
    dose_beta_sd: float
    intercept: float
    var_components: dict[str, float]          # group -> random-intercept SD (logit scale)
    n: int
    groups: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "dose_beta_per_sd": round(self.dose_beta, 4),
            "dose_beta_sd": round(self.dose_beta_sd, 4),
            "dose_effect_positive": bool(self.dose_beta > 0),
            "intercept": round(self.intercept, 4),
            "random_intercept_sd": {g: round(v, 4) for g, v in self.var_components.items()},
            "n_flowers": self.n,
        }


def fit_glmm(df: pd.DataFrame, *, dose_col: str = "V", target: str = "fruit_set_label",
             groups: tuple[str, ...] = ("orchard_id", "year"),
             max_rows: int = 8000, seed: int = SEED) -> GLMMFit:
    """Fit a binomial mixed GLM of ``target`` on standardised dose with random intercepts.

    Parameters
    ----------
    df
        Per-flower frame carrying ``dose_col``, ``target`` and the grouping columns.
    dose_col, target
        Dose (predictor) and binary fruit-set (response) column names.
    groups
        Columns used as random intercepts (e.g. orchard and date/year).
    max_rows
        Optional subsample cap — the variational fit is quick but not free on 80k rows;
        set to 0 to use all rows.
    """
    d = df[[dose_col, target, *groups]].dropna().copy()
    if max_rows and len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    sd = d[dose_col].std()
    d["dose_z"] = (d[dose_col] - d[dose_col].mean()) / (sd if sd > 0 else 1.0)
    d["y"] = d[target].astype(float)

    vc = {g: f"0 + C({g})" for g in groups}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula("y ~ dose_z", vc, d)
        res = model.fit_vb()

    names = list(res.model.exog_names)
    beta_idx = names.index("dose_z")
    # vcp_mean holds one log-SD per grouping factor, in the order of `groups`
    var_components = {g: float(np.exp(res.vcp_mean[i])) for i, g in enumerate(groups)}

    return GLMMFit(
        dose_beta=float(res.fe_mean[beta_idx]),
        dose_beta_sd=float(res.fe_sd[beta_idx]),
        intercept=float(res.fe_mean[names.index("Intercept")]),
        var_components=var_components,
        n=len(d),
        groups=tuple(groups),
    )
