"""Prediction intervals and Monte-Carlo yield propagation (Sections 3.2-3.3).

Turns a :class:`~src.ml_models.dose_response.DoseResponseFit` into interval-carrying
numbers for the report layer. Nothing here returns a bare point estimate: a fruit-set
probability or a yield figure without an interval is, for a small pomegranate model, a
misleading claim of precision (research doc Section 3.5).

Two quantities:

  * **fruit-set prediction interval** at a given dose — propagates *parameter*
    uncertainty (the bootstrap draws of the curve) through the curve.
  * **orchard yield** ``Yield = N_flowers * FruitSet(V) * mean_fruit_mass`` — Monte-Carlo
    over the same parameter draws, optionally adding binomial sampling of individual
    flowers, so a +/-5-point uncertainty on fruit set surfaces as a kilogram band.
"""
from __future__ import annotations

import numpy as np

from src.ml_models.dose_response import DoseResponseFit, fruit_set_curve

SEED = 42


# --------------------------------------------------------------------------- #
# Fruit-set prediction interval
# --------------------------------------------------------------------------- #
def fruit_set_interval(fit: DoseResponseFit, V, *, level: float = 0.95
                       ) -> dict[str, np.ndarray]:
    """Fruit-set probability at dose(s) ``V`` with a parameter credible/bootstrap band.

    Pushes every bootstrap parameter draw through the curve and takes percentiles, so
    the band widens where the fit is uncertain. Returns arrays ``mean``, ``lo``, ``hi``
    aligned to ``V``.
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    a = (1.0 - level) / 2.0
    curves = np.array([fruit_set_curve(V, *theta) for theta in fit.boot])  # (n_boot, len V)
    return {
        "V": V,
        "mean": fit.predict(V),
        "lo": np.percentile(curves, 100 * a, axis=0),
        "hi": np.percentile(curves, 100 * (1 - a), axis=0),
    }


# --------------------------------------------------------------------------- #
# Yield propagation
# --------------------------------------------------------------------------- #
def propagate_yield(fit: DoseResponseFit, dose, n_flowers: int, mean_fruit_mass_kg: float,
                    *, add_binomial: bool = True, level: float = 0.95,
                    seed: int = SEED) -> dict:
    """Monte-Carlo orchard yield (kg) at a representative ``dose`` with an interval.

    For each bootstrap parameter draw, evaluate ``FruitSet(dose)``; optionally draw the
    number of setting flowers as ``Binomial(n_flowers, p)`` to add prediction (not just
    parameter) uncertainty; multiply by ``mean_fruit_mass_kg``. The 2.5/97.5 percentiles
    of the resulting sample are the yield interval (Section 3.3).

    ``dose`` may be a scalar (e.g. the orchard's mean dose) — a single representative
    fruit-set probability drives the whole-orchard figure.
    """
    rng = np.random.default_rng(seed)
    dose = float(np.mean(dose)) if np.ndim(dose) else float(dose)
    a = (1.0 - level) / 2.0

    p_draws = np.clip(np.array([fruit_set_curve(dose, *theta) for theta in fit.boot]),
                      0.0, 1.0)
    if add_binomial:
        n_set = rng.binomial(n_flowers, p_draws)
        set_rate = n_set / n_flowers
    else:
        set_rate = p_draws
    yield_kg = set_rate * n_flowers * mean_fruit_mass_kg

    return {
        "dose": dose,
        "n_flowers": n_flowers,
        "mean_fruit_mass_kg": mean_fruit_mass_kg,
        "fruit_set_mean": float(fit.predict(dose)),
        "fruit_set_ci95": [float(np.percentile(p_draws, 100 * a)),
                           float(np.percentile(p_draws, 100 * (1 - a)))],
        "yield_kg_mean": float(np.mean(yield_kg)),
        "yield_kg_ci95": [float(np.percentile(yield_kg, 100 * a)),
                          float(np.percentile(yield_kg, 100 * (1 - a)))],
    }
