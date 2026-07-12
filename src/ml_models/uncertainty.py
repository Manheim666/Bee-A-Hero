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


def orchard_yield(fit: DoseResponseFit, doses, base_fruit_mass_kg: float, *,
                  size_gain: float = 0.0, add_binomial: bool = True,
                  level: float = 0.95, seed: int = SEED) -> dict:
    """Per-flower orchard yield with optional fruit-size coupling and a 95% interval.

    Two upgrades over :func:`propagate_yield` (which uses a single representative dose):

    1. **Per-flower aggregation.** Yield is summed over *each* flower's own dose,
       ``yield = sum_i FruitSet(V_i) * mass_i``, rather than ``N * FruitSet(mean_dose)``.
       For a curved response these differ (Jensen's inequality): using the mean dose
       systematically mis-states yield, so the per-flower sum is the correct figure.
    2. **Fruit-size coupling.** Better-pollinated flowers set *larger* fruit (research doc,
       Wetzstein et al. 2013), so each fruit's mass scales with its pollination fraction:

           mass_i = base_fruit_mass_kg * (1 + size_gain * frac_i),
           frac_i = clip((FruitSet(V_i) - F0) / (Fmax - F0), 0, 1)

       ``size_gain`` is the fractional size boost of a fully-pollinated fruit over a
       baseline (barely-set) one; ``size_gain=0`` reproduces constant-mass yield.

    Uncertainty is Monte-Carlo over the bootstrap parameter draws (and, if ``add_binomial``,
    a Bernoulli set/no-set draw per flower).

    Parameters
    ----------
    doses
        Per-flower effective dose ``V`` (one value per flower in the orchard/sample).
    base_fruit_mass_kg
        Fruit mass of a baseline (minimally-pollinated) fruit.
    """
    V = np.asarray(doses, dtype=float)
    n = V.size
    rng = np.random.default_rng(seed)
    a = (1.0 - level) / 2.0
    span = max(fit.Fmax - fit.F0, 1e-6)

    yields = np.empty(len(fit.boot))
    for b, theta in enumerate(fit.boot):
        p = np.clip(fruit_set_curve(V, *theta), 0.0, 1.0)
        frac = np.clip((p - theta[0]) / max(theta[1] - theta[0], 1e-6), 0.0, 1.0)
        mass = base_fruit_mass_kg * (1.0 + size_gain * frac)
        set_flag = rng.random(n) < p if add_binomial else p          # Bernoulli or expected
        yields[b] = float(np.sum(set_flag * mass))

    p_point = np.clip(fit.predict(V), 0.0, 1.0)
    frac_point = np.clip((p_point - fit.F0) / span, 0.0, 1.0)
    mass_point = base_fruit_mass_kg * (1.0 + size_gain * frac_point)
    yield_point = float(np.sum(p_point * mass_point))
    # naive comparison: N * FruitSet(mean dose) * base mass (constant mass, mean dose)
    naive = float(n * fit.predict(float(V.mean())) * base_fruit_mass_kg)

    return {
        "n_flowers": n,
        "base_fruit_mass_kg": base_fruit_mass_kg,
        "size_gain": size_gain,
        "mean_fruit_set": float(p_point.mean()),
        "yield_kg_mean": yield_point,
        "yield_kg_ci95": [float(np.percentile(yields, 100 * a)),
                          float(np.percentile(yields, 100 * (1 - a)))],
        "yield_kg_naive_pmean": naive,           # the old N*p(mean)*mass figure, for contrast
        "per_flower_vs_naive_delta": round(yield_point - naive, 2),
    }
