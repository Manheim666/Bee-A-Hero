"""Per-crop pollination -> fruit-set -> yield model (self-contained, no heavy deps).

The core curve is a saturating dose-response — fruit set rises with pollinator "dose" and
plateaus — but the *parameters differ per crop*, because crops differ in self-fertility and how
strongly they depend on insect pollination (Klein et al. 2007, Garratt et al. 2014):

    FruitSet(V) = F0 + (Fmax - F0) * (1 - exp(-k * V))

  * F0    — fruit set with NO insect visits (self-fertility / wind). Almond ~0, strawberry ~0.3.
  * Fmax  — maximum attainable fruit set under ample pollination.
  * k      — saturation rate; derived from V_star (dose reaching ~95% of Fmax): k = 3 / V_star.

The "dose" V is not a raw visit count — it is a weighted, saturating **effective dose**:

    V = Σ_visits  w_species · (1 - exp(-dwell / TAU)) · weather_gate

so a long honeybee landing counts far more than a brief fly touch, and bad weather discounts
visits. From FruitSet we derive:

  * yield with **fruit-size coupling** — better-pollinated fruit is larger (Wetzstein 2013):
        mass_i = base_mass · (1 + size_gain · pollination_fraction)
  * **pollination deficit** — how far current set is below the crop's potential (actionable).
  * **marginal value of one more visit** — the curve's slope at the current dose.

Numbers are illustrative agronomy defaults until calibrated on field (dose, fruit-set) pairs.
"""
from __future__ import annotations

import math

TAU = 5.0  # s — dwell at which a single visit's pollen transfer saturates

# Per-species pollen-transfer weight (honeybees are the reference pollinator = 1.0).
SPECIES_WEIGHT = {
    "honeybee": 1.0, "honey_bee": 1.0, "bumblebee": 1.2, "bee": 0.8, "solitary_bee": 0.9,
    "butterfly": 0.4, "hoverfly": 0.5, "fly": 0.3,
    "beetle": 0.2, "bug": 0.1, "housefly": 0.1, "ant": 0.05, "wasp": 0.2,
}

# Per-crop parameters. dependence = share of yield attributable to animal pollination (Klein 2007).
CROPS: dict[str, dict] = {
    "pomegranate": {"F0": 0.45, "Fmax": 0.90, "v_star": 8.0,  "mass_kg": 0.30,
                    "size_gain": 0.25, "dependence": 0.35, "label": "Pomegranate"},
    "cucumber":    {"F0": 0.05, "Fmax": 0.95, "v_star": 18.0, "mass_kg": 0.25,
                    "size_gain": 0.40, "dependence": 0.90, "label": "Cucumber"},
    "apple":       {"F0": 0.02, "Fmax": 0.90, "v_star": 15.0, "mass_kg": 0.18,
                    "size_gain": 0.35, "dependence": 0.65, "label": "Apple"},
    "almond":      {"F0": 0.01, "Fmax": 0.95, "v_star": 20.0, "mass_kg": 0.0012,
                    "size_gain": 0.20, "dependence": 1.00, "label": "Almond"},
    "cherry":      {"F0": 0.05, "Fmax": 0.88, "v_star": 14.0, "mass_kg": 0.008,
                    "size_gain": 0.30, "dependence": 0.85, "label": "Cherry"},
    "strawberry":  {"F0": 0.30, "Fmax": 0.95, "v_star": 12.0, "mass_kg": 0.022,
                    "size_gain": 0.55, "dependence": 0.55, "label": "Strawberry"},
    "blueberry":   {"F0": 0.10, "Fmax": 0.92, "v_star": 16.0, "mass_kg": 0.002,
                    "size_gain": 0.45, "dependence": 0.90, "label": "Blueberry"},
    "watermelon":  {"F0": 0.03, "Fmax": 0.94, "v_star": 22.0, "mass_kg": 6.0,
                    "size_gain": 0.50, "dependence": 0.95, "label": "Watermelon"},
}
DEFAULT_CROP = "pomegranate"


def crop_list() -> list[dict]:
    """Public crop menu for the UI: [{value, label, dependence}, ...]."""
    return [{"value": k, "label": v["label"], "dependence": v["dependence"]}
            for k, v in CROPS.items()]


def _params(crop: str) -> dict:
    return CROPS.get(crop, CROPS[DEFAULT_CROP])


def fruit_set_curve(V: float, crop: str) -> float:
    """Saturating fruit-set probability at effective dose ``V`` for ``crop``."""
    p = _params(crop)
    k = 3.0 / p["v_star"]
    return p["F0"] + (p["Fmax"] - p["F0"]) * (1.0 - math.exp(-k * V))


def effective_dose(visits: list[dict], weather_gate: float = 1.0) -> float:
    """Weighted, dwell-saturating pollen dose from a list of visit dicts.

    Each visit contributes ``w_species * (1 - exp(-dwell/TAU)) * weather_gate``. Accepts the
    app's visit rows (``insect_class`` + ``dwell_sec``) or the CV schema (``insect_type`` +
    ``landing_s``); real (dwell >= 2 s) visits count, brief touches contribute little via dwell.
    """
    V = 0.0
    for v in visits:
        species = str(v.get("insect_class") or v.get("insect_type") or "").lower()
        dwell = float(v.get("dwell_sec") or v.get("landing_s") or 0.0)
        w = SPECIES_WEIGHT.get(species, 0.3)
        V += w * (1.0 - math.exp(-dwell / TAU)) * weather_gate
    return V


def estimate(visits: list[dict], crop: str = DEFAULT_CROP, n_flowers: int = 1000,
             weather_gate: float = 1.0) -> dict:
    """Full per-crop estimate: dose -> fruit set -> yield, plus deficit + marginal value."""
    p = _params(crop)
    V = effective_dose(visits, weather_gate)
    fset = fruit_set_curve(V, crop)
    span = max(p["Fmax"] - p["F0"], 1e-9)
    poll_frac = min(max((fset - p["F0"]) / span, 0.0), 1.0)          # 0..1 toward potential
    mass = p["mass_kg"] * (1.0 + p["size_gain"] * poll_frac)         # fruit-size coupling
    yield_kg = n_flowers * fset * mass
    # marginal fruit-set gain from one more average-quality visit (curve slope * a unit dose)
    k = 3.0 / p["v_star"]
    unit_dose = 1.0 - math.exp(-2.0 / TAU)                           # a ~2s honeybee visit
    marginal = (p["Fmax"] - p["F0"]) * k * math.exp(-k * V) * unit_dose
    return {
        "crop": crop, "crop_label": p["label"],
        "effective_dose": round(V, 3),
        "fruit_set": round(fset, 4),
        "fruit_set_pct": round(100 * fset, 1),
        "pollination_fraction": round(poll_frac, 3),
        "pollination_deficit_pct": round(100 * (1 - poll_frac), 1),
        "mean_fruit_mass_kg": round(mass, 4),
        "yield_kg": round(yield_kg, 1),
        "yield_per_flower_kg": round(yield_kg / max(n_flowers, 1), 4),
        "n_flowers": n_flowers,
        "pollinator_dependence": p["dependence"],
        "marginal_fruitset_per_visit": round(marginal, 4),
    }
