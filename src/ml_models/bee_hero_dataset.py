"""Modeling-stage data loader: CV visit records -> per-flower dose frame.

This is the glue between the computer-vision tracker and the dose-response model
(``dose_response.py`` / ``uncertainty.py``). It reads a per-visit / per-landing
table — either the synthetic ``visits.csv`` from ``generate_bee_data.py`` or the
real tracker export ``ALL_landings.csv`` from ``cv_engine/video_detect.py`` — applies
the **three qualifying gates** (dwell, velocity, spatial overlap), and aggregates the
qualifying visits into the per-flower **effective dose** ``V`` the curve consumes.

The gate thresholds and the dwell-saturation constant are imported from
``generate_bee_data`` so the labels and the modeling code share one definition and
cannot drift apart.

Tracker contract (fixed input, do not change upstream):
  * synthetic ``visits.csv`` : flower_id, species, dwell_seconds, velocity, fraction_on, ...
  * real ``ALL_landings.csv``: flower_id, insect_type, landing_s, is_real_landing,
                               pollination_weight, conf_mean, ...  (no velocity/fraction_on)

The velocity and fraction-on gates are only applied when the source table carries
those columns; the real tracker does not yet emit them (a documented CV refinement),
so on real data only the dwell gate filters and the others pass through — recorded
explicitly in the returned ``applied_gates`` rather than silently assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_models.generate_bee_data import DWELL_MIN, VEL_MAX, FRAC_MIN, TAU

# Canonical per-visit schema the modeling stage works in. ``velocity`` and
# ``fraction_on`` are optional (absent in the current real tracker export).
_CANON = ("flower_id", "species", "dwell_seconds", "velocity", "fraction_on",
          "weight", "confidence")


# --------------------------------------------------------------------------- #
# Loading / schema normalisation
# --------------------------------------------------------------------------- #
def load_visits(path: str | Path) -> pd.DataFrame:
    """Read a visit/landing CSV and normalise it to the canonical per-visit schema.

    Accepts either the synthetic ``visits.csv`` or the real tracker ``ALL_landings.csv``
    and maps their columns onto ``_CANON``. Columns the source lacks are added as
    ``NaN`` so downstream code can test for them uniformly.
    """
    df = pd.read_csv(path)
    cols = set(df.columns)

    if "dwell_seconds" in cols:                      # synthetic visits.csv
        # synthetic flower_id is globally unique, so it is already the flower key
        out = pd.DataFrame({
            "flower_id": df["flower_id"],
            "species": df.get("species"),
            "dwell_seconds": df["dwell_seconds"],
            "velocity": df.get("velocity"),
            "fraction_on": df.get("fraction_on"),
            "weight": df.get("pollination_prob", 1.0),
            "confidence": df.get("confidence", 1.0),
        })
    elif "landing_s" in cols:                         # real tracker ALL_landings.csv
        # tracker flower IDs are per-clip only (no cross-time identity, research doc
        # Section 16), so 'flower_1' in two videos are different flowers -> compose the
        # key with the video id to keep them distinct when aggregating.
        video = df["video"].astype(str) if "video" in cols else ""
        flower_key = video + "::" + df["flower_id"].astype(str) if "video" in cols \
            else df["flower_id"].astype(str)
        out = pd.DataFrame({
            "flower_id": flower_key,
            "species": df.get("insect_type"),
            "dwell_seconds": df["landing_s"],
            "velocity": np.nan,                       # not emitted by video_detect.py yet
            "fraction_on": np.nan,                    # not emitted by video_detect.py yet
            "weight": df.get("pollination_weight", 1.0),
            "confidence": df.get("conf_mean", 1.0),
        })
    else:
        raise ValueError(
            f"{path}: unrecognised visit schema (need 'dwell_seconds' or 'landing_s'); "
            f"got columns {sorted(cols)}")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The three qualifying gates  (dwell, velocity, fraction-on)
# --------------------------------------------------------------------------- #
def apply_qualifying_gates(visits: pd.DataFrame, *, dwell_min: float = DWELL_MIN,
                           vel_max: float = VEL_MAX, frac_min: float = FRAC_MIN
                           ) -> tuple[pd.DataFrame, list[str]]:
    """Flag each visit as qualifying using up to three gates, matching the generator.

    A visit qualifies as a real, pollen-relevant landing when it clears every gate
    whose measurement is available:

      * **dwell**      ``dwell_seconds >= dwell_min``   — a fly-through is not a visit,
      * **velocity**   ``velocity      <= vel_max``     — slow = settled, fast = mis-track,
      * **fraction-on**``fraction_on   >= frac_min``    — enough of the insect box on the flower.

    Gates whose source column is entirely missing (real tracker output lacks velocity
    and fraction_on) are treated as passed and are **not** listed in the returned
    ``applied_gates``, so a caller can see exactly which gates actually filtered.

    Returns ``(visits_with_gate_columns, applied_gates)``. The added boolean columns are
    ``gate_dwell``, ``gate_velocity``, ``gate_fraction`` and ``qualifying_visit``.
    """
    out = visits.copy()
    applied: list[str] = []

    def _has(col: str) -> bool:
        return col in out.columns and out[col].notna().any()

    if _has("dwell_seconds"):
        out["gate_dwell"] = out["dwell_seconds"] >= dwell_min
        applied.append("dwell")
    else:
        out["gate_dwell"] = True

    if _has("velocity"):
        out["gate_velocity"] = out["velocity"] <= vel_max
        applied.append("velocity")
    else:
        out["gate_velocity"] = True

    if _has("fraction_on"):
        out["gate_fraction"] = out["fraction_on"] >= frac_min
        applied.append("fraction_on")
    else:
        out["gate_fraction"] = True

    out["qualifying_visit"] = (out["gate_dwell"] & out["gate_velocity"]
                               & out["gate_fraction"])
    return out, applied


def visit_effectiveness(dwell_seconds, weight, *, tau: float = TAU):
    """Per-visit pollination effectiveness = ``weight * (1 - exp(-dwell / tau))``.

    The dwell-saturation factor is the same one ``generate_bee_data`` uses to build the
    latent dose; ``weight`` carries the species/pollinator weighting (e.g. honeybees
    count more). Returned per-visit so it can be summed into a per-flower dose.
    """
    dwell = np.asarray(dwell_seconds, dtype=float)
    w = np.asarray(weight, dtype=float)
    return w * (1.0 - np.exp(-dwell / tau))


# --------------------------------------------------------------------------- #
# Per-flower aggregation -> effective dose V
# --------------------------------------------------------------------------- #
def aggregate_flowers(visits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate gated visits to one row per flower with the effective dose ``V``.

    ``V`` sums :func:`visit_effectiveness` over the **qualifying** visits only, so the
    three gates directly determine the dose. Also returns simple, CV-observable
    features (visit counts, mean dwell, mean detector confidence) for diagnostics.

    Input must already carry the gate columns (call :func:`apply_qualifying_gates`
    first); this function does not re-gate.
    """
    if "qualifying_visit" not in visits.columns:
        raise ValueError("call apply_qualifying_gates() before aggregate_flowers()")

    v = visits.copy()
    v["eff"] = visit_effectiveness(v["dwell_seconds"], v["weight"])
    q = v[v["qualifying_visit"]]

    dose = q.groupby("flower_id")["eff"].sum().rename("V")
    n_q = q.groupby("flower_id").size().rename("n_qualifying_visits")
    n_all = v.groupby("flower_id").size().rename("n_visits")
    mean_dwell = v.groupby("flower_id")["dwell_seconds"].mean().rename("mean_dwell_s")
    mean_conf = v.groupby("flower_id")["confidence"].mean().rename("mean_confidence")

    out = (pd.concat([n_all, n_q, dose, mean_dwell, mean_conf], axis=1)
           .reset_index())
    # flowers whose every visit was filtered out get zero dose / zero qualifying
    out[["n_qualifying_visits", "V"]] = out[["n_qualifying_visits", "V"]].fillna(0.0)
    out["n_qualifying_visits"] = out["n_qualifying_visits"].astype(int)
    return out


def effective_dose_from_aggregates(df: pd.DataFrame, *, tau: float = TAU) -> pd.Series:
    """Effective dose ``V`` for an already-per-flower modeling frame (``dataset_training_v6``).

    The processed training frame is aggregated one row per flower and carries the *result*
    of the three qualifying gates (``n_qualifying_visits`` and per-species qualifying counts
    ``nq_*``) plus the constant per-species weights ``w_*`` — but not the raw dose ``V``,
    which is dropped as a generator-only column. This reconstructs the same
    weight x dwell-saturation dose used per visit in :func:`visit_effectiveness`:

        V = ( sum_species  w_species * nq_species ) * (1 - exp(-mean_dwell_s / tau)) * confidence

    Species terms are paired by suffix (``nq_honeybee`` with ``w_honeybee`` etc.); a weight
    column is treated as 1.0 if absent. ``mean_dwell_s`` and ``confidence`` are imputed with
    their medians when missing.
    """
    nq_cols = [c for c in df.columns if c.startswith("nq_")]
    if not nq_cols:
        raise ValueError("no 'nq_*' species columns found; not a v6-style aggregate frame")
    species = pd.Series(0.0, index=df.index)
    for nq in nq_cols:
        w = "w_" + nq[len("nq_"):]
        species = species + df[nq] * (df[w] if w in df.columns else 1.0)

    dwell = df["mean_dwell_s"].fillna(df["mean_dwell_s"].median()) if "mean_dwell_s" in df \
        else 0.0
    conf = df["confidence"].fillna(1.0).clip(0.0, 1.0) if "confidence" in df else 1.0
    return (species * (1.0 - np.exp(-dwell / tau)) * conf).astype(float)


def flower_dose_frame(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Convenience end-to-end: load a visit CSV -> gates -> per-flower dose frame.

    Returns ``(flowers, applied_gates)``. This is the single call the orchestrator
    uses to turn raw tracker output into the modeling frame.
    """
    visits = load_visits(path)
    gated, applied = apply_qualifying_gates(visits)
    return aggregate_flowers(gated), applied


# --------------------------------------------------------------------------- #
# Real fruit-set labels — the integration point for field-collected data
# --------------------------------------------------------------------------- #
# The response variable (did a flower set fruit?), the flower-type label, and the
# cross-time flower identity that links a visit count to that same flower's later fruit
# are *field data / an upstream CV task* (research doc Sections 9, 16) — they cannot be
# produced here. The two functions below are the single, documented place where that data
# joins the modeling frame once collected, so nothing downstream changes when it arrives.

#: minimum schema a real fruit-set label file must provide.
FRUIT_SET_LABEL_SCHEMA = ("flower_uid", "fruit_set")


def load_fruit_set_labels(path: str | Path) -> pd.DataFrame:
    """Load a per-flower fruit-set label file and validate its schema.

    Expected columns:
      * ``flower_uid``   — a **global, cross-time** flower identity (NOT the per-clip
        ``flower_id``); linking the two is the open CV task of research-doc Section 16.
      * ``fruit_set``    — 0/1, scored weeks after bloom (the response variable ``y``).
      * ``flower_type``  — optional, ``bisexual`` vs ``male``; functionally male flowers
        cannot set fruit (Section 9) and should be excluded from the dose-response fit.

    Raises if the required columns are absent. This is intentionally strict: it is the guard
    that stops synthetic stand-ins from being silently mistaken for real labels.
    """
    df = pd.read_csv(path)
    missing = set(FRUIT_SET_LABEL_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: fruit-set label file missing columns {sorted(missing)}; "
                         f"required schema is {FRUIT_SET_LABEL_SCHEMA}")
    return df


def join_fruit_set_labels(flowers: pd.DataFrame, labels: pd.DataFrame, *,
                          key: str = "flower_uid", bisexual_only: bool = True) -> pd.DataFrame:
    """Join real fruit-set labels onto a per-flower dose frame, ready for the fit.

    ``flowers`` must carry the same global ``key`` as the labels — i.e. the cross-time
    identity must already have been resolved upstream (Section 16); this function does not
    invent it. Flowers without a label are dropped (they were not scored). When
    ``bisexual_only`` and a ``flower_type`` column exist, functionally male flowers are
    removed so only fruit-capable flowers enter the dose-response (Section 9).
    """
    if key not in flowers.columns:
        raise ValueError(f"dose frame has no '{key}' column: cross-time flower identity "
                         f"(research doc Section 16) must be resolved before joining labels")
    out = flowers.merge(labels, on=key, how="inner")
    if bisexual_only and "flower_type" in out.columns:
        out = out[out["flower_type"].str.lower().eq("bisexual")]
    return out.reset_index(drop=True)
