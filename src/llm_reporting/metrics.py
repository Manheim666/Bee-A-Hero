"""Loads cv_ml_report.json, computes an aggregate summary across all
flowers, and validates logical bounds before facts are sent to the LLM.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SPECIES_NAMES: list[str] = ["honeybee", "bee", "fly", "beetle", "bug", "butterfly"]


def _dominant_species(species_totals: dict[str, int]) -> tuple[list[str], int]:
    """Returns (dominant_species_list, max_count). Ties are ALL returned,
    never arbitrarily narrowed to one species."""
    if not species_totals or max(species_totals.values()) == 0:
        return [], 0
    max_count = max(species_totals.values())
    return [s for s, c in species_totals.items() if c == max_count], max_count


def _compute_aggregates(facts: dict) -> dict:
    """Computes summary totals/means across every flower in `facts`.

    Args:
        facts: The raw facts dict (as loaded from cv_ml_report.json).

    Returns:
        A dict of aggregate statistics. Empty/zero-safe for an empty
        flowers list.
    """
    flowers = facts.get("flowers", [])
    if not flowers:
        logger.warning("_compute_aggregates called with an empty flowers list")

    species_totals = {s: 0 for s in SPECIES_NAMES}
    total_landings = 0
    total_real_landings = 0
    total_pollinator_visits = 0
    total_non_pollinator_visits = 0
    total_landing_s = 0.0
    pollination_scores: list[float] = []
    fruit_set_probs: list[float] = []
    yields: list[float] = []

    for f in flowers:
        cv = f.get("cv_facts", {})
        total_landings += cv.get("n_landings", 0)
        total_real_landings += cv.get("n_real_landings", 0)
        total_pollinator_visits += cv.get("pollinator_visits", 0)
        total_non_pollinator_visits += cv.get("non_pollinator_visits", 0)
        total_landing_s += cv.get("total_landing_s", 0.0)
        pollination_scores.append(cv.get("pollination_score", 0.0))

        for species in SPECIES_NAMES:
            species_totals[species] += cv.get(f"n_{species}", 0)

        ml = f.get("ml_estimate")
        if ml:
            fruit_set_probs.append(ml["fruit_set_probability"])
            yields.append(ml["yield_kg_per_flower_estimate"])

    n = len(flowers)
    active = sum(1 for f in flowers if f.get("cv_facts", {}).get("n_landings", 0) > 0)
    dominant_pollinators, dominant_count = _dominant_species(species_totals)

    aggregate = {
        "flower_count": n,
        "active_flowers": active,
        "flowers_detected": n,
        "flowers_analyzed": active,
        "total_landings": total_landings,
        "total_real_landings": total_real_landings,
        "total_pollinator_visits": total_pollinator_visits,
        "total_non_pollinator_visits": total_non_pollinator_visits,
        "total_landing_s": round(total_landing_s, 2),
        "species_totals": species_totals,
        "dominant_pollinators": dominant_pollinators,
        "dominant_pollinator_visits": dominant_count,
        "mean_pollination_score": round(sum(pollination_scores) / n, 2) if n > 0 else 0.0,
        "max_pollination_score": round(max(pollination_scores), 2) if pollination_scores else 0.0,
    }

    if fruit_set_probs:
        mean_prob = round(sum(fruit_set_probs) / len(fruit_set_probs), 4)
        aggregate["ml_estimate_summary"] = {
            "assumed_crop_unverified": True,
            "mean_fruit_set_probability": mean_prob,
            "mean_fruit_set_probability_pct": round(mean_prob * 100, 1),
            "total_estimated_yield_kg": round(sum(yields), 4),
            "flowers_with_estimate": len(fruit_set_probs),
        }

    logger.info(
        "Computed aggregate: %d flowers (%d active), %d pollinator / %d non-pollinator visits, "
        "dominant=%s",
        n, active, total_pollinator_visits, total_non_pollinator_visits, dominant_pollinators,
    )
    return aggregate


def to_facts_dict(report_json_path: str | Path) -> dict:
    """Loads a cv_ml_report.json file and attaches its computed aggregate.

    Args:
        report_json_path: Path to the JSON file produced by
            cv_to_json_report.py.

    Returns:
        The loaded facts dict, with an additional "aggregate" key and a
        "report_version" key (defaulted to "1.0" if absent, for
        backward compatibility with older JSON files).

    Raises:
        FileNotFoundError: If `report_json_path` does not exist.
    """
    path = Path(report_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Report JSON not found: {path}")

    with open(path) as f:
        facts = json.load(f)

    facts.setdefault("report_version", "1.0")
    facts["aggregate"] = _compute_aggregates(facts)
    return facts


def validate_bounds(facts: dict) -> list[str]:
    """Catches logically impossible values in CV pipeline output before
    they ever reach the LLM.

    Args:
        facts: The facts dict (as returned by `to_facts_dict`).

    Returns:
        A list of human-readable issue strings. Empty list means all
        checks passed.
    """
    issues: list[str] = []
    flowers = facts.get("flowers", [])

    if not flowers:
        issues.append("flowers list is empty - nothing to report on")
        return issues

    for f in flowers:
        cv = f.get("cv_facts", {})
        flower_id = cv.get("flower_id", "unknown")

        n_landings = cv.get("n_landings", 0)
        n_real_landings = cv.get("n_real_landings", 0)
        if n_real_landings > n_landings:
            issues.append(
                f"{flower_id}: n_real_landings ({n_real_landings}) "
                f"exceeds n_landings ({n_landings})"
            )

        pollinator_sum = cv.get("pollinator_visits", 0) + cv.get("non_pollinator_visits", 0)
        if pollinator_sum != n_real_landings:
            issues.append(
                f"{flower_id}: pollinator_visits + non_pollinator_visits "
                f"({pollinator_sum}) != n_real_landings ({n_real_landings})"
            )

        for field_name in ("pollination_score", "mean_landing_s", "total_landing_s"):
            value = cv.get(field_name, 0)
            if value < 0:
                issues.append(f"{flower_id}: {field_name} is negative ({value})")

    declared_count = facts.get("flower_count")
    if declared_count != len(flowers):
        issues.append(f"flower_count ({declared_count}) != len(flowers) ({len(flowers)})")

    if issues:
        logger.warning("validate_bounds found %d issue(s): %s", len(issues), issues)
    else:
        logger.info("validate_bounds: all %d flower(s) passed", len(flowers))

    return issues
