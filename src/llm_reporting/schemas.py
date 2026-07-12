"""Pydantic schemas for the LLM reporting module's input/output contract.

Validation here is intentionally strict: this is the last structural gate
before data reaches the LLM, so silently accepting malformed CV/ML output
(NaN, Infinity, negative counts, unknown species) would let bad data
propagate into a user-facing report.
"""

import math

from pydantic import BaseModel, field_validator, model_validator

VALID_SPECIES: set[str] = {"honeybee", "bee", "fly", "beetle", "bug", "butterfly"}


def _reject_nan_inf(value: float, field_name: str) -> float:
    if math.isnan(value):
        raise ValueError(f"{field_name} is NaN, which is not a valid measurement")
    if math.isinf(value):
        raise ValueError(f"{field_name} is Infinity, which is not a valid measurement")
    return value


class LandingDetail(BaseModel):
    insect_type: str
    is_pollinator: bool
    landing_s: float
    is_real_landing: bool
    pollination_weight: float
    score_contribution: float

    @field_validator("insect_type")
    @classmethod
    def _species_must_be_known(cls, v: str) -> str:
        if v not in VALID_SPECIES:
            raise ValueError(f"Unknown insect_type '{v}' - expected one of {sorted(VALID_SPECIES)}")
        return v

    @field_validator("landing_s", "pollination_weight", "score_contribution")
    @classmethod
    def _no_nan_inf(cls, v: float, info) -> float:
        return _reject_nan_inf(v, info.field_name)

    @field_validator("landing_s", "pollination_weight", "score_contribution")
    @classmethod
    def _no_negative(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative, got {v}")
        return v


class MLEstimate(BaseModel):
    assumed_crop_unverified: bool = True
    assumed_crop: str
    V: float
    fruit_set_probability: float
    yield_kg_per_flower_estimate: float
    is_measured: bool = False
    curve_source: str
    F0_fit: float
    Fmax_fit: float
    k_fit: float

    @field_validator("V", "fruit_set_probability", "yield_kg_per_flower_estimate",
                      "F0_fit", "Fmax_fit", "k_fit")
    @classmethod
    def _no_nan_inf(cls, v: float, info) -> float:
        return _reject_nan_inf(v, info.field_name)

    @field_validator("fruit_set_probability", "F0_fit", "Fmax_fit")
    @classmethod
    def _must_be_valid_probability(cls, v: float, info) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be in [0, 1], got {v}")
        return v

    @field_validator("V", "yield_kg_per_flower_estimate")
    @classmethod
    def _must_be_non_negative(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative, got {v}")
        return v


class CVFacts(BaseModel):
    video: str
    flower_id: str
    n_landings: int
    n_real_landings: int
    pollinator_visits: int
    non_pollinator_visits: int
    n_honeybee: int = 0
    n_bee: int = 0
    n_fly: int = 0
    n_beetle: int = 0
    n_bug: int = 0
    n_butterfly: int = 0
    total_landing_s: float
    mean_landing_s: float
    pollination_score: float
    video_observation_span_s: float = 0.0
    dominant_pollinators: list[str] = []
    dominant_pollinator_visits: int = 0
    landings_detail: list[LandingDetail] = []

    @field_validator("video", "flower_id")
    @classmethod
    def _not_empty(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    @field_validator(
        "n_landings", "n_real_landings", "pollinator_visits", "non_pollinator_visits",
        "n_honeybee", "n_bee", "n_fly", "n_beetle", "n_bug", "n_butterfly",
    )
    @classmethod
    def _counts_non_negative(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative, got {v}")
        return v

    @field_validator("total_landing_s", "mean_landing_s", "pollination_score")
    @classmethod
    def _floats_no_nan_inf_or_negative(cls, v: float, info) -> float:
        _reject_nan_inf(v, info.field_name)
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative, got {v}")
        return v

    @model_validator(mode="after")
    def _real_landings_within_total(self) -> "CVFacts":
        if self.n_real_landings > self.n_landings:
            raise ValueError(
                f"n_real_landings ({self.n_real_landings}) exceeds "
                f"n_landings ({self.n_landings}) for flower {self.flower_id}"
            )
        return self


class FlowerReport(BaseModel):
    cv_facts: CVFacts
    ml_estimate: MLEstimate | None = None


class ReportInput(BaseModel):
    report_version: str = "1.0"
    crop_note: str
    pollinator_classification_note: str
    source: str
    flower_count: int
    flowers: list[FlowerReport]

    @field_validator("flowers")
    @classmethod
    def _flowers_not_empty(cls, v: list[FlowerReport]) -> list[FlowerReport]:
        if len(v) == 0:
            raise ValueError("flowers list cannot be empty - no data to report on")
        return v

    @model_validator(mode="after")
    def _flower_count_matches(self) -> "ReportInput":
        if self.flower_count != len(self.flowers):
            raise ValueError(
                f"flower_count ({self.flower_count}) does not match "
                f"len(flowers) ({len(self.flowers)})"
            )
        return self


class ReportOutput(BaseModel):
    report_text: str
    fidelity_passed: bool
    flagged_numbers: list[str] = []
