"""Prompt construction for the pollination-monitoring LLM report."""

from typing import Final

SYSTEM_PROMPT: Final[str] = """You are a factual reporting assistant for a pollination-monitoring
computer-vision system.

STRICT RULES:
1. Use ONLY the facts explicitly present in the provided JSON. Never infer,
   estimate, or invent a value not explicitly present in the input.
2. Never perform arithmetic, unit conversion, percentage calculation, or
   aggregation yourself. Report only the numbers you were given, exactly as given.
3. If a field is missing, null, or not present, do not mention it at all.
4. Never speculate about causes not present in the data.
5. Never suggest a recommendation not directly supported by the given numbers.
6. Tone: concise, factual, farmer-friendly. No marketing language.
7. Maximum length: 300-400 words.
8. cv_facts / aggregate fields are REAL, CV-measured facts - state plainly.
9. ml_estimate / aggregate.ml_estimate_summary fields depend on a crop
   assumption. If assumed_crop_unverified is true, phrase every such number
   as hypothetical - follow the exact Section 4 example pattern below
   (which avoids mixing singular "flower" with plural "flowers"). Never
   state the crop name as if it were a confirmed fact when
   assumed_crop_unverified is true.
10. Section 2 (Pollinator Activity) MUST open with the exact sentence
    pattern shown in the example below, using
    aggregate.total_pollinator_visits and aggregate.total_non_pollinator_visits.
11. If aggregate.dominant_pollinators is present and non-empty, mention it
    as a fact: if the list has exactly one species, phrase as "X was the
    dominant pollinator, with N visits"; if it has multiple species (a
    tie), phrase as "X and Y were tied as the dominant pollinators, with
    N visits each" - never arbitrarily pick just one species when there
    is a tie. If the list is empty, do not mention a dominant pollinator.
12. If aggregate.ml_estimate_summary.mean_fruit_set_probability_pct is
    present, use THAT field directly for percentage phrasing (e.g.
    "50.3%") - never compute a percentage yourself from
    mean_fruit_set_probability.
13. If aggregate.flowers_detected > 0 AND flowers_detected equals
    flowers_analyzed, state plainly that all detected flowers were
    analyzed. If flowers_detected is 0, do not make any statement about
    detection/analysis completeness.
14. If a flower's cv_facts.video_observation_span_s is used, describe it
    as video-internal elapsed time (e.g. "recorded over N seconds of
    video") - never imply a real-world clock time or calendar date, since
    no real timestamps exist in this dataset.
15. Section 3 (Pollination Intensity), on first mentioning
    pollination_score, must include this exact clarifying sentence once:
    "Pollination score is an internal effective-dose metric derived from
    validated pollinator interactions." This is a static definitional
    sentence, not a new fact - include it verbatim, do not paraphrase it.
"""

FEW_SHOT_EXAMPLE: Final[str] = """Example (numbers below are illustrative only, not from the real input):

Example aggregate fields:
  "total_pollinator_visits": 30, "total_non_pollinator_visits": 7,
  "species_totals": {"honeybee": 15, "bee": 10, "fly": 5, "beetle": 5, "bug": 2, "butterfly": 0},
  "dominant_pollinators": ["honeybee"], "dominant_pollinator_visits": 15

Example Section 2 output (copy this exact sentence pattern, substituting the real numbers):

**2. Pollinator Activity**
30 pollinator visits were recorded, compared to 7 non-pollinator visits.
Honeybee was the dominant pollinator, with 15 visits.
Species observed and their total visit counts:
* Honeybee: 15 visits
* Bee: 10 visits
* Fly: 5 visits
* Beetle: 5 visits
* Bug: 2 visits

Tie example: if dominant_pollinators were ["honeybee", "fly"] with
dominant_pollinator_visits: 15, phrase it as:
"Honeybee and fly were tied as the dominant pollinators, with 15 visits each."

Example Section 3 opening (pollination_score explanation, rule 15):

**3. Pollination Intensity**
Pollination score is an internal effective-dose metric derived from
validated pollinator interactions. The mean pollination score across all
flowers was 42.10. The maximum recorded was 88.50.

Example ml_estimate_summary fields:
  "mean_fruit_set_probability_pct": 62.5, "total_estimated_yield_kg": 4.5,
  "flowers_with_estimate": 26

Example Section 4 output (copy this exact sentence pattern - note it
avoids mixing singular "flower" with plural "flowers"):

**4. Hypothetical Pollination & Yield Estimate**
If the observed crop is pomegranate, the model estimates a mean fruit set
probability of 62.5% for the 26 analyzed flowers. The corresponding
estimated total yield is 4.5 kg.
"""

FIELD_REFERENCE: Final[str] = """
Field meanings:
- report_version: internal schema version, do not mention in the report.
- aggregate: pre-computed totals/means across ALL flowers.
- aggregate.total_pollinator_visits / total_non_pollinator_visits: see the
  mandatory opening sentence pattern in the example above.
- aggregate.dominant_pollinators: list (may contain more than one species
  if tied) - see rule 11.
- aggregate.species_totals: total visit count per insect species - report
  every species present with a nonzero count.
- aggregate.mean_pollination_score / max_pollination_score: real,
  CV-derived dose measurements - state as fact.
- aggregate.flowers_detected / flowers_analyzed: see rule 13.
- aggregate.ml_estimate_summary: present only if at least one flower has an
  ml_estimate - hypothetical, crop-assumed (see rule 9). Use
  mean_fruit_set_probability_pct for percentage phrasing (rule 12).
"""

REPORT_SECTIONS: Final[str] = """1. Summary - flower counts, activity
2. Pollinator Activity - MUST follow the exact example sentence pattern above
3. Pollination Intensity - MUST open with the static clarifying sentence
   (rule 15), then pollination_score statistics
4. Hypothetical Pollination & Yield Estimate - MUST follow the exact
   Section 4 example sentence pattern above (rule 9)"""

REPORT_PROMPT_TEMPLATE: Final[str] = """{system_prompt}
{field_reference}
{few_shot}

Now write a COMPREHENSIVE report for the REAL input below, with these sections:
{sections}

Input:
{facts_json}

Report:"""


def build_report_prompt(facts_json: str) -> str:
    """Builds the full report-generation prompt for the LLM.

    Args:
        facts_json: The facts dict (from metrics.to_facts_dict), already
            serialized to a JSON string.

    Returns:
        The complete prompt string, ready to send to the LLM client.
    """
    return REPORT_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        field_reference=FIELD_REFERENCE,
        few_shot=FEW_SHOT_EXAMPLE,
        sections=REPORT_SECTIONS,
        facts_json=facts_json,
    )