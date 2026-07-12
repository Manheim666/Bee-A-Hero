# Bee-A-Hero — LLM Reporting Module

Turns real computer-vision pollination data into a factual, farmer-readable
report using Google Gemini — with a hard guarantee that the LLM **never
invents a number**.

```
real CV output (ALL_landings.csv)
        │
        ▼
cv_to_json_report.py   ── aggregates real landings, fits a dose-response
        │                  curve, writes cv_ml_report.json
        ▼
metrics.py              ── loads the JSON, computes session-wide totals
        │                  (aggregate block), validates logical bounds
        ▼
schemas.py               ── Pydantic validation (rejects NaN/Infinity/
        │                    negative counts/unknown species/empty data)
        ▼
prompts.py                 ── builds the LLM prompt (rules + few-shot examples)
        │
        ▼
llm_client.py                ── calls Gemini (retry, timeout, no-thinking-truncation)
        │
        ▼
fidelity_check.py              ── verifies every number in the report is
        │                          grounded in the source facts
        ▼
generate.py                      ── orchestrates all of the above
        │
        ▼
   final report text
```

---

## 1. File structure

```
src/llm_reporting/
├── schemas.py          # Pydantic models — input validation
├── metrics.py           # loads JSON, computes aggregate, validate_bounds()
├── prompts.py             # THE PROMPT — see full text below
├── llm_client.py            # Gemini API call (retry/timeout/logging)
├── fidelity_check.py          # post-hoc hallucination check
└── generate.py                 # ties everything together

cv_to_json_report.py    # separate script: real CV CSV -> cv_ml_report.json
docs/schemas/cv_ml_report.json   # the actual data file the LLM reads
tests/test_llm.py        # mock tests (no API key) + real Gemini demo
```

---

## 2. What the LLM actually sees — the full prompt

This is assembled by `prompts.build_report_prompt()` from three pieces:
`SYSTEM_PROMPT` (the rules), `FEW_SHOT_EXAMPLE` (copy-this-pattern examples),
`FIELD_REFERENCE` (what each JSON field means), plus the real facts JSON
appended at the end.

### 2.1 `SYSTEM_PROMPT`

```
You are a factual reporting assistant for a pollination-monitoring
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
```

### 2.2 `FEW_SHOT_EXAMPLE`

Few-shot examples are used **on purpose, instead of prose instructions**,
for the parts of the report that kept coming out wrong when only described
in words (see §6, "Why few-shot instead of instructions"). Illustrative
numbers only — never the real data.

```
Example (numbers below are illustrative only, not from the real input):

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
```

### 2.3 `FIELD_REFERENCE`

```
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
```

### 2.4 The final template

```
{SYSTEM_PROMPT}
{FIELD_REFERENCE}
{FEW_SHOT_EXAMPLE}

Now write a COMPREHENSIVE report for the REAL input below, with these sections:
1. Summary - flower counts, activity
2. Pollinator Activity - MUST follow the exact example sentence pattern above
3. Pollination Intensity - MUST open with the static clarifying sentence
   (rule 15), then pollination_score statistics
4. Hypothetical Pollination & Yield Estimate - MUST follow the exact
   Section 4 example sentence pattern above (rule 9)

Input:
{facts_json}   ← the real cv_ml_report.json content, serialized

Report:
```

---

## 3. Module-by-module

### `cv_to_json_report.py`
Reads `test_video_result/ALL_landings.csv` (real, per-landing CV output),
groups by flower, and computes real facts: landing counts, species
breakdown, `pollination_score` (using the exact `SPECIES_WEIGHT` formula
from `video_detect.py`), `dominant_pollinators` (tie-aware — returns *all*
tied species, never picks one arbitrarily), and `video_observation_span_s`
(elapsed time *within the video*, not a real clock duration — no real
timestamps exist in this dataset).

It also fits a dose-response curve (`F0`, `Fmax`, `k`) live from
`dataset_training_v11.csv` via `scipy.optimize.curve_fit`, and uses it to
produce an `ml_estimate` per flower (`fruit_set_probability`,
`yield_kg_per_flower_estimate`). **This estimate assumes a crop
(`assumed_crop_unverified: true`)** — the test videos are generic stock
footage, not verified orchard recordings, so the crop is unknown. The LLM
is instructed (rule 9) to always phrase this as a hypothesis.

### `metrics.py`
`to_facts_dict()` loads the JSON and adds an `aggregate` block: totals
across every flower (pollinator vs non-pollinator visits, species totals,
mean/max pollination score, mean fruit-set probability *and its
percentage form*, total estimated yield, dominant pollinator(s)).

`validate_bounds()` catches impossible CV output before it reaches the LLM
(e.g. more "real" landings than total landings, negative durations,
`flower_count` mismatch). If this returns any issues, `generate.py` returns
an empty report and skips the LLM call entirely.

### `schemas.py`
Pydantic models reject: NaN, Infinity, negative counts, probabilities
outside `[0, 1]`, unknown species names, and empty flower lists — before
the data is trusted enough to build a prompt from.

### `llm_client.py`
Calls Gemini with a 30s timeout and up to 3 retries (exponential backoff)
on transient errors (rate limits, 5xx). Explicitly **disables Gemini's
"thinking" mode** (`thinking_budget=0`) — Gemini 2.5 models think by
default and thinking tokens are deducted from `max_output_tokens`, which
was silently truncating reports mid-sentence once the prompt grew longer.
Also detects `finish_reason == MAX_TOKENS` and raises a clear error instead
of ever returning cut-off text.

### `fidelity_check.py`
After the LLM responds, extracts every number in the report text
(integers, decimals, negatives, percentages) and checks each one is
grounded in the source facts (with a narrow ±0.01 tolerance for float
noise only — not for real rounding). Video-ID digit sequences (e.g.
`108220-680177956`) are excluded from both sides, since they're
identifiers, not facts. A separate, best-effort `check_species_associations`
heuristic (logged as a warning, doesn't fail the check) flags sentences
where a species name and a number don't obviously correspond — using
word-boundary matching so "honeybee" is never confused with "bee".

### `generate.py`
The orchestrator: `to_facts_dict()` → `validate_bounds()` (short-circuits
on failure) → `ReportInput(**facts)` (schema check) → `build_report_prompt()`
→ `generate()` (Gemini) → `check_fidelity()` → `ReportOutput`.

---

## 4. Setup

`.env` at the **project root**:
```
LLM_MODEL=gemini-2.5-flash
LLM_API_KEY=your_real_key_here
LLM_TEMPERATURE=0.1
```

Optional, all have defaults:
```
LLM_MAX_OUTPUT_TOKENS=4096
LLM_TIMEOUT_S=30
LLM_MAX_RETRIES=3
```

```bash
pip install google-genai python-dotenv pydantic tenacity scipy pandas numpy
```

---

## 5. Running it

```bash
python tests/test_llm.py
```

This runs, in order:
1. Two **mocked** tests (no API key needed) — one confirms a well-formed
   report passes fidelity, one deliberately injects a fabricated number
   ("999") and confirms it gets caught.
2. A call to the **real Gemini API**, printing the full generated report.

`pytest` is intentionally not used — this project's test files are plain
Python scripts (`python tests/test_llm.py`), not `pytest`-discovered tests.

---

## 6. Design decisions worth knowing

**Why few-shot examples instead of prose instructions?** Early versions
described the required sentence patterns only in words ("MUST report both
numbers together"). This was silently ignored by the LLM even when marked
"mandatory" in multiple ways. Replacing the instruction with a literal
example sentence (copy-this-pattern) fixed it reliably — LLMs follow shown
patterns far more consistently than descriptions of a pattern.

**Why is `check_species_associations` a warning, not a hard failure?** It's
a sentence-level heuristic, not a guarantee — it can produce false
positives on natural phrasing (e.g. a sentence mentioning two species and
one number). Failing the whole report on a heuristic would create more
false rejections than it prevents real hallucinations.

**Why `thinking_budget=0`?** This is a pure data-reformatting task — the
LLM is never asked to reason, only to phrase already-computed facts in
prose. Gemini 2.5's default "thinking" mode adds latency and cost for no
benefit here, and — critically — was silently truncating output before
this was found and fixed.

---

## 7. Known limitations (stated plainly, not hidden)

- **Crop identity is unverified.** `test_video_result/` contains generic
  CV-pipeline test footage (Pexels stock videos), not confirmed pomegranate
  or cucumber orchard recordings. Every `ml_estimate` is explicitly flagged
  `assumed_crop_unverified: true` for this reason.
- **No real timestamps.** The CV pipeline (`video_detect.py`) never
  populates `flower_species` or real-world timestamps. `video_observation_span_s`
  is video-internal elapsed time only.
- **`k_synthetic` (used to fit the dose-response curve) is synthetic.** No
  real fruit-set ground truth exists for this project — the curve fit
  validates that the *machinery* works, not real-world biological accuracy.
- **`check_species_associations` is a heuristic**, not a complete guarantee
  against species/value misattribution — see §6.
