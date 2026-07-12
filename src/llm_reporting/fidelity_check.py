"""Fidelity checking for LLM-generated pollination reports.

Verifies that every number the LLM writes in its report text is grounded
in the source facts JSON - i.e. the LLM did not hallucinate, round, or
invent a value. This is the last line of defense before a report reaches
a user.

DESIGN NOTE ON TOLERANCE: tolerance is intentionally very narrow (see
DEFAULT_TOLERANCE). The purpose of this check is to catch hallucination,
not to be lenient - a wider tolerance would let the LLM "round" numbers
undetected, defeating the check's purpose. Tolerance here only absorbs
genuine floating-point representation noise, not meaningful rounding.

KNOWN LIMITATION - species/value association: this module verifies that
every NUMBER in the report text also appears somewhere in the facts. It
does NOT verify that a number is attributed to the CORRECT species/field
(e.g. it cannot fully prevent the LLM from writing "43 honeybee visits"
when the fact was actually "43 fly visits", if 43 legitimately appears
elsewhere in the facts for a different field). `check_species_associations`
below provides a best-effort, sentence-level heuristic for this, but it is
a heuristic, not a guarantee - see its own docstring.
"""

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.01  # absolute tolerance for float comparison - see module docstring

# Matches: integers, decimals, negative numbers, and numbers with a
# trailing '%', as single tokens (e.g. "63.87", "-2", "50.31%").
_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*%?")

SPECIES_NAMES = ("honeybee", "bee", "fly", "beetle", "bug", "butterfly")


def _parse_number_token(token: str) -> float | None:
    """Converts a matched number token (possibly with a trailing '%') to
    a plain float.

    Percentages are returned as their percent VALUE (e.g. "50.31%" ->
    50.31), not divided by 100 - `_extract_numbers_from_facts` separately
    adds the x100 form of any 0-1 fact so both phrasings are grounded.
    """
    try:
        return float(token.rstrip("%"))
    except ValueError:
        logger.warning("Could not parse number token: %r", token)
        return None


def _extract_numbers(text: str) -> set[float]:
    """Extracts all numeric values (int, float, negative, percentage) from
    free text as a set of floats."""
    numbers: set[float] = set()
    for token in _NUMBER_PATTERN.findall(text):
        value = _parse_number_token(token)
        if value is not None:
            numbers.add(value)
    return numbers


def _extract_numbers_from_facts(facts: dict) -> set[float]:
    """Recursively walks a facts dict/list and collects every numeric value
    found, including numbers embedded in strings (e.g. video filenames).

    Probabilities in [0, 1] are also added in their x100 percentage form,
    since reports commonly phrase e.g. 0.5031 as "50.31%".
    """
    numbers: set[float] = set()

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list):
            for v in value:
                _walk(v)
        elif isinstance(value, bool):
            return  # bool is a subclass of int in Python - skip explicitly
        elif isinstance(value, (int, float)):
            numbers.add(float(value))
            if 0 <= value <= 1:
                numbers.add(round(float(value) * 100, 6))
        elif isinstance(value, str):
            numbers.update(_extract_numbers(value))

    _walk(facts)
    return numbers


def _get_video_digit_numbers(facts: dict) -> set[float]:
    """Video IDs (e.g. '108220-680177956') contain digit sequences that
    are identifiers, not facts - they must never be flagged as
    unexplained just because the LLM did not reproduce them verbatim.
    """
    video_numbers: set[float] = set()
    for f in facts.get("flowers", []):
        video = f.get("cv_facts", {}).get("video", "")
        for token in re.findall(r"\d+", video):
            try:
                video_numbers.add(float(token))
            except ValueError:
                continue
    return video_numbers


def _matches_within_tolerance(value: float, fact_numbers: set[float], tolerance: float) -> bool:
    """True if `value` matches any fact number within `tolerance`."""
    return any(abs(value - fact) <= tolerance for fact in fact_numbers)


def check_species_associations(report_text: str, facts: dict) -> list[str]:
    """Best-effort check that species names mentioned near a number in the
    report correspond to a real (species, value) pairing somewhere in the
    facts.

    This is a HEURISTIC, not a guarantee (see module docstring): it only
    inspects the sentence containing each species mention, and only
    returns warnings (never fails fidelity on its own) because it can
    produce false positives on natural phrasing (e.g. a sentence
    mentioning two species and one number).

    Word-boundary matching is used for species names (e.g. "\\bbee\\b") so
    that "honeybee" is never misread as containing a separate "bee"
    mention, and "butterfly" is never misread as containing "fly".

    Args:
        report_text: The LLM-generated report to inspect.
        facts: The source facts dict.

    Returns:
        A list of human-readable warning strings (empty if no concerns).
    """
    warnings: list[str] = []
    species_totals: dict[str, int] = {}
    for f in facts.get("flowers", []):
        cv = f.get("cv_facts", {})
        for species in SPECIES_NAMES:
            species_totals[species] = species_totals.get(species, 0) + cv.get(f"n_{species}", 0)
    agg_species = facts.get("aggregate", {}).get("species_totals", {})
    if agg_species:
        species_totals.update(agg_species)

    sentences = re.split(r"(?<=[.!?])\s+", report_text)
    for sentence in sentences:
        lower = sentence.lower()
        mentioned_species = [s for s in SPECIES_NAMES if re.search(rf"\b{s}\b", lower)]
        if not mentioned_species:
            continue
        numbers_in_sentence = _extract_numbers(sentence)
        for species in mentioned_species:
            true_value = species_totals.get(species)
            if true_value is None or not numbers_in_sentence:
                continue
            if not any(abs(n - true_value) <= DEFAULT_TOLERANCE for n in numbers_in_sentence):
                warnings.append(
                    f"Sentence mentions '{species}' with number(s) {sorted(numbers_in_sentence)}, "
                    f"but the known total for '{species}' is {true_value}: {sentence.strip()!r}"
                )
    return warnings


def check_fidelity(
    report_text: str,
    facts: dict,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Checks that every number in `report_text` is grounded in `facts`.

    Public interface is unchanged from the previous version (returns a
    (passed, flagged_numbers) tuple) so callers such as generate.py do not
    need to change. Species-association warnings (see
    `check_species_associations`) are logged, not returned, to keep this
    interface stable.

    Args:
        report_text: The LLM-generated report to verify.
        facts: The source facts dict the report was generated from.
        tolerance: Absolute tolerance for float matching. Kept narrow by
            design - see module docstring.

    Returns:
        A (passed, flagged_numbers) tuple, where flagged_numbers is a
        sorted list of number strings from the report that could not be
        grounded in facts.
    """
    report_numbers = _extract_numbers(report_text)
    fact_numbers = _extract_numbers_from_facts(facts)
    video_numbers = _get_video_digit_numbers(facts)

    unexplained = sorted(
        n for n in report_numbers
        if not _matches_within_tolerance(n, fact_numbers, tolerance)
        and n not in video_numbers
    )
    flagged_str = [f"{n:g}" for n in unexplained]

    species_warnings = check_species_associations(report_text, facts)
    if species_warnings:
        logger.warning(
            "Fidelity check found %d species-association warning(s): %s",
            len(species_warnings), species_warnings,
        )

    passed = len(unexplained) == 0
    if not passed:
        logger.warning("Fidelity check FAILED: %d unexplained number(s): %s", len(unexplained), flagged_str)
    else:
        logger.info("Fidelity check passed (%d numbers verified)", len(report_numbers))

    return passed, flagged_str