import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from unittest.mock import patch

from src.llm_reporting.generate import generate_report
from src.llm_reporting.metrics import to_facts_dict

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = str(ROOT / "docs" / "schemas" / "cv_ml_report.json")


def test_generate_report_with_valid_mock():
    facts = to_facts_dict(REPORT_PATH)
    agg = facts["aggregate"]
    fake_report = (
        f"{agg['flower_count']} flowers were monitored. "
        f"{agg['total_pollinator_visits']} pollinator visits were recorded, "
        f"compared to {agg['total_non_pollinator_visits']} non-pollinator visits. "
        f"The mean pollination score was {agg['mean_pollination_score']}."
    )

    with patch("src.llm_reporting.generate.generate", return_value=fake_report):
        result = generate_report(REPORT_PATH)

    assert result.report_text != "", "report_text should not be empty"
    assert result.fidelity_passed is True, f"expected True, got flagged: {result.flagged_numbers}"
    assert result.flagged_numbers == [], f"expected [], got {result.flagged_numbers}"
    print("test_generate_report_with_valid_mock PASSED")


def test_generate_report_with_invalid_mock():
    fake_report = "The system detected 999 unexplained pollinator events."

    with patch("src.llm_reporting.generate.generate", return_value=fake_report):
        result = generate_report(REPORT_PATH)

    assert result.fidelity_passed is False, "expected fidelity_passed False for a hallucinated number"
    assert "999" in result.flagged_numbers, f"expected '999' in flagged_numbers, got {result.flagged_numbers}"
    print("test_generate_report_with_invalid_mock PASSED")


def show_real_report():
    """Calls the REAL Gemini API (requires LLM_API_KEY in .env) and prints
    the full generated report - this is what actually shows you the output,
    the two tests above only check pass/fail with fake mock text.
    """
    print("\n" + "=" * 70)
    print("REAL LLM REPORT (Gemini API)")
    print("=" * 70)

    result = generate_report(REPORT_PATH)

    print(result.report_text)
    print()
    print("-" * 70)
    print("Fidelity passed:", result.fidelity_passed)
    print("Flagged numbers:", result.flagged_numbers)
    print("-" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("MOCK TESTS (no API key needed, fast sanity checks)")
    print("=" * 70)
    test_generate_report_with_valid_mock()
    test_generate_report_with_invalid_mock()
    print("\nAll mock tests passed.\n")

    show_real_report()