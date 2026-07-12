import json

from src.llm_reporting.schemas import ReportInput, ReportOutput
from src.llm_reporting.metrics import to_facts_dict, validate_bounds
from src.llm_reporting.prompts import build_report_prompt
from src.llm_reporting.fidelity_check import check_fidelity
from src.llm_reporting.llm_client import generate


def generate_report(report_json_path: str) -> ReportOutput:
    facts = to_facts_dict(report_json_path)

    issues = validate_bounds(facts)
    if issues:
        return ReportOutput(report_text="", fidelity_passed=False, flagged_numbers=issues)

    ReportInput(**facts)  # validate shape before sending to the LLM

    prompt = build_report_prompt(json.dumps(facts, indent=2))
    report_text = generate(prompt)
    fidelity_passed, flagged = check_fidelity(report_text, facts)

    return ReportOutput(
        report_text=report_text,
        fidelity_passed=fidelity_passed,
        flagged_numbers=flagged,
    )