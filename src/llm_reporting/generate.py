"""LLM reporting stage: turn the CV + ML numbers into a plain-language grower report.

This is the final stage of the Bee-A-Hero pipeline (``Data -> CV -> ML -> LLM``). It reads
the *grounded* numbers produced upstream — the per-flower landing tables from the CV tracker
(``test_video_result/csv/ALL_flower_summary.csv``) and the fitted fruit-set curve + yield band
from the ML stage (``models/dose_response_v11.json`` + ``models/yield_report.json``) — and writes
a short, farmer-friendly Markdown report.

Two back-ends, chosen automatically:

* **Claude** (default when the ``anthropic`` SDK is installed *and* a credential resolves) — a
  grounded narrative written by ``claude-opus-4-8``. The prompt hands the model only the measured
  facts and instructs it never to invent numbers, so every figure in the prose traces to a CSV/JSON.
* **Offline template** (fallback) — a deterministic Markdown report built from the same facts with
  no network call. This keeps the whole pipeline runnable from a clean checkout with no API key.

Setup (once):  pip install -r src/llm_reporting/requirements-llm.txt
Run:           python -m src.llm_reporting.generate            # offline unless a key resolves
               python -m src.llm_reporting.generate --model claude-opus-4-8 --out report.md

Credentials: the ``anthropic`` SDK resolves ``ANTHROPIC_API_KEY`` (or an ``ant auth login``
profile) automatically — no key is hard-coded here. Without one, the offline template is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Repo paths (self-contained, no config dependency — portable across machines and branches).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSV_DIR = _REPO_ROOT / "test_video_result" / "csv"
_MODELS_DIR = _REPO_ROOT / "models"

MODEL = "claude-opus-4-8"  # most capable Claude model; narrates the numbers from src.ml_models

# The grower-facing crop for the yield figure. Illustrative until real orchard counts are supplied.
DEFAULT_CROP = "pomegranate"


# --------------------------------------------------------------------------- #
# 1. Gather the grounded facts (the only numbers the report is allowed to use)
# --------------------------------------------------------------------------- #
def collect_facts(csv_dir: Path = _CSV_DIR, models_dir: Path = _MODELS_DIR,
                  crop: str = DEFAULT_CROP) -> dict:
    """Assemble every measured number the report may cite, from the CV + ML artifacts.

    Reads only files the upstream stages actually wrote; anything missing is reported as such
    rather than guessed, so the report never presents an absent figure as measured.
    """
    facts: dict = {"crop": crop, "sources": []}

    # --- CV stage: per-flower landing summary ---
    summ_path = csv_dir / "ALL_flower_summary.csv"
    land_path = csv_dir / "ALL_landings.csv"
    if summ_path.exists():
        fs = pd.read_csv(summ_path)
        facts["n_flowers"] = int(len(fs))
        facts["pollination_score_total"] = round(float(fs["pollination_score"].sum()), 1)
        facts["n_videos"] = int(fs["video"].nunique()) if "video" in fs.columns else None
        facts["sources"].append("test_video_result/csv/ALL_flower_summary.csv")
    if land_path.exists():
        land = pd.read_csv(land_path)
        facts["n_episodes"] = int(len(land))
        real = land[land["is_real_landing"] == True]  # noqa: E712 (pandas mask)
        facts["n_real_landings"] = int(len(real))
        facts["real_by_type"] = {k: int(v) for k, v in
                                 real["insect_type"].value_counts().items()}
        facts["n_honeybee_real"] = int((real["insect_type"] == "honeybee").sum())
        facts["sources"].append("test_video_result/csv/ALL_landings.csv")

    # --- ML stage: fitted curve + yield band ---
    curve_path = models_dir / "dose_response_v11.json"
    yield_path = models_dir / "yield_report.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        facts["gates"] = curve.get("gates")
        c = curve.get("crops", {}).get(crop) or next(iter(curve.get("crops", {}).values()), {})
        facts["curve"] = {"F0": c.get("F0"), "Fmax": c.get("Fmax"), "k": c.get("k")}
        facts["sources"].append("models/dose_response_v11.json")
    if yield_path.exists():
        yr = json.loads(yield_path.read_text())
        app = yr.get("tracker_application", {})
        y = app.get("yield_estimate", {})
        facts["mean_effective_dose"] = app.get("mean_effective_dose")
        facts["fruit_set_mean"] = y.get("fruit_set_mean")
        facts["fruit_set_ci95"] = y.get("fruit_set_ci95")
        facts["yield_kg_mean"] = y.get("yield_kg_mean")
        facts["yield_kg_ci95"] = y.get("yield_kg_ci95")
        facts["yield_note"] = app.get("note")
        facts["sources"].append("models/yield_report.json")

    return facts


# --------------------------------------------------------------------------- #
# 2a. Offline back-end — deterministic template (no API key needed)
# --------------------------------------------------------------------------- #
def offline_report(facts: dict) -> str:
    """A grounded Markdown report built directly from the facts, with no model call."""
    L = ["# Pollination report", ""]
    crop = facts.get("crop", "crop")
    nv = facts.get("n_videos")
    L.append(f"**Crop:** {crop}  ·  **Footage analysed:** "
             f"{nv if nv is not None else 'n/a'} clips")
    L.append("")
    L.append("## What the camera saw")
    if "n_real_landings" in facts:
        L.append(f"- **{facts['n_real_landings']} real pollination visits** "
                 f"(insects that stayed >= 2 s) across "
                 f"**{facts.get('n_flowers', 'n/a')} flowers**, out of "
                 f"{facts.get('n_episodes', 'n/a')} total landing episodes.")
        hb = facts.get("n_honeybee_real")
        if hb is not None:
            L.append(f"- **{hb} of those were honeybees** — the strongest pollinators, "
                     f"weighted most heavily in the score.")
        by = facts.get("real_by_type", {})
        if by:
            L.append("- Visits by insect type: "
                     + " . ".join(f"{k} {v}" for k, v in by.items()) + ".")
    if "pollination_score_total" in facts:
        L.append(f"- Combined **pollination score: {facts['pollination_score_total']}** "
                 f"(honeybee visits count 10x).")
    L.append("")
    L.append("## What it means for fruit set")
    if facts.get("fruit_set_mean") is not None:
        ci = facts.get("fruit_set_ci95") or [None, None]
        L.append(f"- Estimated **fruit set: {facts['fruit_set_mean']:.0%}** "
                 f"(95% range {ci[0]:.0%}-{ci[1]:.0%}) at the observed visit rate.")
    if facts.get("yield_kg_mean") is not None:
        ci = facts.get("yield_kg_ci95") or [None, None]
        L.append(f"- Illustrative **yield: {facts['yield_kg_mean']:.0f} kg/tree** "
                 f"(95% range {ci[0]:.0f}-{ci[1]:.0f}).")
    if facts.get("yield_note"):
        L.append(f"- *{facts['yield_note']}*")
    L.append("")
    L.append("## Caveats")
    L.append("- Visits are measured directly; fruit set and yield are model estimates that need "
             "field ground-truth before they are firm.")
    L.append("- The honeybee split is provisional and currently over-calls honeybee.")
    L.append("")
    L.append("---")
    L.append("*Sources: " + ", ".join(facts.get("sources", [])) + ".*")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 2b. Claude back-end — grounded narrative from claude-opus-4-8
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are an agronomy assistant writing a short, plain-language pollination report for an "
    "orchard grower. You are given a JSON block of MEASURED facts from a computer-vision + "
    "modeling pipeline. Rules: use ONLY the numbers in the facts; never invent or extrapolate a "
    "figure that is not present; if a number is missing say so plainly. Distinguish measured "
    "visits (direct) from modeled fruit-set/yield (estimates with uncertainty). Keep it under "
    "~250 words, warm and concrete, Markdown with short sections. End with a one-line caveat that "
    "fruit-set/yield are estimates pending field data."
)


def claude_report(facts: dict, model: str = MODEL) -> str:
    """Generate the report with Claude, grounding it strictly in ``facts``.

    Raises if the ``anthropic`` SDK is absent or no credential resolves — the caller falls back
    to :func:`offline_report`.
    """
    import anthropic  # imported lazily so the offline path needs no dependency

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile from the environment
    user = ("Here are the measured facts as JSON. Write the grower report.\n\n"
            + json.dumps(facts, indent=2))
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},  # let Claude decide how much to reason
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# --------------------------------------------------------------------------- #
# 3. Orchestration
# --------------------------------------------------------------------------- #
def generate_report(out_path: Path, *, model: str = MODEL, crop: str = DEFAULT_CROP,
                    force_offline: bool = False) -> tuple[str, str]:
    """Build the report, preferring Claude and falling back to the offline template.

    Returns ``(report_markdown, backend_used)`` where backend is ``"claude"`` or ``"offline"``.
    """
    facts = collect_facts(crop=crop)
    backend = "offline"
    if not force_offline:
        try:
            report = claude_report(facts, model=model)
            backend = "claude"
        except Exception as e:  # missing SDK, no key, network, etc. -> deterministic fallback
            print(f"[llm] Claude backend unavailable ({type(e).__name__}: {e}); "
                  f"using offline template.")
            report = offline_report(facts)
    else:
        report = offline_report(facts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n")
    print(f"[llm] wrote {backend} report -> {out_path}")
    return report, backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO_ROOT / "docs" / "results" / "cv" /
                                         "pollination_report.md"),
                    help="output Markdown path")
    ap.add_argument("--model", default=MODEL, help="Claude model id")
    ap.add_argument("--crop", default=DEFAULT_CROP)
    ap.add_argument("--offline", action="store_true",
                    help="force the offline template (skip the Claude call)")
    args = ap.parse_args()
    generate_report(Path(args.out), model=args.model, crop=args.crop,
                    force_offline=args.offline)


if __name__ == "__main__":
    main()
