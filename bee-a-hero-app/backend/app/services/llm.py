"""Chat provider interface.

`chat(messages, user_context)` returns the assistant's reply string.

Two providers, chosen by env:
  * Anthropic — when ANTHROPIC_API_KEY is set.
  * Mock — otherwise; a canned but sensible reply that references user_context,
    so the assistant feature demos with no key.
"""

from ..config import settings

from pathlib import Path
import json

# Models set in ONE constant each — change here to swap.
ANTHROPIC_MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-2.5-flash"

# Read the real CV + ML result files so the assistant answers are grounded in them.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CV_CSV = _REPO_ROOT / "test_video_result" / "csv"
_ML_YIELD = _REPO_ROOT / "models" / "yield_report.json"


def read_result_context() -> str:
    """A compact grounding block from the real CV + ML outputs (empty if none present).

    CV: per-flower summary (flower count, real landings by type, pollination score) from
    test_video_result/csv/ALL_flower_summary.csv. ML: the fruit-set + yield band from
    models/yield_report.json. Only measured numbers — the assistant must not invent others.
    """
    lines: list[str] = []
    summ = _CV_CSV / "ALL_flower_summary.csv"
    if summ.exists():
        try:
            import csv
            rows = list(csv.DictReader(open(summ)))
            n_flowers = len(rows)
            score = round(sum(float(r.get("pollination_score", 0) or 0) for r in rows), 1)
            real = sum(int(r.get("n_real_landings", 0) or 0) for r in rows)
            hb = sum(int(r.get("n_honeybee", 0) or 0) for r in rows)
            lines.append(f"CV (test-video results): {n_flowers} flowers tracked, {real} real "
                         f"landings ({hb} honeybee), total pollination score {score}.")
        except Exception:
            pass
    if _ML_YIELD.exists():
        try:
            y = json.loads(_ML_YIELD.read_text()).get("tracker_application", {}).get(
                "yield_estimate", {})
            if y:
                fs, yk = y.get("fruit_set_mean"), y.get("yield_kg_mean")
                lines.append(f"ML (fruit-set model): estimated fruit set {fs:.0%}, "
                             f"illustrative yield {yk:.0f} kg/tree (synthetic-fit, not field-calibrated).")
        except Exception:
            pass
    return " ".join(lines)

SYSTEM_PROMPT = (
    "You are the Bee-A-Hero assistant. Bee-A-Hero is a computer-vision system "
    "that watches pomegranate flowers, detects and tracks visiting insects, "
    "classifies each as pollinator or non-pollinator, and counts pollination "
    "visits per flower. Explain results and pollination concepts clearly and "
    "concisely, and answer questions about the user's own detection stats. "
    "When the user's stats are provided, ground your answers in them."
)


def _anthropic_chat(messages: list[dict], user_context: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system = SYSTEM_PROMPT
    if user_context:
        system += f"\n\nThe current user's data:\n{user_context}"

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _mock_chat(messages: list[dict], user_context: str) -> str:
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    context_line = (
        f" Based on your data — {user_context.strip()} —"
        if user_context
        else ""
    )
    return (
        f"(demo assistant){context_line} you asked: “{last_user}”. "
        "In Bee-A-Hero, a 'visit' is counted each time a tracked insect enters "
        "a flower's region; pollinator visits (bees, hoverflies) are the ones "
        "that drive pollination. Set ANTHROPIC_API_KEY to get full AI answers."
    )


def _hf_chat(messages: list[dict], user_context: str) -> str:
    # Open-source model via Hugging Face Inference. Low temperature for factual, technical
    # answers: softmax T=0.3 with top-p=0.9 nucleus sampling on the truncated vocabulary.
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=settings.hf_model, token=settings.hf_api_token or None)
    system = SYSTEM_PROMPT + (f"\n\nThe current user's data:\n{user_context}" if user_context else "")
    msgs = [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    out = client.chat_completion(messages=msgs, max_tokens=1024, temperature=0.3, top_p=0.9)
    return (out.choices[0].message.content or "").strip()


def _gemini_chat(messages: list[dict], user_context: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    system = SYSTEM_PROMPT + (f"\n\nThe current user's data:\n{user_context}" if user_context else "")
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1024),
    )
    return (resp.text or "").strip()


# provider id -> (settings key that must be set, chat fn). "auto" tries them in this order.
_PROVIDERS = {
    "gemini": ("gemini_api_key", _gemini_chat),
    "huggingface": ("hf_api_token", _hf_chat),
    "anthropic": ("anthropic_api_key", _anthropic_chat),
}
_AUTO_ORDER = ["gemini", "huggingface", "anthropic"]


def available_providers() -> list[dict]:
    """What the Assistant tab offers. `available` marks providers whose key/token is set."""
    out = [{"id": "auto", "label": "Auto", "available": True}]
    labels = {"gemini": "Gemini", "huggingface": "Hugging Face", "anthropic": "Claude"}
    for pid in _AUTO_ORDER:
        key = _PROVIDERS[pid][0]
        out.append({"id": pid, "label": labels[pid], "available": bool(getattr(settings, key, ""))})
    out.append({"id": "mock", "label": "Demo (offline)", "available": True})
    return out


def chat(messages: list[dict], user_context: str, provider: str | None = None) -> str:
    # ground every answer in the real CV + ML result files, alongside the caller's DB stats
    results = read_result_context()
    if results:
        user_context = (user_context + "\n" + results).strip() if user_context else results

    prov = (provider or "auto").lower()
    if prov == "mock":
        return _mock_chat(messages, user_context)
    if prov in _PROVIDERS:
        order = [prov]                      # user picked a specific provider
    else:
        order = _AUTO_ORDER                 # "auto"/unknown -> priority chain

    for pid in order:
        key, fn = _PROVIDERS[pid]
        if getattr(settings, key, ""):      # only try a provider whose key/token is configured
            try:
                return fn(messages, user_context)
            except Exception as exc:        # never let the demo break on an API hiccup
                print(f"[llm] {fn.__name__} failed ({type(exc).__name__}: {exc}); falling back.")
    return _mock_chat(messages, user_context)  # nothing configured / all failed
