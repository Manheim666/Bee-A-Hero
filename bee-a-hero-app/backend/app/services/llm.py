"""Chat provider interface.

`chat(messages, user_context)` returns the assistant's reply string.

Two providers, chosen by env:
  * Anthropic — when ANTHROPIC_API_KEY is set.
  * Mock — otherwise; a canned but sensible reply that references user_context,
    so the assistant feature demos with no key.
"""

from ..config import settings

# Model is set in ONE constant — change here to swap models.
ANTHROPIC_MODEL = "claude-sonnet-4-5"

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


def chat(messages: list[dict], user_context: str) -> str:
    if settings.anthropic_api_key:
        try:
            return _anthropic_chat(messages, user_context)
        except Exception as exc:  # never let the demo break on an API hiccup
            return (
                "(assistant fell back to demo mode after an API error: "
                f"{exc}). {_mock_chat(messages, user_context)}"
            )
    return _mock_chat(messages, user_context)
