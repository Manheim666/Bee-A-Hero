"""Gemini API client for the LLM reporting module.

Handles authentication, request configuration, timeouts, retries with
exponential backoff, and error translation into informative exceptions.
Public interface (`generate(prompt: str) -> str`) is unchanged from the
previous version.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMGenerationError(Exception):
    """Raised when the LLM API call fails after all retries, or returns an
    unusable (empty/blocked) response. Wraps the underlying API error with
    a clearer, actionable message.
    """


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient errors (rate limits, server errors, timeouts);
    do not retry on permanent errors (bad API key, invalid request)."""
    if isinstance(exc, genai_errors.APIError):
        # 429 = rate limited, 5xx = server-side transient error
        return exc.code == 429 or (exc.code is not None and 500 <= exc.code < 600)
    return isinstance(exc, (TimeoutError, ConnectionError))


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type((genai_errors.APIError, TimeoutError, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini(client: genai.Client, prompt: str, temperature: float, max_output_tokens: int) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # Gemini 2.5 models "think" by default, and thinking tokens are
            # deducted from max_output_tokens - this task is deterministic
            # reformatting of already-given facts, not reasoning, so
            # thinking is disabled to avoid silently truncating the visible
            # report (a real bug hit in production: a longer prompt caused
            # more internal "thinking", leaving too little budget for the
            # actual report text, cutting it off mid-sentence).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)),
        ),
    )

    candidate = response.candidates[0] if response.candidates else None
    finish_reason = getattr(candidate, "finish_reason", None) if candidate else None

    if finish_reason is not None and str(finish_reason) not in ("FinishReason.STOP", "STOP"):
        usage = getattr(response, "usage_metadata", None)
        logger.error(
            "Gemini response did not finish normally: finish_reason=%s, usage=%s",
            finish_reason, usage,
        )
        if str(finish_reason) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS"):
            raise LLMGenerationError(
                f"Gemini response was truncated (finish_reason=MAX_TOKENS) - "
                f"max_output_tokens={max_output_tokens} was not enough. "
                f"Increase LLM_MAX_OUTPUT_TOKENS or shorten the prompt."
            )

    if not response.text:
        raise LLMGenerationError(
            "Gemini returned an empty response - the prompt may have been "
            "blocked by safety filters, or the model produced no output."
        )
    return response.text


def generate(
    prompt: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Generates text from the configured Gemini model.

    Args:
        prompt: The full prompt to send.
        temperature: Overrides the default LLM_TEMPERATURE env value if given.
        max_output_tokens: Overrides the default LLM_MAX_OUTPUT_TOKENS env
            value if given.

    Returns:
        The generated text.

    Raises:
        LLMGenerationError: If the API call fails after all retries, the
            API key is missing, or the response is empty/blocked.
    """
    if not API_KEY:
        raise LLMGenerationError(
            "LLM_API_KEY is not set - add it to your .env file at the project root."
        )

    effective_temperature = TEMPERATURE if temperature is None else temperature
    effective_max_tokens = MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens

    client = genai.Client(api_key=API_KEY)
    logger.info(
        "Calling Gemini model=%s temperature=%.2f max_output_tokens=%d timeout=%.0fs",
        MODEL_NAME, effective_temperature, effective_max_tokens, TIMEOUT_S,
    )

    try:
        text = _call_gemini(client, prompt, effective_temperature, effective_max_tokens)
    except genai_errors.APIError as exc:
        logger.error("Gemini API call failed after retries: %s", exc)
        raise LLMGenerationError(f"Gemini API call failed: {exc}") from exc
    except (TimeoutError, ConnectionError) as exc:
        logger.error("Gemini API call timed out or lost connection: %s", exc)
        raise LLMGenerationError(f"Gemini API network error: {exc}") from exc

    logger.info("Gemini call succeeded (%d chars returned)", len(text))
    return text