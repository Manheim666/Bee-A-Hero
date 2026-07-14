# 05 · LLM Assistant (Generative AI — Speaker 3)

## What it is
A chat assistant in the web app that answers questions about the user's own pollination results and the pollination science, **grounded in the real CV + ML outputs** — not a generic chatbot.

## Provider choice (per chat, in the Assistant tab)
A dropdown lets the user pick the model **per message**:
| Provider | Backend | Notes |
|---|---|---|
| **Gemini** | `google-genai`, model `gemini-2.5-flash` | needs `GEMINI_API_KEY` |
| **Hugging Face** | `huggingface_hub.InferenceClient`, `meta-llama/Llama-3.1-8B-Instruct` | open-source, needs `HF_API_TOKEN` |
| **Auto** | tries Gemini → HF | first configured provider wins |
| **Demo (offline)** | built-in mock | always works, references the user's stats |

Providers whose key isn't set show **"(add key)"** and are skipped; the chain always falls back to the grounded mock, so the feature **never breaks in a demo**. *(Anthropic/Claude was intentionally removed.)*

## Grounding — why answers are trustworthy
Every request is assembled from two grounding blocks so the model answers from measured numbers, not hallucination:
1. **The user's own DB stats** (videos processed, total visits, % pollinator, avg visits/flower, top flower).
2. **The real result files:** CV summary (`test_video_result/csv/ALL_flower_summary.csv` — flowers, real landings, honeybee count, pollination score) and the ML `yield_report.json` (fruit-set, illustrative yield). The system prompt tells it to **use only measured numbers and never invent others**, and to label the yield as illustrative.

## Decoding strategy (deterministic, factual)
Low temperature for technical Q&A — softmax with **T = 0.3** and **top-p = 0.9** nucleus sampling, `max_tokens = 1024`:
```
P(y_t | y_<t, x) = exp(z_t / T) / Σ_{j ∈ V^(p)} exp(z_j / T)
```
`T ≤ 0.3` keeps answers factual and repeatable; top-p truncates the vocabulary to the nucleus `V^(p)` so low-probability tokens can't derail a technical answer.

## Architecture
- Backend: `app/services/llm.py` (`chat(messages, user_context, provider)` + `available_providers()`), router `app/routers/chat.py` (`POST /conversations/{id}/messages` carries `provider`; `GET /conversations/providers` lists availability). Conversations + messages persisted in SQLite.
- Frontend: `ChatWindow.jsx` renders the provider dropdown, fetches availability, sends the choice with each message.
- **Security:** keys live only in the git-ignored `backend/.env`, read by an **absolute** `env_file` path so any launcher loads them without printing or committing. Verified absent from git history.

## Why Hugging Face (open-source path)
- No proprietary lock-in; a free-tier open model (Llama-3.1-8B-Instruct) via serverless Inference.
- `InferenceClient.chat_completion` gives an OpenAI-style interface with explicit temperature/top-p control.
- Swappable in one constant (`hf_model`) → could move to vLLM / a local endpoint for lower latency.

## Anticipated questions
- *"Does it hallucinate numbers?"* → It's given the real CV+ML numbers as context and instructed to use only those; the fallback mock also quotes real stats.
- *"Why low temperature?"* → factual, repeatable technical answers; top-p prevents unlikely tokens.
- *"What if the API is down / no key?"* → provider chain falls back to a grounded mock; the chat never dies.
- *"Open-source vs proprietary?"* → both offered; user picks per chat.
