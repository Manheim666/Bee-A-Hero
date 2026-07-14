# 10 · Presentation Script — The Clear Path (built for full marks)

**Goal:** maximize the Jury 40 (= 30% of grade). The four scored things are **Clarity & structure**, **Delivery & confidence**, **Problem definition & relevance**, **Solution logic & feasibility**. This script is engineered to hit **9–10 on each**.

> Winner formula: `(Technical/100)×0.7 + (Jury/40)×0.3`. The jury only sees the pitch — so the pitch must *show* the rigor, not just claim it.

---

## The narrative arc (this is the "clear structure" the rubric rewards)
**Problem → Insight → Solution → Proof (demo) → How → Honest limits → Impact.**
One straight line. No backtracking. Every section hands off to the next in one sentence.

## Timing & speaker map (~5 min pitch + Q&A) — "communicates as a unit"
Four speakers, balanced (the rubric explicitly rewards a *balanced team*). Roles map to who built what.

| # | Beat | Time | Speaker | Owns |
|---|---|---|---|---|
| 1 | Hook | 0:00–0:20 | **Asif** (lead) | the story |
| 2 | Problem & relevance | 0:20–0:55 | Asif | evidence-backed problem |
| 3 | The insight (lift ≠ dependency) | 0:55–1:20 | Asif → Raul | the differentiator |
| 4 | Solution overview | 1:20–1:45 | **Raul** | detect→count→report |
| 5 | **LIVE DEMO** | 1:45–3:15 | **Khaver** drives, **Narmin** narrates | the proof |
| 6 | How it works (1 slide) | 3:15–4:00 | Raul (CV) + Narmin (ML/LLM) | pipeline + math |
| 7 | Feasibility, trade-offs, limits | 4:00–4:35 | Narmin | Excellent-tier honesty |
| 8 | Impact + close + ask | 4:35–5:00 | Asif | the memorable ending |
| — | Q&A | after | **all** — see routing below | handle as a unit |

*Every member speaks. Practice the hand-offs out loud — a clean hand-off is what "confident, balanced team" looks like.*

---

## The script (say roughly this)

### 1 · Hook — Asif (20s)
> "A third of the food we eat depends on pollinators — yet a farmer cannot *see* what the bees actually do for their crop. It's invisible, so it's unmanaged. We made it visible."

### 2 · Problem & relevance — Asif (35s) — *targets Problem 9–10: specific, evidence-backed*
> "Meet the pomegranate grower in Baku — pomegranate is a national crop here, there's a whole festival for it. They can't measure pollinator value, so they can't justify hives, time their spraying, or protect habitat. Manual watching doesn't scale — one human, one flower. And the science is subtle: pomegranate is **self-fertile** — it sets about **45%** of fruit with *no* insects, up to **68%** with cross-pollination. So the real question isn't 'are there bees' — it's *'how much do they add?'*"

### 3 · The insight — Asif → Raul (25s) — *the differentiator; wins Solution logic*
> "Most projects would count bees and say 'more bees, more fruit.' For a self-fertile crop that's **biologically wrong**. We model the **marginal lift** insects add over that 45% floor, capped at the 68% ceiling — a bounded, defensible number. That one decision is the difference between a bee-counter and a measurement system."
> *(hand-off)* "Here's how we built it — Raul."

### 4 · Solution overview — Raul (25s)
> "Bee-A-Hero: **Detect → Count → Report.** We detect the flowers and the insects, track each insect so we never double-count, count real *feeding visits* — not frames — and turn visitation into a pollination-lift and yield estimate. It runs as a web app, a live phone-camera mode, and an AI assistant you can just talk to. Let me show you."

### 5 · LIVE DEMO — Khaver drives, Narmin narrates (90s) — **the single most important 90 seconds**
Choreography (rehearse until it's muscle memory):
1. **Upload** a prepared clip (bee on a pomegranate flower). *Narmin:* "One upload — the same pipeline runs whether it's a file, the website, or the live camera. One source of truth."
2. **Show the annotated video** — boxes, insect ID + type, the flower, live counts. *Narmin:* "Watch — it tracks this honeybee, and when it dips behind a petal it's re-linked, not counted twice."
3. **Show the result:** flowers, pollinator visits, the pollination score. "A honeybee is weighted 10× a random fly — because it actually pollinates."
4. **Ask the assistant:** type *"How many pollinator visits did I get, and what's my pollination lift?"* → it answers **from the real numbers**. "It's grounded in our actual results — it can't make numbers up."
5. **Live camera** (if network allows): point at the flower on a second screen → boxes appear live; a landing gets logged to CSV. *"Same detector, live, writing pollination data as it happens."*

> **Backup plan (say nothing, just switch):** if upload/live is slow or the Wi-Fi drops, play the **pre-recorded annotated clip** already open in a tab. Never debug on stage. Having a backup *is* the "confident, prepared" score.

### 6 · How it works — Raul (CV) then Narmin (ML/LLM) (45s, one diagram slide)
> **Raul:** "YOLO26 detects flowers and insects; BoT-SORT tracks them. The core trick is a **visit state machine** — a landing only counts at **2 seconds** of dwell, and we stitch tracks across occlusion so one bee is one visit."
> **Narmin:** "Then the math: visits → a saturating dose-response between the 45% floor and 68% ceiling → a pollination **lift**. And the assistant runs on **open models — Gemini or Hugging Face** — grounded in those exact numbers, at low temperature so it stays factual."

### 7 · Feasibility, trade-offs & limits — Narmin (35s) — *this is the Excellent unlock*
> "We were deliberate about scope. It runs **on CPU** — no GPU needed. We report **relative lift, not absolute kilograms**, because you can't calibrate tonnage in one bloom season — and claiming you can is the weakest thing you could show a jury. Our honest limits: the classifier confuses some insect species, the flower detector misses the occasional bloom, and multi-camera is designed but not shipped. We know exactly what's measured and what's estimated."

### 8 · Impact + close — Asif (25s)
> "Bee-A-Hero turns a camera into a pollination instrument — so growers can value the bees, researchers get an automated index, and conservation gets hard evidence. We didn't build a bee counter. We built a way to **make a hero visible.** Thank you — we'd love your questions."

---

## Q&A routing — handle it *as a unit* (Delivery 9–10)
Whoever owns the topic answers; one person doesn't hog it. Keep answers to ~2 sentences (full bank in `09_QA`).

| Question theme | Lead answer | Backup |
|---|---|---|
| Science / "more bees = more fruit?" / yield | Asif | Narmin |
| CV / detection / tracking / double-count | Raul | Asif |
| ML math / lift / how you'd validate | Narmin | Asif |
| LLM / assistant / hallucination | Narmin | Khaver |
| App / live / architecture / "is it real-time?" | Khaver | Raul |
| Business / "who pays" / market / scale | Asif | Khaver |

**Golden rules:** (1) Repeat the question so everyone hears it. (2) If you don't know, say "great question — here's how we'd find out," never bluff (bluffing tanks Delivery). (3) End every answer looking at the juror, not the screen.

---

## Pre-flight checklist (do this 30 min before)
- [ ] Servers up: `bash run-website.sh` → backend :8000, frontend :5173, live :8001 green.
- [ ] `GEMINI_API_KEY` / `HF_API_TOKEN` in `backend/.env` so the assistant answers live (else it falls back to the grounded mock — still fine).
- [ ] Prepared upload clip + the **pre-recorded annotated video** open in a tab (demo backup).
- [ ] Live camera pointed at a **real flower** or a **test video playing full-screen** (photos on a screen won't detect — out of distribution).
- [ ] Assistant question pre-typed and ready to send.
- [ ] One slide = the pipeline diagram (`07_Architecture_Deployment.md`), one slide = the 45%→68% lift, one number board (mAP 0.808 / 0.669, visit ≥ 2 s, honeybee 10×).
- [ ] Each speaker has said their lines out loud at least 3×. Time it: ≤ 5:00.

## The three sentences that must land
1. "**Lift, not dependency** — the marginal value bees add over a 45% self-pollination floor."
2. "We count **feeding visits, not frames** — one bee behind a petal is counted **once**."
3. "**One pipeline, one source of truth** — the website, the CSVs, and the live camera can't disagree."
