# 11 · Rubric Alignment, Creative Reasoning & the Full-Marks Plan

The jury scores **40 pts = 30% of the final grade**. Winner = `(Technical/100)×0.7 + (Jury/40)×0.3`. This file maps our project to **each rubric criterion** so we score **Excellent (9–10) on all four**, plus the creative hooks that win the two **Special Mention** votes.

---

## Part A — Score 9–10 on every jury criterion

### 1. Clarity & structure — /10 → aim 10
*Rubric: "clear narrative arc; problem, solution, outcome easy to follow; compelling, well-paced."*
- **What we do:** one straight line — Problem → Insight → Solution → Demo → How → Limits → Impact (see `10_PRESENTATION_SCRIPT.md`). Every beat hands to the next in one sentence.
- **Proof of pacing:** ≤ 5 min, timed, each of the 4 speakers owns a segment.
- **Say this to signal structure:** open with "Here's the one idea: make pollination visible and *valued*," and callback to it in the close.

### 2. Delivery & confidence — /10 → aim 10
*Rubric: "confident, handles Q&A, communicates as a unit; balanced team."*
- **Balanced team:** all four speak; clean rehearsed hand-offs.
- **Confidence signals:** repeat each question, answer in 2 sentences, "here's how we'd find out" instead of bluffing, eye contact on the juror.
- **Preparedness = confidence:** a **pre-recorded demo backup** means we never debug on stage. The `09_QA` bank means no question surprises us.
- **Unit language:** always "we," topic-owner answers, one backup adds a sentence — never talk over each other.

### 3. Problem definition & relevance — /10 → aim 10
*Rubric: "specific, grounded, evidence-backed real problem worth solving."*
- **Specific persona:** the pomegranate grower in Baku (local, real, culturally iconic — Goychay festival).
- **Evidence-backed:** 45% self-set → 68% cross-set (literature); open-field 24.93 vs control 15.36 kg/plant. Numbers, not vibes.
- **Why it's worth solving:** pollinator value is invisible → mis-managed hives/habitat/spraying; a third of food depends on pollinators.

### 4. Solution logic & feasibility — /10 → aim 10
*Rubric: "addresses the problem; realistic scope; trade-offs and limitations acknowledged."* ← **the Excellent unlock is honesty.**
- **Directly addresses it:** we measure the *marginal lift*, which is exactly the grower's question.
- **Realistic scope:** CPU-only, batch + live, one crop, one camera — shipped and working end-to-end.
- **Trade-offs stated out loud (this is what most teams forget):**
  - Report **relative lift, not absolute kg** — can't calibrate tonnage in one season.
  - **Lift model, not dependency** — the scientifically correct, less flashy choice.
  - Known limits: species-classifier confusion, occasional missed bloom, multi-camera designed-not-shipped, honeybee sub-classifier is provisional (thin *Apis* data).
- **Feasibility evidence:** live end-to-end demo; committed weights; one-command launch.

---

## Part B — Creative reasoning (the memorable angles that make jurors *want* to score us high)

1. **"Make a hero visible."** The name isn't decoration — the bee is the unsung hero of the food system, and we literally draw a box around it and score its contribution. The whole pitch is one metaphor executed.
2. **The intellectual flex — "lift, not dependency."** Any team can count bees. We noticed the crop is self-fertile and refused the easy-but-wrong linear story. That single decision signals scientific maturity — it's catnip for a jury that rewards rigor.
3. **"One source of truth."** The website, the offline CSVs, and the live camera run the *identical* pipeline. It's an engineering-integrity story: our numbers *can't* disagree with themselves. Jurors trust demos they can't poke holes in.
4. **The honeybee 10× weighting.** A tiny, concrete, memorable detail that proves domain thinking — not all "visits" are equal; a honeybee ≠ a random fly.
5. **"We count visits, not frames."** A crisp reframe that instantly communicates why naive approaches over-count, and shows we understood the *behaviour*, not just the pixels.
6. **Graceful degradation as a flex.** No GPU? runs on CPU. No API key? the assistant still answers from grounded data. No phone? the webcam works. It reads as "production maturity," which reads as "these people ship."
7. **Local roots, global relevance.** Baku pomegranate → pollinator crisis worldwide. Small, specific, and scalable — the storytelling sweet spot.

---

## Part C — Winning the two Special Mention votes (Student + Jury)
These are **story/effort/idea** votes, not rubric scores — emotion and memorability win.
- **The one-liner they'll remember and write on the ballot:** *"Bee-A-Hero — makes the invisible pollinator visible and proves its worth."*
- **The effort story (jury shout-out loves this):** trained our own detectors, built a two-pass tracker that survives occlusion, a real-time camera mode, a grounded assistant on open models, and honest science — a full Data+AI stack, shipped.
- **The emotional hook (student vote loves this):** pollinators are collapsing; we built the tool that lets a farmer *see and value* them. It's a project with a heart.
- **Make it easy to vote for us:** put the **project name + number + one line** on the last slide so anyone filling a ballot has the exact words: *"Bee-A-Hero (Web Dev + AI) — turns flower video into a pollination-value signal."*

---

## Part D — Scorecard self-check (rehearse against this)
| Criterion | Target | Our evidence | Risk if we skip it |
|---|---|---|---|
| Clarity | 10 | one arc, timed, callbacks | rambling → 6–8 |
| Delivery | 10 | all speak, Q&A bank, demo backup | one person dominates / demo dies → 3–5 |
| Problem | 10 | Baku grower, 45→68%, festival | generic "help bees" → 6–8 |
| Solution | 10 | lift model + shipped demo + **stated trade-offs** | no limits mentioned → capped at Good |

**Full jury target: 40/40.** Combined with a strong technical score, that's the winning formula.

## Part E — Three ways we could lose points (avoid these)
1. **Demo dies live** → always have the pre-recorded backup; never debug on stage.
2. **Over-claiming** ("we predict exact yield") → the jury punishes it under "trade-offs"; say *relative lift*.
3. **One person talks the whole time** → rehearse hand-offs; every member owns a beat and a Q&A lane.
