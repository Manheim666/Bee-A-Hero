# Bee-A-Hero — Presentation & Q&A Pack

> Everything needed for the pitch **and** the questioning phase: technical, algorithmic, scientific, product, and non-technical. Grounded in the actual codebase, not slideware.

**Project in one line:** a computer-vision system that watches pomegranate flowers, detects & tracks visiting insects without double-counting, classifies each as pollinator / non-pollinator, counts real feeding *visits*, and converts visitation into a defensible **pollination-lift → yield** estimate — with a live camera mode and an AI assistant grounded in the results.

**Setting:** Baku, Azerbaijan (40.4°N, 49.9°E) · Crop: *Punica granatum* (pomegranate) · Built July 2026.

---

## Files in this pack

| File | Owner / focus | Use it for |
|---|---|---|
| `01_Problem_and_Vision.md` | Speaker 1 — the hook | Why this matters, the "lift not dependency" thesis, local impact |
| `02_Data_Pipeline.md` | Data | Sourcing, annotation, splits, the domain-gap problem |
| `03_CV_Engine.md` | CV (the core) | Detection, tracking, the visit state machine, every veto/NMS/stitch |
| `04_ML_Pollination_Yield.md` | ML / science | The four-link math, constants, worked example, validation |
| `05_LLM_Assistant.md` | Generative AI | Gemini/Hugging Face providers, grounding, decoding params |
| `06_WebApp_and_LiveCamera.md` | Full-stack | Upload flow, dashboard, live DroidCam + rolling landing log |
| `07_Architecture_Deployment.md` | Systems | Services, Docker, ports, security, data flow |
| `08_Metrics_and_Evaluation.md` | Rigor | Metric per stage, targets, how we know it works |
| `09_QA_Anticipated_Questions.md` | **Everyone** | The question bank — technical **+ business/jury** — with model answers |
| `10_PRESENTATION_SCRIPT.md` | **Everyone** | ⭐ The clear path: narrative arc, 4-speaker timing, demo choreography, Q&A routing, checklist |
| `11_RUBRIC_ALIGNMENT_and_Creative.md` | **Everyone** | ⭐ How we hit 9–10 on every jury criterion + creative hooks + special-mention strategy |
| `COMBINED_Bee_A_Hero.md` | — | One compact file with the whole story |

## How this pack scores full marks (jury = 30% of grade)
Jury 40 pts = **Clarity /10 · Delivery /10 · Problem /10 · Solution /10**. Winner = `(Technical/100)×0.7 + (Jury/40)×0.3`.
- **Clarity & Delivery** → `10_PRESENTATION_SCRIPT.md` (one arc, timed, all 4 speak, demo backup, Q&A routing).
- **Problem & Solution** → `01_Problem_and_Vision.md` (specific, evidence-backed) + the honest **trade-offs/limits** that unlock Excellent (`11_…` Part A.4).
- **Special-mention votes** (student + jury) → the creative hooks + one-liner in `11_…` Parts B–C.
Read `10` and `11` first — they turn the technical files into a winning pitch.

## Headline numbers to have on the tip of your tongue

- **Flower detection:** mAP@0.5 = **0.808**
- **Insect detection:** mAP@0.5 = **0.669** (recall-first — a missed bee is a missed visit)
- **Honeybee sub-classifier:** accuracy = **0.978**
- **Pollination-lift band:** self-set **45%** → cross-set **68%** → the entire insect contribution is a **bounded +23 pp**
- **A "visit"** = a tracked insect on a flower for **≥ 2 s** (feeding, not a fly-through)
- **Honeybee pollination weight = 10×** a generic bee (butterfly 2×, fly/beetle 0.5×, bug 0.2×)

## The three sentences that win the room

1. "We don't claim *more bees = proportionally more fruit* — that's biologically wrong for a self-fertile crop. We model the **marginal lift** insects add over a 45% self-pollination floor, capped at a 68% ceiling."
2. "We count **feeding visits, not frames** — a state machine with a 2-second dwell gate and occlusion re-linking, so a bee dipping behind a petal is counted **once**."
3. "Same pipeline, one source of truth: the website upload, the offline CSVs, and the live camera all run the **identical** detector + tracker + landing logic — the numbers can't diverge."
