# 01 · Problem & Vision (Speaker 1 — the hook)

## The problem
Pollinator activity is invisible to growers. They can't see how much bees actually contribute to their crop, so they can't value pollinator habitat, time insecticide spraying around foraging, or justify hive rental. Manual observation doesn't scale — a human can watch one flower, not an orchard, and not for 12 hours a day.

## What we built
A camera-based system that **automatically measures pollination activity** on pomegranate flowers and turns it into an economic signal:

> **Detect → Count → Trend → Report**, plus a pomegranate-specific yield extension.

- **Detect** flowers and the insects visiting them.
- **Count** genuine feeding *visits* (not raw detections), tracking each insect so it isn't double-counted.
- **Trend** visits over time / daypart / flower.
- **Report** the pollination **lift** and yield estimate, and answer plain-language questions via an AI assistant.

## The scientific thesis — "lift, not dependency" (this is what makes it defensible)
Pomegranate (*Punica granatum*) is **andromonoecious and self-fertile**: it sets fruit *without* insects. So a naive "more bees → more fruit" model is **biologically wrong** — and it's the first thing a reviewer will attack.

Instead we model the **marginal value** insects add, bounded by literature:

| Pollination mode | Fruit set | Source |
|---|---|---|
| Bagged (self-pollinated) | ~**45%** | Purdue/Morton horticulture |
| Cross-pollinated | ~**68%** | Purdue/Morton horticulture |
| Open field vs. control yield | 24.93 vs. 15.36 kg/plant | entomophily field study |

The insect contribution is the **bounded lift of +23 percentage points** — no more, no less. We report the **relative lift** insects add over a self-pollination baseline. That is a stronger, more honest claim than "we counted bees," and it's the core differentiator from a generic capstone.

## Why pomegranate, why Baku
- **Local relevance:** pomegranate is culturally and agriculturally iconic in Azerbaijan (the national Goychay Pomegranate Festival). Real local impact, not a toy problem.
- **Citable baselines:** the 45%→68% fruit-set numbers exist in the literature, so our model is anchored, not invented.
- **Phenology fits a project:** bloom is **May–June**, harvest **~170–180 days later** (autumn) — a clear, bounded data-collection window.

## Who it's for / impact
- **Growers:** quantify pollinator value → decide on hives, habitat, spray timing.
- **Researchers / agronomists:** an automated, reproducible pollination index.
- **Policy / conservation:** hard evidence for the economic value of pollinators.

## The 30-second demo narrative
"Upload a clip of a flower. The system finds the flower, tracks the bee, ignores the fly-throughs, counts the real landings, scores their pollination value (a honeybee is worth 10× a random fly), and tells you the pollination lift — then you can just *ask* the assistant 'how many pollinator visits did I get?' and it answers from your real numbers. Point a phone at a flower and it does the same thing live."

## Non-technical talking points
- We chose the **harder, more correct** science (marginal lift) over the flashier, wrong one (linear dependency).
- Everything is **one pipeline, one source of truth** — the demo, the CSVs, and the live camera can't disagree.
- It **degrades gracefully** — no GPU? runs on CPU. No API key? the assistant still answers from a grounded fallback. No phone? the live viewer uses the laptop webcam.
