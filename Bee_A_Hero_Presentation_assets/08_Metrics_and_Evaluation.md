# 08 · Metrics & Evaluation ("How do you know it works?")

Match the metric to the stage — have an answer for every one.

| Stage | Primary metric | Our number / target |
|---|---|---|
| Flower detection | mAP@0.5, mAP@0.5:0.95 | **0.808** (mAP@0.5); > 0.85 realistic for large static flowers |
| Insect detection | precision / recall, mAP@0.5 | **0.669**; recall-first (a missed bee = a missed visit) |
| Honeybee sub-classifier | accuracy | **0.978** |
| Single-cam tracking | **HOTA**, MOTA, IDF1, **ID switches** | HOTA is the modern standard; ID switches drive double-counting |
| **Visit counting** | visit precision/recall vs. hand-labeled clips | **the core validation** |
| Yield model | fit residuals, relative-lift index | report **relative lift**, not absolute kg |

## Why these metrics
- **Recall > precision for insects:** a missed bee is a lost visit; a false bee is caught downstream by the FP gates and the dwell/track filters.
- **HOTA over MOTA for tracking:** HOTA balances detection + association in one number and correlates with human judgment of track quality. ID-switch rate is what inflates visit counts.
- **Visit precision/recall is THE number:** the whole product is "count real visits," so we validate it directly against hand-labeled clips (visit precision/recall, ID-switch rate).
- **Yield = relative index, not kg:** you can't calibrate absolute tonnage in one season; a validated relative pollination index is defensible.

## Tooling
- **TrackEval** for HOTA/MOTA/IDF1 (the standard the honeybee-tracking paper used).
- A **held-out set of real clips hand-labeled with ground-truth visits** — the single artifact that turns "we built a pipeline" into "we measured it counts visits at X% precision."

## Behaviour we can demonstrate live (qualitative validation)
- Occlusion re-link: a bee dips behind a petal and is still **one** visit.
- Fly-through rejection: a < 2 s pass doesn't count.
- FP gates: pointing the webcam at a room yields **zero** detections (no "whole screen is a flower").
- Consistency: the same clip gives the same numbers on the website and in the CSVs.
- Verified on test clips: single-flower videos report **1 flower**; honeybee landings are counted and weighted 10×.

## Honest limitations (say them before they're asked)
- The insect **classifier** confuses some honeybees with fly/beetle (a model-accuracy limit); the cumulative type-vote + honeybee sub-classifier recover most.
- The **flower detector** occasionally fails to fire on a real bloom (then landings fall back to inferred synthetic flowers) — a detector limit, not a logic bug.
- **Absolute yield is not calibrated** — we report relative lift by design.
- **Multi-camera** is designed, not shipped (single camera per viewer today).
