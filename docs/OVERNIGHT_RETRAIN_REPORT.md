# Bee-A-Hero — Overnight Anti-FP Retrain Report

_Generated 2026-07-15 07:49_

Fine-tuned both detectors from their v2 `best.pt` with **1000 COCO negative images** (people/hands/indoor/objects, potted-plant/vase excluded) added as label-free backgrounds. FP measured on 199 held-out negatives.

| Model | mAP50 before | mAP50 after | FP imgs before | FP imgs after | FP boxes before → after | Deployed |
|---|---|---|---|---|---|---|
| flower | 0.806 | 0.796 | 27.6% | 1.0% | 66 → 2 | ✅ yes |
| insect | 0.664 | 0.663 | 41.2% | 1.0% | 108 → 2 | ✅ yes |

**Deploy gate:** ship only if mAP ≥ baseline−0.03 AND FP boxes at least halved.
Old weights backed up as `best_v2_prev.pt` next to each deployed model.
