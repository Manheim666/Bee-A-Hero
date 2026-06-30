# BEE_HERo - Option 3 Re-split Report

_Generated 2026-06-29 22:40:39_

## Inputs
- Pooled labeled images (train_mini+val): **151560**
- Unreadable/skipped: 0
- Classes: 2526

## De-duplication
- Exact duplicates dropped (kept 1 each): **22**
- Near-dup grouping: within-class, hamming <= 5
- Near-dup groups (observation units): **151525**
- Usable images after dedup: **151538**

## Final split (80/10/10, stratified, group-safe)
- train: **121226** (80.0%)
- val: **15157** (10.0%)
- test: **15155** (10.0%)

## Leakage verification
- Dup-groups spanning multiple splits: **0** (0 = no leakage, guaranteed by construction)
- Classes with no val sample: 0
- Classes with no test sample: 0

## Notes
- `public_test/` (500k) is unlabeled -> inference-only, excluded.
- Original `train_mini/` and `val/` folders are untouched; this
  re-split is expressed purely as file lists -> fully reversible.
