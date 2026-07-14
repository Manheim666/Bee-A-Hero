# 02 · Data Pipeline

## What data the system needs
Two very different kinds:
1. **Images with bounding boxes** to train the detectors (flowers, insects).
2. **Video** of insects on pomegranate flowers to test tracking + visit counting.

The scarce link is **video of insects on pomegranate flowers** — highly niche. Strategy: pretrain detectors on abundant image data, curate test video separately.

## Sources
| Source | What | Licence note |
|---|---|---|
| **iNaturalist Open Dataset** (~13M insect images, AWS Open Data) | insect pretraining | CC0 / CC-BY / CC-BY-NC — track per image |
| **iNaturalist competition sets** (iNat 2017–2021) | fine-grained taxonomy | curated |
| **GBIF occurrence + media API** | research-grade, region-filterable (filter to *Apis*, box near Caucasus) | republishes research-grade iNat; image cache ≤ 1200×1200 |
| **Flower boxes** | annotated ourselves in Roboflow / CVAT | Kaggle "flower" sets are *classification*, not detection — wrong framing |
| **Test video** | YouTube macro footage via `yt-dlp`; Zenodo/Dryad insect-tracking clips | scraped frames = **private testing only**, never redistributed |

## The critical caveat: iNaturalist is a *classification* dataset
iNat images are **one centered, human-framed organism, no boxes**. You **cannot** feed it to a detector as if it were a detection set — the framing distribution is wrong. Correct two-step use:
1. **Pretrain a backbone** on the iNat bee/insect subset → learn fine-grained insect features.
2. **Attach a detection head** and fine-tune on *boxed orchard frames*.

## The real risk: domain gap
An iNat bee (sharp, centered, fills the frame) looks nothing like a **12–20 px motion-blurred blob** crossing a fixed garden camera. Mitigations:
- **Small-object detection head** (`yolo26-p2.yaml`) — bees are often < 20 px.
- **Heavy augmentation:** motion blur, downscale, copy-paste real bee crops onto real flower backgrounds (synthetic positives).
- **A few hundred in-domain boxes** from our own footage. *A small in-domain set beats a large out-of-domain one* — highest-leverage labeling we can do.

## Splits & discipline (locked project rules)
- Fixed **train / val / test** splits; the test set is never touched during tuning.
- **Per-image licence tracking**; CC-BY-NC kept out of anything commercial; scraped frames never committed.
- Data prep runs on its own branch; raw data backed up before transforms.

## Validation artifact
A **held-out set of real clips hand-labeled with ground-truth visits.** This single artifact turns "we built a pipeline" into "we built a pipeline and measured it counts visits at X% precision." (See `08_Metrics_and_Evaluation.md`.)

## Likely questions
- *"Why not just train a detector on iNaturalist directly?"* → It's a classification set (no boxes, wrong framing). We pretrain a backbone on it, then fine-tune a detection head on boxed in-domain frames.
- *"How do you handle tiny, blurry bees?"* → P2 small-object head + motion-blur/downscale augmentation + in-domain labels.
- *"Licensing?"* → per-image licence tracked; research-grade CC media attributed; scraped YouTube frames are private-test only, never in the repo.
