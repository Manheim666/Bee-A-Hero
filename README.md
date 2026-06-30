# Bee-A-Hero

**Bee-A-Hero** is a computer-vision project for detecting, tracking, and counting
insects — with a primary focus on **bees** — and classifying them into
**pollinators** and **non-pollinators**. Starting from object tracking, the project
grows into a quantitative tool that links insect activity to **pollination rate**,
and, as an optional goal, to the **economic return** of the resulting harvest.

---

## Overview

At its core, the system answers three questions from video and image streams:

1. **How many?** — count insects present in a scene over time.
2. **What kind?** — separate bees from other insects, then split insects into
   pollinators and non-pollinators.
3. **So what?** — relate the counts and types, together with **time of day** and
   **season**, to the pollination rate of the observed plants.

The project begins as an **object-tracking** problem and is extended with data
research, ML modeling, and LLM-based reporting on top of the tracking core.

---

## Goals

### Primary
- Real-time / batch **detection and tracking** of insects in the field.
- **Counting** unique individuals (avoiding double-counting across frames).
- **Classification** of insects: bee vs. non-bee, then pollinator vs.
  non-pollinator.

### Research extensions
- **3D modeling & multi-camera fusion** — combine several camera views so the same
  insect is not tracked and counted twice across overlapping fields of view.
- **Pollination modeling** — derive relations and formulas connecting
  *insect count × insect type × time of day* to the **pollination rate**.

### Optional
- **Economic modeling** — connect pollination rate to the **increase in income**
  from selling products after harvest.

---

## Seasonal & Daily Time-Series Challenges

Insect activity is not constant — it is a time series with strong structure that the
models and formulas must account for:

- **Daily cycles.** Foraging activity shifts with time of day (morning vs. midday vs.
  evening), temperature, and light. The same plot yields very different counts at
  different hours, so counts must always be timestamped and normalized against the
  daily activity curve.
- **Seasonal cycles.** Species presence, bloom periods, and pollinator abundance vary
  across the season. A model trained on one period may not transfer to another, and
  pollination rate must be interpreted relative to the bloom stage.
- **Weather & noise.** Wind, rain, and cloud cover introduce gaps and irregular
  sampling, making the series uneven and noisy.
- **Drift.** Camera placement, plant growth, and changing backgrounds shift the data
  distribution over the season and require periodic re-validation.

These factors mean the count → type → pollination-rate relationship is
**time-dependent**, and any income formula built on top inherits the same seasonal
and daily variability.

---

## Repository Structure

```
Bee-A-Hero/
├── data/                       # Datasets (large raw data is git-ignored)
│   ├── raw/                    # Source data
│   │   ├── Flower/             # Flower imagery
│   │   └── iNaturist/          # iNaturalist dataset (contents git-ignored)
│   ├── interim/                # Intermediate, partially processed data
│   └── processed/              # Final, model-ready data
│
├── src/                        # Source code
│   ├── config.py               # Central configuration
│   ├── data_pipeline/          # Data ingestion & preparation
│   │   └── ingest.py
│   ├── cv_engine/              # Computer-vision core
│   │   ├── detect.py           # Insect detection
│   │   └── track.py            # Multi-frame / multi-camera tracking
│   ├── ml_models/              # Model training & classification
│   │   └── train.py
│   └── llm_reporting/          # LLM-based reporting & summaries
│       └── generate.py
│
├── notebooks/                  # Exploration & prototyping
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_cv_tests.ipynb       # CV experiments
│   ├── 03_ml_prototypes.ipynb  # ML prototyping
│   └── 04_llm_prompts.ipynb    # LLM prompt experiments
│
├── scripts/                    # Helper scripts
│   ├── setup_env.sh            # Environment setup
│   └── run_pipeline.sh         # End-to-end pipeline runner
│
├── tests/                      # Test suite
│   ├── test_cv.py
│   └── test_data.py
│
├── Dockerfile                  # Container image
├── docker-compose.yml          # Service orchestration
├── Makefile                    # Common tasks
├── pyproject.toml              # Project metadata & dependencies
├── requirements.txt            # Python dependencies
└── .env.example                # Example environment variables
```

---

## Getting Started

> The pipeline is under active development; the steps below describe the intended
> workflow as the scaffolding is filled in.

```bash
# 1. Set up the environment
bash scripts/setup_env.sh

# 2. Configure environment variables
cp .env.example .env        # then edit .env

# 3. Run the end-to-end pipeline
bash scripts/run_pipeline.sh
```

Containerized usage is supported via `Dockerfile` and `docker-compose.yml`.

---

## Team

| Member             | Role |
|--------------------|------|
| **Asif Habilov**   | Team lead — planning & tasking. Also contributes to data research, ML engineering, computer vision, and QA. Leads research on **3D modeling, multi-camera fusion** (avoiding duplicate insect tracking), and the **count × type × time → pollination-rate** relations, including the optional **pollination → income** modeling. |
| **Raul Ibrahimov** | Data research and ML engineering. |
| **Narmin Dirayeva**| LLM and ML engineering. |
| **Khaver**         | Data and LLM. |

---

## Status

Early stage. The repository structure and module scaffolding are in place; detection,
tracking, classification, pollination modeling, and reporting components are being
implemented incrementally.
