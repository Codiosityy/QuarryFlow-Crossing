# QuarryFlow Crossing

QuarryFlow Crossing is a behavior-aware railway-crossing traffic simulator and adaptive control demo built for the KnowledgeQuarry hackathon.

## What It Does

- Simulates two-sided mixed traffic at an urban railway crossing.
- Models cars, bikes, and auto-rickshaws with cautious, compliant, opportunistic, assertive, aggressive, and reckless driver behavior.
- Compares `free-flow`, `static alternating`, and `adaptive staged-release` control policies.
- Trains a surrogate model that predicts short-horizon traffic outcomes for candidate actions.
- Learns research-driven features such as wrong-side queueing, dilemma-zone pressure, restart delay, and idling waste.
- Provides a Streamlit dashboard with judge-facing tabs, replay visuals, and technical-depth panels.
- Generates pitch-ready markdown briefs for reporting and presentation.

## Project Layout

```text
Planning/plan.md
src/quarryflow/
app/streamlit_app.py
scripts/
tests/
artifacts/
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run A Quick Comparison

```powershell
python scripts/compare_policies.py --scenario peak --seed 11 --model artifacts\models\surrogate.pkl
```

## Generate Training Data

```powershell
python scripts/generate_training_data.py
python scripts/train_surrogate.py
```

## Train The Hybrid Controller

Fast, hackathon-friendly training:

```powershell
python scripts/train_hybrid_controller.py --profile fast --scenarios light peak chaotic --stage-passes 1 --n-models 2
```

Fuller offline training:

```powershell
python scripts/train_hybrid_controller.py --profile full
```

## Generate A Pitch Brief

```powershell
python scripts/generate_pitch_report.py --scenario peak --seed 11
```

## Launch The Dashboard

```powershell
streamlit run app/streamlit_app.py
```

## Current Notes

- The simulator is deterministic for a given seed.
- If `scikit-learn` is not installed, the surrogate model falls back to a NumPy ridge regressor.
- The main judging flow works best when you start with `peak` and then switch to `chaotic`.
- `artifacts/models/hybrid_controller.json` is produced by the hybrid training script and enables the full `Hybrid Adaptive` dashboard mode.
- Generated reports are written under `artifacts/reports/`.
