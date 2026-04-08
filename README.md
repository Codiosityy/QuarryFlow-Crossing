# QuarryFlow Crossing

QuarryFlow Crossing is a microsimulation framework for modelling and optimising vehicle
traffic at railway level crossings, with a focus on the high-disorder discharge phase
typical of emerging-market environments (mixed cars, bikes, and auto-rickshaws).

## Documentation

📚 **[Full Documentation Wiki](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki)** — Complete documentation with navigation

The wiki contains comprehensive documentation including:

- [Project Overview](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/00-project-overview) — Architecture, subsystems, and design decisions
- [Core Simulation Engine](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/01-simulation-engine) — Simulator lifecycle, vehicle models, scenarios
- [Control Policies](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/02-control-policies) — Baseline, adaptive, and hybrid policies
- [Machine Learning](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/03-machine-learning) — Surrogate models and LinUCB bandit controller
- [Training Pipeline](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/04-training-and-evaluation-pipeline) — Data generation, training scripts, benchmarking
- [Streamlit Dashboard](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/05-streamlit-dashboard) — Interactive visualisation and reporting
- [Testing](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/06-testing) — Test suite and patterns
- [Glossary](https://github.com/Codiosityy/QuarryFlow-Crossing/wiki/08-glossary) — Key terms and definitions

## Features

- Two-sided railway crossing simulation
- Mixed vehicle types (cars, bikes, auto-rickshaws)
- Driver aggressiveness modelling
- Policy comparison (free flow, alternating, adaptive)
- RandomForest-based surrogate policy selection
- Streamlit dashboard with decision-rationale inspection

## Project Layout

- Planning/plan.md
- src/quarryflow/       # Core simulation package
- app/streamlit_app.py  # Interactive dashboard
- scripts/              # Training, generation, and benchmarking scripts
- tests/                # Test suite
- artifacts/            # Persisted models, data, and reports


## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Run policy comparison:**
```bash
python scripts/compare_policies.py --scenario peak --seed 11
```

**Train surrogate model:**
```bash
python scripts/generate_training_data.py
python scripts/train_surrogate.py
```

**Launch dashboard:**
```bash
streamlit run app/streamlit_app.py
```
