# QuarryFlow Crossing

QuarryFlow Crossing is a simulation of traffic buildup at a railway crossing
with adaptive vehicle release strategies after barrier reopening.

## Features

- Two-sided railway crossing simulation
- Mixed vehicle types (cars, bikes, auto-rickshaws)
- Driver aggressiveness modeling
- Policy comparison (free flow, alternating, adaptive)
- RandomForest-based policy selection
- Streamlit dashboard

## Project Layout

Planning/plan.md
src/quarryflow/
app/streamlit_app.py
scripts/
tests/
artifacts/

## Setup

pip install -r requirements.txt

## Run comparison

python scripts/compare_policies.py --scenario peak --seed 11

## Train model

python scripts/generate_training_data.py
python scripts/train_surrogate.py

## Launch dashboard

streamlit run app/streamlit_app.py
