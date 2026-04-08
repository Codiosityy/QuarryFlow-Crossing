# QuarryFlow Crossing

## Overview

Simulation of vehicle queues at a railway crossing and evaluation of
release strategies after barrier reopening.

## Components

Traffic Simulator
- two opposing approaches
- shared crossing box
- mixed vehicle behavior

Metrics
- waiting time
- throughput
- queue length
- clearance time

Control Policies
- free release
- alternating release
- adaptive staged release

Model

RandomForestRegressor trained on:

- queue length
- mean speed
- vehicle mix
- aggressiveness mix
- time since open

Predicts:

- waiting time
- throughput
- congestion length

## Structure

src/quarryflow/
simulator.py
policy.py
metrics.py
model.py

app/
streamlit_app.py
