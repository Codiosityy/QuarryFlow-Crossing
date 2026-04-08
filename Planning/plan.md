# QuarryFlow Crossing

## 1. Project Overview

### Working Title
QuarryFlow Crossing: Railway-Crossing Surge Optimizer

### One-Line Pitch
An ML-driven, behavior-aware traffic simulation and control system for urban railway crossings that reduces waiting time and improves throughput by coordinating vehicle release after barrier reopening.

### Core Thesis
The train causes the interruption, but human behavior before and after gate reopening determines how severe congestion becomes. If we model that behavior well, we can choose better release strategies and clear the crossing faster.

## 2. Hackathon Fit

### Track
ML Engineering Track

### Why This Fits
- Predicts congestion and flow outcomes from simulated state.
- Simulates traffic dynamics in a constrained bottleneck.
- Optimizes control strategies using a data-driven policy selector.
- Produces a live demo that judges can understand quickly.

### Single Bottleneck Scope
Urban railway crossing with two opposing road approaches and one shared crossing box.

### Out of Scope
- Physical infrastructure redesign
- Legal or enforcement systems
- Hardware implementation
- Multiple crossings or city-scale coordination

## 3. Problem Statement

When the railway barrier is down, vehicles accumulate near the gate. After it reopens, drivers often accelerate aggressively, weave laterally, and compete for the same constrained space. This creates post-opening disorder, delays clearance, and can extend congestion far beyond the actual train closure duration.

The goal is to simulate that behavior, analyze why it causes inefficiency, and apply a control policy that improves:
- Average waiting time
- Throughput
- Congestion length

Supporting metrics:
- Post-open clearance time
- Conflict count
- Crossing-box occupancy risk
- Fairness between approaches

## 4. Judge Strategy

### Impact-Focused Judges
Desired impression:
- Real problem
- Immediate visual understanding
- Clear before/after improvement
- Useful in real urban settings

What they need to see:
- Barrier down -> queues build
- Barrier opens -> baseline chaos
- Adaptive control -> faster, cleaner clearance
- Simple metric cards showing improvement

### Technical Judges
Desired impression:
- Solid simulation logic
- Clear feature engineering
- Legitimate control design
- ML is used meaningfully, not decoratively

What they need to see:
- Event-driven simulator
- Mixed-traffic driver behavior modeling
- Defined state, action, and evaluation metrics
- Baseline comparisons and ablation
- Clear assumptions and limitations

### Anchor Message
The barrier creates the pause; disorderly human release creates the lasting jam.

## 5. Solution Shape

The project will have four main layers:

1. Traffic Simulator
- Event-driven microscopic railway-crossing simulation
- Two opposing approaches
- Shared crossing box
- Mixed vehicle classes and behavior profiles

2. Metrics and Feature Extraction
- Queue length, mean speed, density, disorder score, conflicts, clearance time
- Snapshot generation for training and policy selection

3. ML Surrogate + Policy Engine
- Train a lightweight predictive model on simulated data
- Estimate short-horizon outcomes under candidate control actions
- Select the best action every few seconds after reopening

4. Demo Dashboard
- Live top-down animation
- Barrier status
- Baseline vs adaptive comparison
- Metric panels and control explanation

## 6. Technical Design

### 6.1 Scenario Defaults
- Time step: 0.5 seconds
- Episode duration: 10 to 15 minutes
- Approach length per side: 100 meters
- Pre-gate disorder zone: 15 meters
- Crossing box length: 12 meters
- Downstream recovery zone: 80 meters

### 6.2 Traffic Defaults
- Cars: 60%
- Bikes: 25%
- Auto-rickshaws: 15%

### 6.3 Driver Profiles
- Compliant
  - Higher lane discipline
  - Lower restart aggressiveness
  - Larger accepted gaps
- Opportunistic
  - Moderate disorder
  - Medium restart aggressiveness
  - Medium gap acceptance
- Aggressive
  - High queue encroachment
  - Fast restart
  - Small gap acceptance

### 6.4 Train Event Defaults
- Repeated barrier closures per episode
- Closure duration varies by scenario preset
- Presets:
  - Light
  - Peak
  - Chaotic reopening

## 7. Core Metrics

### Required Competition Metrics
- Average waiting time
- Throughput
- Congestion length

### Additional Internal Metrics
- Post-open clearance time
- Conflict count
- Disorder index
- Crossing-box occupancy risk
- Fairness gap between approaches

### Definitions
- Average waiting time: average time a vehicle spends queued or moving below a low-speed threshold
- Throughput: vehicles cleared through the crossing per unit time
- Congestion length: farthest upstream queued vehicle distance from the barrier
- Clearance time: time from barrier reopening until the queue dissipates below a threshold
- Disorder index: weighted score from lateral crowding, failed launch attempts, and near-gate density

## 8. Control Policies

### Baseline Policies
- Free-for-all reopening
- Static alternating release

### Final Policy
Adaptive staged release

### Candidate Actions
- Immediate free release
- 4 second left / 4 second right alternating bursts
- 6 second left / 6 second right alternating bursts
- Queue-weighted directional priority
- Short settling delay before staged release

### Policy Objective
Maximize throughput while minimizing waiting time and congestion length, with penalties for:
- Starving one side for too long
- High disorder
- Elevated crossing-box occupancy risk

## 9. ML Layer

### Goal
Predict near-term traffic outcome under each candidate action so the controller can choose the best action after barrier reopening.

### Model Type
Lightweight tabular surrogate model.

Recommended first implementation:
- scikit-learn HistGradientBoostingRegressor or RandomForestRegressor

Optional later alternatives:
- XGBoost or LightGBM if installation and time allow

### Prediction Horizon
Next 90 seconds after a state snapshot

### Input Features
- Queue length by side
- Mean speed
- Barrier state or time-since-open
- Disorder index
- Vehicle mix
- Driver aggression mix
- Crossing-box occupancy
- Recent conflict count
- Candidate policy action

### Target Outputs
- Predicted average waiting time
- Predicted throughput
- Predicted congestion length

## 10. Core Interfaces

### ScenarioConfig
Stores:
- Geometry
- Time step
- Episode duration
- Traffic composition
- Driver profile mix
- Train schedule
- Allowed control actions

### VehicleAgent
Stores:
- Vehicle class
- Dimensions
- Speed profile
- Approach side
- Driver profile
- Current position
- Current state

### CrossingStateSnapshot
Stores:
- Current time
- Barrier state
- Queue length by side
- Disorder score
- Crossing-box occupancy
- Downstream blockage state
- Active control action

### PolicyAction
Stores:
- Control mode
- Directional release pattern
- Settling delay
- Any priority weighting

### EpisodeMetrics
Stores:
- Average waiting time
- Throughput
- Congestion length
- Clearance time
- Conflict count
- Occupancy risk
- Fairness gap

## 11. Proposed Repository Structure

```
Planning/
  plan.md
src/
  quarryflow/
    __init__.py
    config.py
    types.py
    simulator.py
    behaviors.py
    metrics.py
    policy.py
    model.py
    scenarios.py
    dashboard_data.py
app/
  streamlit_app.py
scripts/
  run_baseline.py
  generate_training_data.py
  train_surrogate.py
  compare_policies.py
artifacts/
  data/
  models/
  reports/
tests/
  test_simulator.py
  test_metrics.py
  test_policy.py
requirements.txt
README.md
```

## 12. Coding Requirements

### Language and Runtime
- Python 3.13 is available in the current environment

### Required Packages
- numpy
- pandas
- scikit-learn
- streamlit
- plotly

### Current Environment Check
Available:
- numpy
- pandas

Missing during last check:
- scikit-learn
- streamlit
- plotly

### Engineering Rules
- Keep modules small and focused
- Use dataclasses for core entities where helpful
- Keep simulation deterministic when given a random seed
- Separate simulation logic from UI logic
- Save experiment outputs to artifacts/
- Avoid hardcoding visualization assumptions into the simulator

### Logging and Reproducibility
- Use fixed random seeds for demos
- Persist trained models
- Save policy comparison outputs as CSV or JSON
- Keep scenario presets named and reproducible

## 13. Dashboard Requirements

The live demo must include:
- Railway crossing animation
- Barrier open/closed status
- Queue length and throughput cards
- Baseline vs adaptive comparison
- Current policy explanation in plain language
- Preset selector for light, peak, and chaotic reopening

The dashboard should be simple enough for judges to understand in under one minute.

## 14. Testing Requirements

### Functional Tests
- Simulator runs full episodes without crashes
- Barrier events correctly block movement
- Queueing forms on both sides during closure
- Release logic changes discharge behavior after reopening

### Behavior Tests
- Aggressive driver mixes increase disorder relative to compliant mixes
- Longer closures create larger queues and longer clearance times

### Policy Tests
- Adaptive policy should not perform worse than baseline in light traffic
- Adaptive policy should improve waiting time or clearance time in peak and chaotic cases
- Fairness checks prevent one-sided starvation

### Demo Tests
- Light preset works quickly
- Peak preset shows measurable difference
- Chaotic reopening preset creates a dramatic baseline-vs-adaptive visual

## 15. Build Order

### Phase 1: Core Simulation
- Implement config and types
- Implement vehicle generation
- Implement barrier state changes
- Implement movement and queue logic
- Implement metric collection

### Phase 2: Policy Layer
- Implement baseline policies
- Implement adaptive action interface
- Compare baseline outputs

### Phase 3: ML Layer
- Generate training data from simulation sweeps
- Train surrogate model
- Integrate model into policy selection

### Phase 4: Demo Layer
- Build Streamlit dashboard
- Add scenario presets
- Add baseline-vs-adaptive controls
- Export demo-ready charts

### Phase 5: Polish
- Tune parameters for strong visible improvements
- Add README
- Add assumptions page or section
- Prepare pitch screenshots and metrics

## 16. Deliverables

### Required Deliverables
- Working simulation code
- Trained or trainable surrogate model
- One dashboard app
- One reproducibility script or notebook
- README
- Metrics summary for presentation

### Presentation Deliverables
- Before/after comparison screenshots
- One short architecture diagram
- One assumptions slide
- One metrics slide

## 17. Risks and Mitigations

### Risk: Scope becomes too large
Mitigation:
- Keep to one crossing
- Keep action space small
- Use only three scenario presets

### Risk: ML feels unnecessary
Mitigation:
- Show baseline vs heuristic vs adaptive comparison
- Use feature importance or clear control explanations

### Risk: Simulation feels unrealistic
Mitigation:
- State assumptions explicitly
- Use mixed vehicle classes
- Include disorder features tied to visible behavior

### Risk: Dashboard takes too long
Mitigation:
- Keep one-screen layout
- Use precomputed or efficiently computed episode outputs

## 18. Competitive Positioning

### Strengths
- Strong real-world relevance
- Lower overlap risk than generic traffic topics
- Good visual story for judges
- Strong technical narrative when implemented well

### Weaknesses
- Simulation-only credibility risk
- Railway crossings are harder to model than simple merges
- ML must justify its presence

### Novelty View
- High hackathon novelty
- Moderate technical originality

### Patentability View
- Weak as pure simulation software
- Better if later framed as a deployable control system with sensing and physical outputs

## 19. Success Criteria

The project is successful if:
- It clearly demonstrates the railway-crossing bottleneck problem
- It shows measurable gains over baseline
- Judges understand the value in under one minute
- Technical judges can see the logic behind the controller
- The demo runs reliably during presentation

## 20. Immediate Next Coding Tasks

1. Create repository scaffolding
2. Add requirements.txt
3. Implement core dataclasses and config
4. Implement the railway-crossing simulator
5. Add baseline metrics and reporting
6. Add policy comparison script
7. Add surrogate training pipeline
8. Add Streamlit dashboard
9. Validate presets and presentation outputs