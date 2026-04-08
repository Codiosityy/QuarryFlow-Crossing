# QuarryFlow Crossing - Assumptions & Problem Modelling

## Problem Statement Compliance (C-01, C-02, C-03)

This document explicitly states the assumptions and modelling decisions for the KnowledgeQuarry Data Science Challenge, Problem 01: The Bottleneck Problem.

---

## C-01: Single Bottleneck Type

**Constraint**: Focus on a single bottleneck type (e.g., lane merge, railway crossing).

**Implementation**: 
- **Type**: Urban railway crossing with barrier gate
- **Location**: `src/quarryflow/simulator.py:25`
- **Geometry**: Two opposing road approaches (left/right), one shared crossing box
- **Approach length**: 100 meters per side
- **Disorder zone**: 15 meters pre-gate
- **Crossing box**: 12 meters
- **Recovery zone**: 80 meters downstream

---

## C-02: Defined Metrics

**Constraint**: Optimize at least one of: Avg Waiting Time, Throughput, Congestion Length.

**Implementation** in `src/quarryflow/metrics.py:13-15`:

| Metric | Definition |
|-------|------------|
| `waiting_time` | Average time a vehicle spends queued or moving below 1 m/s |
| `throughput` | Vehicles cleared through crossing per unit time |
| `congestion_length` | Farthest upstream queued vehicle distance from barrier |

---

## C-03: Assumptions (Explicitly Stated)

### Traffic Rules Considered

1. **Barrier Behavior**: Trains trigger barrier closure; vehicles queue behind barrier until reopen
2. **Right-of-Way**: Vehicles in crossing box have priority; others yield
3. **Lane Discipline**: Vehicles attempt to maintain position but may encroach laterally
4. **Speed Limits**: Reduced speed near gate (0.5x), normal elsewhere

### Driver Behaviour Model

Six profiles in `src/quarryflow/config.py:40-119`:

| Profile | Key Behavior |
|---------|-------------|
| `cautious` | Large gaps, slow restart, low aggression |
| `compliant` | Moderate gaps, cooperative merging |
| `opportunistic` | Smaller gaps, moderate encroachment |
| `assertive` | Active lane negotiation |
| `aggressive` | Minimum gaps, fast restart |
| `reckless` | No gaps, aggressive rushing |

**Behaviour Parameters**:
- `min_gap`: Minimum following distance (meters)
- `encroachment_bias`: Likelihood of crossing centerline (0-1)
- `restart_gain`: Acceleration multiplier after gate opens (0.8-1.3)
- `gate_rush_bias`: Tendency to speed up near gate (0-1)

### Data Source

- **Type**: Simulated (not observed)
- **Generation**: `scripts/generate_training_data.py`
- **Model Training**: `scripts/train_surrogate.py`
- **Validation**: Cross-validation with holdout set

---

## Problem 02: Learning Under Constraints (Secondary)

The system also addresses Problem 02 with adaptive policy selection:

- **Agent Types**: Different driver profiles (6 types)
- **Resource Constraints**: Episode time limits, vehicle capacity
- **Learning**: Surrogate model predicts outcomes; hybrid controller adapts

---

## Evaluation Evidence

| Criterion | Evidence Location |
|-----------|----------------|
| Clarity of modelling | This document, README, config.py |
| Effectiveness | `scripts/compare_policies.py` output |
| Traffic behaviour | 6 driver profiles in config.py |
| Learning | Surrogate model in artifacts/models/ |
| Technical depth | Hybrid controller in hybrid.py |

---

*Last updated: April 2026*
*For KnowledgeQuarry CONVOKE 8.0 Data Science Challenge*