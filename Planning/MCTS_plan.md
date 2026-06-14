# MCTS Rollout Agent — Implementation Plan

This document outlines the step-by-step plan for building **Component A: The MCTS Rollout Agent**, which will replace the static Gradient Boosting Regressor (GBR) policy selector with a forward-looking simulation-based search.

## 1. Phase 1: Benchmarking & Profiling (The Budget)
Before writing the search algorithm, we must know how fast the existing `RailwayCrossingSimulator` can run. MCTS requires running thousands of future ticks per decision.

*   **Task:** Create `scripts/benchmark_sim.py`.
*   **Action:** Run a tight loop of `simulator.step()` for 1,000, 5,000, and 10,000 ticks.
*   **Goal:** Measure ticks-per-second (TPS). This will define our compute budget:
    *   $K$: Number of ticks to simulate into the future per rollout (e.g., 300 ticks = 30 seconds of simulation time).
    *   $M$: Number of Monte Carlo rollouts to average per candidate action (e.g., 10 rollouts).
    *   If TPS is too low, we will need to create a `FastForwardSimulator` that strips out heavy microscopic physics (like lateral filtering) during the search phase.

## 2. Phase 2: State Cloning & Fast Reset
MCTS works by testing future actions. To do this, it must simulate the future *without* permanently altering the present state of the crossing.

*   **Task:** Implement a highly efficient state copying mechanism.
*   **Action:** Add a `clone()` or `save_state()` / `restore_state()` method to `RailwayCrossingSimulator`.
*   **Goal:** `copy.deepcopy()` is notoriously slow in Python. We need to ensure we can save and restore the queue sizes, vehicle positions, and barrier states in less than 1 millisecond so the search loop doesn't bottleneck here.

- [x] **Phase 1: Performance Baseline (Ticks-Per-Second Test)**
  - Create `scripts/benchmark_sim.py`.
  - Instantiate `RailwayCrossingSimulator` using the `peak` scenario.
  - Run a tight loop of `simulator.step(FreeFlowPolicy())` for 10,000 ticks.
  - Calculate TPS to determine our compute budget ($K \times M$).

- [x] **Phase 2: State Cloning Pipeline**
  - Use `simulator.clone()` to capture macroscopic traffic state.
  - Benchmark deepcopy performance (~2.3ms).

- [x] **Phase 3: The Search Core (Adapting Critical-Mass)**
  - Create `src/quarryflow/mcts.py` with `MCTSRolloutEngine`.
  - Implement caching via a Transposition Table string hash (`_hash_state`).

- [x] **Phase 4: Multi-Objective Evaluation**
  - Leverage `simulator.evaluate_horizon()` for queue length and waiting time calculation.

- [x] **Phase 5: Dashboard Integration**
  - Expose `MCTSRolloutPolicy` in `src/quarryflow/dashboard_data.py`.
  - Make sure the Streamlit app runs it alongside the ML baseline.

---

### Open Questions / Next Steps
1. Does the current `RailwayCrossingSimulator` use classes that are easily serializable (for the Transposition table hash)?
2. Are we keeping the rollout strictly single-threaded, or should we use Python's `multiprocessing` to run the $M$ rollouts in parallel?
