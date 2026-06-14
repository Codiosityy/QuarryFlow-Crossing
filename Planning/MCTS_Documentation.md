# MCTS Rollout Agent — Technical Documentation

This document serves as a historical record and technical reference for the implementation of the Monte Carlo Tree Search (MCTS) Rollout Agent in the QuarryFlow-Crossing project. It outlines the architectural changes, performance bottlenecks encountered, and the specific solutions applied to stabilize the system.

## 1. Objective
The original project utilized a static ML model (`SurrogateModel` and `BootstrapSurrogateEnsemble`) and a Reinforcement Learning agent (`LinUCBResidual`) to predict the best traffic light / barrier release policy based on the current state. 

Our goal was to implement a **forward-looking MCTS Rollout Policy** (`MCTSRolloutPolicy`). Instead of simply guessing the best action mathematically, MCTS physically simulates the consequences of all possible actions seconds into the future, evaluates the resulting traffic congestion, and picks the action that results in the least delay.

## 2. Changes Made

### Core Components
- **`src/quarryflow/mcts.py`**: Created to house the `MCTSRolloutPolicy` engine. It handles tree search, simulation cloning, and state hashing.
- **`src/quarryflow/dashboard_data.py`**: Updated to inject the `MCTSRolloutPolicy` into the Streamlit evaluation suite.
- **`src/quarryflow/reporting.py`**: Updated report parsing logic to recognize `"MCTS Rollout"` instead of the obsolete `"Hybrid Adaptive"`.

### Obsolete Code Purge
Because the MCTS Rollout proved highly effective (and since the hackathon explicitly moves away from standard ML toward agentic simulation), we permanently deleted the legacy Reinforcement Learning components:
- Deleted `BootstrapSurrogateEnsemble` from `model.py`.
- Deleted `LinUCBResidual` from `hybrid.py`.
- Deleted `HybridAdaptivePolicy` from `policy.py`.
- Deleted legacy scripts (`train_hybrid_controller.py`, `evaluate_hybrid_controller.py`, etc.).

## 3. Problems Encountered & Solutions

### Problem 1: State Cloning (Deepcopy) Overhead
**Issue:** MCTS works by simulating the future. To do this, it must create a "save state" of the current crossing, run a simulation forward, and then revert back to the save state to test the next action. In Python, using `copy.deepcopy()` on the entire `RailwayCrossingSimulator` object was incredibly slow (~2.3ms per copy), bottlenecking the search.
**Fix:** We avoided cloning the entire simulator. Instead, we implemented a lightweight `simulator.clone()` method that only copies the raw data structures (vehicles, queues, and barrier states). 

### Problem 2: Transposition Table Hashing
**Issue:** MCTS uses a "Transposition Table" to cache states it has already evaluated, saving compute time. However, generating a unique string hash of the entire simulator state at every tick was computationally expensive.
**Fix:** We utilized the existing `CrossingStateSnapshot` mechanism. Instead of hashing raw memory, we extracted macroscopic metrics (`queue_lengths`, `disorder_index`, `occupancy_risk`) and hashed a rounded tuple of these values. This provided a fast, highly accurate state identifier for the cache.

### Problem 3: Simulation Time (The "1-Minute Delay")
**Issue:** Even with optimized cloning and hashing, MCTS had to evaluate 6 possible actions (`left`, `right`, `both`, `flash_left`, `flash_right`, `close`), simulating each one hundreds of ticks into the future. This caused the Streamlit dashboard to freeze for over 60 seconds while evaluating a single scenario.
**Fix (The Hybrid ML-MCTS Architecture):** We repurposed the lightweight `SurrogateModel` as a **heuristic filter**. Before MCTS simulates the future, the ML model predicts a quick "intuition score" for all 6 actions. MCTS then takes *only the top 2 candidate actions* and runs full physics simulations on them. 
**Result:** By ignoring the bottom 4 actions, we reduced the tree search branching factor by 66%, bringing simulation evaluation time down from over 1 minute to approximately 15-18 seconds.

## 4. Final Architecture
The final `MCTSRolloutPolicy` operates as follows:
1. **Receive State:** Grabs the current snapshot.
2. **Filter Candidates:** Uses the `SurrogateModel` to instantly reject the 4 worst actions.
3. **Rollout:** Clones the simulator and runs the top 2 actions forward by $K$ ticks.
4. **Evaluate:** Calculates the total horizon penalty (delay + risk).
5. **Execute:** Returns the action with the lowest penalty.

This architecture acts precisely like modern game-playing AI (e.g., AlphaGo), combining the fast pattern-recognition of Neural Networks/ML with the rigorous logical verification of Monte Carlo Tree Search.
