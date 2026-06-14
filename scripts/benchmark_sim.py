import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.scenarios import build_scenario
from quarryflow.simulator import RailwayCrossingSimulator
from quarryflow.policy import FreeFlowPolicy

def benchmark_ticks(num_ticks: int, scenario_name: str = "peak"):
    config = build_scenario(scenario_name)
    sim = RailwayCrossingSimulator(config)
    policy = FreeFlowPolicy()
    
    print(f"Benchmarking {num_ticks} ticks on '{scenario_name}' scenario...")
    
    start_time = time.perf_counter()
    
    for _ in range(num_ticks):
        # We don't record history to avoid memory bloat during benchmark
        sim.step(policy, record_history=False)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    tps = num_ticks / duration
    
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Ticks Per Second (TPS): {tps:.2f}")
    print(f"Simulation time progressed: {num_ticks * config.time_step} seconds")
    print("-" * 40)

if __name__ == "__main__":
    print("Starting Phase 1: MCTS Rollout Engine Benchmarking")
    print("=" * 40)
    for ticks in [1000, 5000, 10000]:
        benchmark_ticks(ticks)
