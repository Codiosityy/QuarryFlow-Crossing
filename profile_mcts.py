import cProfile
import pstats
from pathlib import Path
from quarryflow.dashboard_data import run_policy_suite

def main():
    print("Running profiling...")
    run_policy_suite('chaotic_long_gate', seed=11, model_path='artifacts/models/surrogate.pkl', record_history=False, fast_mode=False)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('tottime')
        stats.print_stats(50)
