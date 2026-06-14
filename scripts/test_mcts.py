import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.dashboard_data import run_policy_suite

def main():
    print("Testing full ML simulation suite with MCTS...")
    # We use fast_mode=False to ensure MCTS is included, 
    # but we will only run a small number of steps by overriding episode_seconds?
    # run_policy_suite does not allow overriding episode_seconds directly unless we modify it, 
    # but we can just run it. The peak scenario is 600s, it might take 10-20s.
    results = run_policy_suite("peak", fast_mode=False, record_history=False)
    
    print("Policies executed:", list(results.keys()))
    if "MCTS Rollout" in results:
        mcts_res = results["MCTS Rollout"]
        print("MCTS Wait Time:", mcts_res.metrics.average_waiting_time)
        print("MCTS Throughput:", mcts_res.metrics.throughput)
        print("MCTS successfully evaluated!")
    else:
        print("MCTS NOT FOUND IN RESULTS!")

if __name__ == "__main__":
    main()
