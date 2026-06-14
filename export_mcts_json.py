import json
import os
from pathlib import Path

# Adjust path to find the src module
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quarryflow.dashboard_data import run_policy_suite

def export_mcts_json():
    print("Running MCTS pipeline to export JSON data...")
    # Run the policy suite on the chaotic scenario with a random seed
    seed = 42
    results = run_policy_suite(
        scenario_name="chaotic_aggressive",
        seed=seed,
        model_path=None,
        record_history=True,
        record_every=1,
        fast_mode=False
    )
    
    # We just want the MCTS Rollout results
    if "MCTS Rollout" in results:
        mcts_result = results["MCTS Rollout"]
        
        # Convert the decision trace history to JSON
        traces = [trace.to_dict() for trace in mcts_result.decision_traces]
        
        output_file = Path("artifacts") / "mcts_seed42_data.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with output_file.open("w") as f:
            json.dump(traces, f, indent=2)
            
        print(f"Successfully exported {len(traces)} MCTS decision traces to {output_file}")
    else:
        print("MCTS Rollout not found in results!")

if __name__ == "__main__":
    export_mcts_json()
