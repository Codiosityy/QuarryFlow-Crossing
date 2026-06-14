import json
from quarryflow.dashboard_data import run_policy_suite
import random

def main():
    seed = random.randint(1, 1000)
    print(f"Running MCTS policy suite with random seed: {seed}")
    
    # Run the simulation (fast_mode=False ensures MCTS is run)
    results_map = run_policy_suite(
        'chaotic_long_gate', 
        seed=seed, 
        model_path='artifacts/models/surrogate.pkl', 
        record_history=True, 
        fast_mode=False
    )
    
    # Extract the MCTS result specifically
    mcts_result = results_map["MCTS Rollout"]
    
    # Build a clean JSON-serializable dictionary
    data = {
        "scenario": "chaotic_long_gate",
        "seed": seed,
        "metrics": mcts_result.metrics.to_dict(),
        "history": mcts_result.history,
        "actions_taken": mcts_result.actions_taken,
        "decision_traces": [dt.to_dict() for dt in mcts_result.decision_traces]
    }
    
    output_file = "mcts_output.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully generated JSON data for MCTS at {output_file}")

if __name__ == "__main__":
    main()
