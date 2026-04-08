from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.domain_types import AdaptivePolicyConfig
from quarryflow.evaluation import HOLDOUT_SEEDS, VALIDATION_SEEDS, evaluate_policy_suite
from quarryflow.hybrid import load_hybrid_controller
from quarryflow.model import BootstrapSurrogateEnsemble
from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy, HybridAdaptivePolicy, StaticAlternatingPolicy
from quarryflow.scenarios import list_scenarios


def main() -> None:
    model_path = ROOT / "artifacts" / "models" / "surrogate_ensemble.pkl"
    controller_path = ROOT / "artifacts" / "models" / "hybrid_controller.json"
    eval_dir = ROOT / "artifacts" / "eval"
    ensemble = BootstrapSurrogateEnsemble.load(model_path)
    config, bandit, metadata = load_hybrid_controller(controller_path)
    config = config if isinstance(config, AdaptivePolicyConfig) else AdaptivePolicyConfig()

    policies = {
        "Free Flow": FreeFlowPolicy(),
        "Static Alternating": StaticAlternatingPolicy(),
        "Legacy Adaptive": AdaptivePolicy(model=ensemble, config=config),
        "Hybrid Adaptive": HybridAdaptivePolicy(model=ensemble, bandit=bandit, config=config),
    }
    scenarios = list_scenarios()
    validation = evaluate_policy_suite(
        scenario_names=scenarios,
        seeds=VALIDATION_SEEDS,
        policies=policies,
        config=config,
    )
    holdout = evaluate_policy_suite(
        scenario_names=scenarios,
        seeds=HOLDOUT_SEEDS,
        policies=policies,
        config=config,
    )
    validation.to_csv(eval_dir / "validation_episode_rows.csv", index=False)
    holdout.to_csv(eval_dir / "holdout_episode_rows.csv", index=False)
    summary = {
        "hybrid_default_ok": metadata.get("hybrid_default_ok"),
        "validation_mean_reward": round(
            float(validation[validation["policy"] == "Hybrid Adaptive"]["episode_reward"].mean()), 3
        ),
        "holdout_mean_reward": round(
            float(holdout[holdout["policy"] == "Hybrid Adaptive"]["episode_reward"].mean()), 3
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
