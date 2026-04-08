from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.domain_types import AdaptivePolicyConfig
from quarryflow.evaluation import (
    CURRICULUM_STAGES,
    HOLDOUT_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    collect_counterfactual_rows,
    evaluate_policy_suite,
    split_feature_targets,
)
from quarryflow.hybrid import LinUCBResidual, save_hybrid_controller
from quarryflow.model import BootstrapSurrogateEnsemble, TARGET_COLUMNS
from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy, HybridAdaptivePolicy, StaticAlternatingPolicy
from quarryflow.scenarios import build_scenario, list_scenarios
from quarryflow.simulator import RailwayCrossingSimulator

FAST_TRAIN_SEEDS = [7, 11, 13]
FAST_VALIDATION_SEEDS = [29]
FAST_HOLDOUT_SEEDS = [37]


def collect_rows_for_suite(scenario_names: list[str], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for scenario_name in scenario_names:
        for seed in seeds:
            rows.extend(collect_counterfactual_rows(scenario_name, seed))
    return rows


def evaluate_surrogate(model: BootstrapSurrogateEnsemble, rows: list[dict]) -> dict[str, object]:
    features, targets = split_feature_targets(rows)
    mae: dict[str, float] = {}
    for target_column in TARGET_COLUMNS:
        errors = []
        for feature_row, target_row in zip(features, targets):
            prediction = model.predict_row(feature_row)
            errors.append(abs(prediction[target_column] - target_row[target_column]))
        mae[target_column] = round(sum(errors) / max(len(errors), 1), 6)
    return {"rows": len(rows), "mae": mae}


def build_policy_bundle(
    *,
    ensemble: BootstrapSurrogateEnsemble,
    bandit: LinUCBResidual | None,
    config: AdaptivePolicyConfig,
    include_hybrid: bool = True,
) -> dict[str, object]:
    policies: dict[str, object] = {
        "Free Flow": FreeFlowPolicy(),
        "Static Alternating": StaticAlternatingPolicy(),
        "Legacy Adaptive": AdaptivePolicy(model=ensemble, config=config),
    }
    if include_hybrid and bandit is not None:
        policies["Hybrid Adaptive"] = HybridAdaptivePolicy(
            model=ensemble,
            bandit=bandit,
            config=config,
            training_mode=False,
        )
    return policies


def train_bandit(
    *,
    ensemble: BootstrapSurrogateEnsemble,
    config: AdaptivePolicyConfig,
    curriculum_stages: list[tuple[str, list[str]]],
    train_seeds: list[int],
    validation_seeds: list[int],
    stage_passes: int,
) -> tuple[LinUCBResidual, pd.DataFrame]:
    bandit = LinUCBResidual(alpha=config.linucb_alpha)
    best_bandit_payload = copy.deepcopy(bandit.to_dict())
    best_validation_reward = float("-inf")
    learning_rows: list[dict[str, float | int | str]] = []
    iteration = 0
    seen_scenarios: list[str] = []

    for stage_name, stage_scenarios in curriculum_stages:
        for scenario_name in stage_scenarios:
            if scenario_name not in seen_scenarios:
                seen_scenarios.append(scenario_name)

        for pass_index in range(stage_passes):
            iteration += 1
            print(
                f"[bandit] stage={stage_name} pass={pass_index + 1}/{stage_passes} "
                f"scenarios={','.join(stage_scenarios)}"
            )
            for scenario_name in stage_scenarios:
                for seed in train_seeds:
                    simulator = RailwayCrossingSimulator(
                        build_scenario(scenario_name, seed=seed),
                        seed=seed,
                    )
                    policy = HybridAdaptivePolicy(
                        model=ensemble,
                        bandit=bandit,
                        config=config,
                        training_mode=True,
                    )
                    simulator.run_episode(policy, record_history=False)

            validation = evaluate_policy_suite(
                scenario_names=seen_scenarios,
                seeds=validation_seeds,
                policies=build_policy_bundle(ensemble=ensemble, bandit=bandit, config=config),
                config=config,
            )
            hybrid_reward = float(
                validation[validation["policy"] == "Hybrid Adaptive"]["episode_reward"].mean()
            )
            legacy_reward = float(
                validation[validation["policy"] == "Legacy Adaptive"]["episode_reward"].mean()
            )
            if hybrid_reward > best_validation_reward:
                best_validation_reward = hybrid_reward
                best_bandit_payload = copy.deepcopy(bandit.to_dict())
            learning_rows.append(
                {
                    "iteration": iteration,
                    "stage": stage_name,
                    "pass_index": pass_index + 1,
                    "validation_hybrid_reward": round(hybrid_reward, 3),
                    "validation_legacy_reward": round(legacy_reward, 3),
                    "best_validation_reward": round(best_validation_reward, 3),
                }
            )

    return LinUCBResidual.from_dict(best_bandit_payload), pd.DataFrame(learning_rows)


def summarize_episode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("policy", as_index=False)
        .agg(
            average_waiting_time=("average_waiting_time", "mean"),
            throughput=("throughput", "mean"),
            max_congestion_length=("max_congestion_length", "mean"),
            clearance_time=("clearance_time", "mean"),
            worst_clearance_time=("worst_clearance_time", "mean"),
            occupancy_risk=("occupancy_risk", "mean"),
            fairness_gap=("fairness_gap", "mean"),
            episode_reward=("episode_reward", "mean"),
        )
        .round(3)
    )
    return summary


def hybrid_gate(holdout_frame: pd.DataFrame) -> tuple[bool, dict[str, float]]:
    rewards = holdout_frame.groupby("policy")["episode_reward"].mean().to_dict()
    waits = (
        holdout_frame.groupby(["policy", "scenario"])["average_waiting_time"]
        .mean()
        .to_dict()
    )
    free_peak = waits.get(("Free Flow", "peak"), 1.0)
    free_chaotic = waits.get(("Free Flow", "chaotic"), 1.0)
    legacy_peak = waits.get(("Legacy Adaptive", "peak"), free_peak)
    hybrid_peak = waits.get(("Hybrid Adaptive", "peak"), free_peak)
    legacy_chaotic = waits.get(("Legacy Adaptive", "chaotic"), free_chaotic)
    hybrid_chaotic = waits.get(("Hybrid Adaptive", "chaotic"), free_chaotic)

    legacy_peak_gain = (free_peak - legacy_peak) / max(free_peak, 1e-6) * 100.0
    hybrid_peak_gain = (free_peak - hybrid_peak) / max(free_peak, 1e-6) * 100.0
    legacy_chaotic_gain = (free_chaotic - legacy_chaotic) / max(free_chaotic, 1e-6) * 100.0
    hybrid_chaotic_gain = (free_chaotic - hybrid_chaotic) / max(free_chaotic, 1e-6) * 100.0

    gate_ok = (
        rewards.get("Hybrid Adaptive", float("-inf")) > rewards.get("Legacy Adaptive", float("-inf"))
        and hybrid_peak_gain >= legacy_peak_gain - 5.0
        and hybrid_chaotic_gain >= legacy_chaotic_gain - 5.0
    )
    return gate_ok, {
        "hybrid_reward": round(float(rewards.get("Hybrid Adaptive", 0.0)), 3),
        "legacy_reward": round(float(rewards.get("Legacy Adaptive", 0.0)), 3),
        "hybrid_peak_wait_gain_pct": round(hybrid_peak_gain, 3),
        "legacy_peak_wait_gain_pct": round(legacy_peak_gain, 3),
        "hybrid_chaotic_wait_gain_pct": round(hybrid_chaotic_gain, 3),
        "legacy_chaotic_wait_gain_pct": round(legacy_chaotic_gain, 3),
    }


def resolve_seed_profile(profile: str) -> tuple[list[int], list[int], list[int]]:
    if profile == "full":
        return TRAIN_SEEDS[:], VALIDATION_SEEDS[:], HOLDOUT_SEEDS[:]
    return FAST_TRAIN_SEEDS[:], FAST_VALIDATION_SEEDS[:], FAST_HOLDOUT_SEEDS[:]


def select_curriculum_stages(selected_scenarios: list[str]) -> list[tuple[str, list[str]]]:
    selected = set(selected_scenarios)
    stages: list[tuple[str, list[str]]] = []
    for stage_name, stage_scenarios in CURRICULUM_STAGES:
        filtered = [scenario for scenario in stage_scenarios if scenario in selected]
        if filtered:
            stages.append((stage_name, filtered))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--stage-passes", type=int, default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--n-models", type=int, default=None)
    parser.add_argument("--eval-dir", default=str(ROOT / "artifacts" / "eval"))
    parser.add_argument("--model-dir", default=str(ROOT / "artifacts" / "models"))
    args = parser.parse_args()

    config = AdaptivePolicyConfig()
    all_scenarios = args.scenarios or list_scenarios()
    train_seeds, validation_seeds, holdout_seeds = resolve_seed_profile(args.profile)
    stage_passes = args.stage_passes if args.stage_passes is not None else (1 if args.profile == "fast" else 2)
    n_models = args.n_models if args.n_models is not None else (2 if args.profile == "fast" else 3)
    curriculum_stages = select_curriculum_stages(all_scenarios)
    if not curriculum_stages:
        raise ValueError("No curriculum stages remain after filtering scenarios.")

    print(
        json.dumps(
            {
                "profile": args.profile,
                "scenarios": all_scenarios,
                "train_seeds": train_seeds,
                "validation_seeds": validation_seeds,
                "holdout_seeds": holdout_seeds,
                "stage_passes": stage_passes,
                "n_models": n_models,
            },
            indent=2,
        )
    )

    print("[data] collecting counterfactual rows")
    train_rows = collect_rows_for_suite(all_scenarios, train_seeds)
    validation_rows = collect_rows_for_suite(all_scenarios, validation_seeds)
    holdout_rows = collect_rows_for_suite(all_scenarios, holdout_seeds)
    train_features, train_targets = split_feature_targets(train_rows)

    print("[model] fitting surrogate ensemble")
    ensemble = BootstrapSurrogateEnsemble(n_models=n_models, random_seed=42).fit(train_features, train_targets)
    eval_dir = Path(args.eval_dir)
    model_dir = Path(args.model_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ensemble.save(model_dir / "surrogate_ensemble.pkl")

    surrogate_eval = {
        "validation": evaluate_surrogate(ensemble, validation_rows),
        "holdout": evaluate_surrogate(ensemble, holdout_rows),
    }
    (eval_dir / "surrogate_eval.json").write_text(json.dumps(surrogate_eval, indent=2), encoding="utf-8")

    print("[bandit] training hybrid controller")
    bandit, learning_curve = train_bandit(
        ensemble=ensemble,
        config=config,
        curriculum_stages=curriculum_stages,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        stage_passes=stage_passes,
    )
    learning_curve.to_csv(eval_dir / "learning_curve.csv", index=False)

    print("[eval] validation suite")
    validation_frame = evaluate_policy_suite(
        scenario_names=all_scenarios,
        seeds=validation_seeds,
        policies=build_policy_bundle(ensemble=ensemble, bandit=bandit, config=config),
        config=config,
    )
    validation_frame.to_csv(eval_dir / "validation_episode_rows.csv", index=False)
    summarize_episode_frame(validation_frame).to_csv(eval_dir / "validation_summary.csv", index=False)

    print("[eval] holdout suite")
    holdout_frame = evaluate_policy_suite(
        scenario_names=all_scenarios,
        seeds=holdout_seeds,
        policies=build_policy_bundle(ensemble=ensemble, bandit=bandit, config=config),
        config=config,
    )
    holdout_frame.to_csv(eval_dir / "holdout_episode_rows.csv", index=False)
    summarize_episode_frame(holdout_frame).to_csv(eval_dir / "holdout_summary.csv", index=False)

    gate_ok, gate_metrics = hybrid_gate(holdout_frame)
    metadata = {
        "profile": args.profile,
        "stage_passes": stage_passes,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "holdout_seeds": holdout_seeds,
        "scenarios": all_scenarios,
        "hybrid_default_ok": gate_ok,
        "gate_metrics": gate_metrics,
        "learning_curve_path": str(eval_dir / "learning_curve.csv"),
        "validation_summary_path": str(eval_dir / "validation_summary.csv"),
        "holdout_summary_path": str(eval_dir / "holdout_summary.csv"),
        "surrogate_eval_path": str(eval_dir / "surrogate_eval.json"),
        "model_path": str(model_dir / "surrogate_ensemble.pkl"),
    }
    save_hybrid_controller(
        model_dir / "hybrid_controller.json",
        config=config,
        bandit=bandit,
        metadata=metadata,
    )
    print(json.dumps({"hybrid_default_ok": gate_ok, **gate_metrics}, indent=2))


if __name__ == "__main__":
    main()
