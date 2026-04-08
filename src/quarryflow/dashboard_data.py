from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import compare_results, improvement_summary
from .hybrid import load_hybrid_controller
from .model import BootstrapSurrogateEnsemble, SurrogateModel
from .policy import AdaptivePolicy, FreeFlowPolicy, HybridAdaptivePolicy, StaticAlternatingPolicy
from .reporting import summarize_results
from .scenarios import build_scenario
from .simulator import RailwayCrossingSimulator


def run_policy_suite(
    scenario_name: str,
    *,
    seed: int = 7,
    model_path: str | None = None,
    ensemble_path: str | None = None,
    controller_path: str | None = None,
    record_history: bool = True,
):
    config = build_scenario(scenario_name, seed=seed)
    legacy_model = None
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            legacy_model = SurrogateModel.load(candidate)
    ensemble = None
    controller_config = None
    bandit = None
    controller_metadata = {}
    if ensemble_path:
        ensemble_candidate = Path(ensemble_path)
        if ensemble_candidate.exists():
            ensemble = BootstrapSurrogateEnsemble.load(ensemble_candidate)
    if controller_path:
        controller_candidate = Path(controller_path)
        if controller_candidate.exists():
            controller_config, bandit, controller_metadata = load_hybrid_controller(controller_candidate)

    policies = {
        "Free Flow": FreeFlowPolicy(),
        "Static Alternating": StaticAlternatingPolicy(),
        "Legacy Adaptive": AdaptivePolicy(model=legacy_model or ensemble),
    }
    if ensemble is not None and bandit is not None and controller_config is not None:
        policies["Hybrid Adaptive"] = HybridAdaptivePolicy(
            model=ensemble,
            bandit=bandit,
            config=controller_config,
        )

    results = {}
    for label, policy in policies.items():
        simulator = RailwayCrossingSimulator(config, seed=seed)
        results[label] = simulator.run_episode(policy, record_history=record_history)
    return results


def comparison_frame(results) -> pd.DataFrame:
    return pd.DataFrame(compare_results(results))


def improvement_frame(results) -> pd.DataFrame:
    baseline = results["Free Flow"].metrics
    rows = []
    for label, result in results.items():
        if label == "Free Flow":
            continue
        row = {"policy": label}
        row.update(improvement_summary(baseline, result.metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def history_frame(result) -> pd.DataFrame:
    return pd.DataFrame(result.history)


def vehicle_frame(result, time_index: int) -> pd.DataFrame:
    if not result.history:
        return pd.DataFrame()
    clipped_index = max(0, min(time_index, len(result.history) - 1))
    vehicles = result.history[clipped_index]["vehicles"]
    return pd.DataFrame(vehicles)


def actions_frame(result) -> pd.DataFrame:
    return pd.DataFrame(result.actions_taken)


def action_summary_frame(result) -> pd.DataFrame:
    actions = actions_frame(result)
    if actions.empty:
        return pd.DataFrame(columns=["action", "count", "share_pct"])

    counts = (
        actions.groupby("action")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = counts["count"].sum()
    counts["share_pct"] = counts["count"] / total * 100.0
    return counts


def decision_trace_frame(result) -> pd.DataFrame:
    rows = []
    for trace in getattr(result, "decision_traces", []):
        for action_score in trace.action_scores:
            rows.append(
                {
                    "time": trace.time,
                    "chosen_action": trace.chosen_action,
                    "candidate_action": action_score["action"],
                    "score": action_score["score"],
                    "base_utility": action_score["base_utility"],
                    "linucb_mean": action_score["linucb_mean"],
                    "linucb_bonus": action_score["linucb_bonus"],
                    "utility_std": action_score["utility_std"],
                    "veto_reason": action_score["veto_reason"] or "",
                }
            )
    return pd.DataFrame(rows)


def judge_summary(results, scenario_name: str) -> dict[str, object]:
    return summarize_results(results, scenario_name)
