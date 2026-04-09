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


def _apply_dashboard_tuning(config, *, fast_mode: bool = False):
    """Patch scenario config for faster dashboard runs.

    Reduces prediction_horizon (90 → 45 s).  This halves the number
    of steps in each counterfactual rollout while still capturing the
    full post-gate-reopening congestion dynamics.  The core physics,
    vehicle mix, and metric calculations remain identical.

    When fast_mode is True, also increases time_step to 1.0 for a
    ~2x speedup on initial loads.
    """
    config.prediction_horizon = 45.0
    if fast_mode:
        config.prediction_horizon = 30.0
        config.time_step = 1.0
    return config


def run_policy_suite(
    scenario_name: str,
    *,
    seed: int = 7,
    model_path: str | None = None,
    ensemble_path: str | None = None,
    controller_path: str | None = None,
    record_history: bool = True,
    record_every: int = 2,
    fast_mode: bool = False,
    progress_callback=None,
):
    config = _apply_dashboard_tuning(
        build_scenario(scenario_name, seed=seed), fast_mode=fast_mode
    )
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
    }

    if fast_mode:
        # Heuristic-only adaptive for fast initial loads (no rollout overhead)
        policies["Legacy Adaptive"] = AdaptivePolicy(model=None)
    else:
        policies["Legacy Adaptive"] = AdaptivePolicy(model=legacy_model or ensemble)

    if not fast_mode and ensemble is not None and bandit is not None and controller_config is not None:
        policies["Hybrid Adaptive"] = HybridAdaptivePolicy(
            model=ensemble,
            bandit=bandit,
            config=controller_config,
        )

    results = {}
    total_policies = len(policies)
    for idx, (label, policy) in enumerate(policies.items()):
        if progress_callback:
            progress_callback(idx / total_policies, f"Running {label}...")
        simulator = RailwayCrossingSimulator(config, seed=seed)
        if record_history and record_every > 1:
            # Subsample: run step-by-step, only record on every Nth step
            simulator.reset(seed=simulator.seed)
            step_counter = 0
            simulator._record_state()  # record initial state
            while simulator.time < config.episode_seconds:
                step_counter += 1
                should_record = (step_counter % record_every == 0)
                simulator.step(policy, record_history=should_record)
            results[label] = simulator._build_result()
        else:
            results[label] = simulator.run_episode(policy, record_history=record_history)
    if progress_callback:
        progress_callback(1.0, "Done")
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
