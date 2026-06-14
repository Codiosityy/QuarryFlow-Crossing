from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .domain_types import AdaptivePolicyConfig, EpisodeMetrics
from .hybrid import StateVectorBuilder
from .model import TARGET_COLUMNS
from .policy import FixedActionPolicy
from .scenarios import build_scenario
from .simulator import RailwayCrossingSimulator


TRAIN_SEEDS = [7, 11, 13, 17, 19, 23]
VALIDATION_SEEDS = [29, 31]
HOLDOUT_SEEDS = [37, 41]

CURRICULUM_STAGES: list[tuple[str, list[str]]] = [
    ("stage_1_light", ["light"]),
    ("stage_2_peak", ["peak"]),
    ("stage_3_chaotic", ["chaotic"]),
    (
        "stage_4_stress",
        ["peak_left_skew", "peak_right_skew", "chaotic_aggressive", "chaotic_long_gate"],
    ),
]


def collect_counterfactual_rows(scenario_name: str, seed: int) -> list[dict]:
    """Collect counterfactual training rows for surrogate model fitting.

    NOTE: For each snapshot, all actions are evaluated with the same state
    features (only the action column differs). This creates correlated
    training rows. The model handles this via the action one-hot encoding,
    but be aware that effective sample size is smaller than the row count.
    """
    config = build_scenario(scenario_name, seed=seed)
    simulator = RailwayCrossingSimulator(config, seed=seed)
    rows: list[dict] = []
    sample_marks = [5.0, 15.0, 30.0]
    captured_marks: set[tuple[int, int]] = set()

    while simulator.time < config.episode_seconds:
        simulator.step(FixedActionPolicy("alternating_4s"))
        snapshot = simulator.build_snapshot()
        if snapshot.barrier_closed:
            continue

        for index, mark in enumerate(sample_marks):
            capture_key = (int(snapshot.time // 120), index)
            if snapshot.time_since_open < mark or capture_key in captured_marks:
                continue
            captured_marks.add(capture_key)
            for action in config.actions:
                feature_row = StateVectorBuilder.build(snapshot, action.name, config)
                outcome = simulator.evaluate_horizon(
                    config.prediction_horizon,
                    FixedActionPolicy(action.name),
                )
                rows.append(feature_row | outcome.to_dict())
    return rows


def split_feature_targets(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    features = []
    targets = []
    for row in rows:
        features.append({key: value for key, value in row.items() if key not in TARGET_COLUMNS})
        targets.append({key: row[key] for key in TARGET_COLUMNS})
    return features, targets


def aggregate_episode_reward(metrics: EpisodeMetrics, config: AdaptivePolicyConfig) -> float:
    return (
        config.throughput_weight * metrics.throughput
        - config.waiting_time_weight * metrics.average_waiting_time
        - config.congestion_weight * metrics.max_congestion_length
        - config.occupancy_risk_weight * metrics.occupancy_risk
        - config.fairness_gap_weight * metrics.fairness_gap
        - config.wrong_side_weight * metrics.wrong_side_queue_peak
        - config.idling_fuel_weight * metrics.total_idling_fuel_liters
        - config.worst_clearance_weight * metrics.worst_clearance_time
    )


def flatten_metrics(
    *,
    label: str,
    scenario_name: str,
    seed: int,
    metrics: EpisodeMetrics,
    config: AdaptivePolicyConfig,
) -> dict[str, float | int | str]:
    row = {
        "policy": label,
        "scenario": scenario_name,
        "seed": seed,
        "episode_reward": aggregate_episode_reward(metrics, config),
    }
    row.update(metrics.to_dict())
    return row


def evaluate_policy_suite(
    *,
    scenario_names: Iterable[str],
    seeds: Iterable[int],
    policies: dict[str, object],
    config: AdaptivePolicyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario_name in scenario_names:
        for seed in seeds:
            scenario = build_scenario(scenario_name, seed=int(seed))
            for label, policy in policies.items():
                simulator = RailwayCrossingSimulator(scenario, seed=int(seed))
                result = simulator.run_episode(policy, record_history=False)
                rows.append(
                    flatten_metrics(
                        label=label,
                        scenario_name=scenario_name,
                        seed=int(seed),
                        metrics=result.metrics,
                        config=config,
                    )
                )
    return pd.DataFrame(rows)


def iter_curriculum_cases(
    *,
    stage_names: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[tuple[str, str, int]]:
    selected_stages = set(stage_names or [stage for stage, _ in CURRICULUM_STAGES])
    selected_seeds = list(seeds or TRAIN_SEEDS)
    cases: list[tuple[str, str, int]] = []
    for stage, scenarios in CURRICULUM_STAGES:
        if stage not in selected_stages:
            continue
        for scenario_name in scenarios:
            for seed in selected_seeds:
                cases.append((stage, scenario_name, int(seed)))
    return cases
