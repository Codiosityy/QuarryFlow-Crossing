from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_ACTIONS
from .domain_types import AdaptivePolicyConfig, CrossingStateSnapshot, ScenarioConfig


SCENARIO_COLUMNS = [
    "scenario_light",
    "scenario_peak",
    "scenario_chaotic",
    "scenario_peak_left_skew",
    "scenario_peak_right_skew",
    "scenario_chaotic_aggressive",
    "scenario_chaotic_long_gate",
]

ACTION_COLUMNS = [f"action_{action.name}" for action in DEFAULT_ACTIONS]

BASE_COLUMNS = [
    "time_since_open",
    "time_to_open",
    "time_to_close",
    "barrier_closed",
    "queue_left",
    "queue_right",
    "queue_count_left",
    "queue_count_right",
    "mean_speed",
    "disorder_index",
    "bike_infiltration",
    "wrong_side_queue_share",
    "dilemma_zone_pressure",
    "mean_reaction_time_near_gate",
    "closure_frustration_index",
    "idling_vehicle_share",
    "idling_fuel_rate_lph",
    "idling_co2_rate_kgph",
    "crossing_occupancy",
    "occupancy_risk",
    "pressure_imbalance",
    "aggressive_share_near_gate",
    "conflict_count",
    "vehicles_spawned",
    "vehicles_cleared",
    "arrival_left",
    "arrival_right",
]

FEATURE_COLUMNS = BASE_COLUMNS + SCENARIO_COLUMNS + ACTION_COLUMNS


class StateVectorBuilder:
    @staticmethod
    def feature_columns() -> list[str]:
        return FEATURE_COLUMNS[:]

    @staticmethod
    def build(snapshot: CrossingStateSnapshot, action_name: str, config: ScenarioConfig) -> dict[str, float]:
        vector = {
            "time_since_open": float(snapshot.time_since_open),
            "time_to_open": float(snapshot.time_to_open),
            "time_to_close": float(snapshot.time_to_close),
            "barrier_closed": float(int(snapshot.barrier_closed)),
            "queue_left": float(snapshot.queue_lengths["left"]),
            "queue_right": float(snapshot.queue_lengths["right"]),
            "queue_count_left": float(snapshot.queue_counts["left"]),
            "queue_count_right": float(snapshot.queue_counts["right"]),
            "mean_speed": float(snapshot.mean_speed),
            "disorder_index": float(snapshot.disorder_index),
            "bike_infiltration": float(snapshot.bike_infiltration),
            "wrong_side_queue_share": float(snapshot.wrong_side_queue_share),
            "dilemma_zone_pressure": float(snapshot.dilemma_zone_pressure),
            "mean_reaction_time_near_gate": float(snapshot.mean_reaction_time_near_gate),
            "closure_frustration_index": float(snapshot.closure_frustration_index),
            "idling_vehicle_share": float(snapshot.idling_vehicle_share),
            "idling_fuel_rate_lph": float(snapshot.idling_fuel_rate_lph),
            "idling_co2_rate_kgph": float(snapshot.idling_co2_rate_kgph),
            "crossing_occupancy": float(snapshot.crossing_occupancy),
            "occupancy_risk": float(snapshot.occupancy_risk),
            "pressure_imbalance": float(snapshot.pressure_imbalance),
            "aggressive_share_near_gate": float(snapshot.aggressive_share_near_gate),
            "conflict_count": float(snapshot.conflict_count),
            "vehicles_spawned": float(snapshot.vehicles_spawned),
            "vehicles_cleared": float(snapshot.vehicles_cleared),
            "arrival_left": float(config.arrival_rate_per_minute["left"]),
            "arrival_right": float(config.arrival_rate_per_minute["right"]),
        }
        for column in SCENARIO_COLUMNS:
            vector[column] = 0.0
        scenario_column = f"scenario_{config.name}"
        if scenario_column in vector:
            vector[scenario_column] = 1.0
        for column in ACTION_COLUMNS:
            vector[column] = 0.0
        action_column = f"action_{action_name}"
        if action_column in vector:
            vector[action_column] = 1.0
        return vector

    @staticmethod
    def as_array(row: dict[str, float], feature_columns: list[str] | None = None) -> np.ndarray:
        columns = feature_columns or FEATURE_COLUMNS
        return np.array([float(row.get(column, 0.0)) for column in columns], dtype=float)

    @staticmethod
    def state_summary(snapshot: CrossingStateSnapshot) -> dict[str, float | int | str]:
        return {
            "time": round(snapshot.time, 2),
            "time_since_open": round(snapshot.time_since_open, 2),
            "queue_left": round(snapshot.queue_lengths["left"], 2),
            "queue_right": round(snapshot.queue_lengths["right"], 2),
            "queue_count_left": snapshot.queue_counts["left"],
            "queue_count_right": snapshot.queue_counts["right"],
            "disorder_index": round(snapshot.disorder_index, 3),
            "wrong_side_queue_share": round(snapshot.wrong_side_queue_share, 3),
            "closure_frustration_index": round(snapshot.closure_frustration_index, 3),
            "occupancy_risk": round(snapshot.occupancy_risk, 3),
            "pressure_imbalance": round(snapshot.pressure_imbalance, 3),
            "current_action": snapshot.current_action,
        }


class LinUCBResidual:
    def __init__(self, feature_columns: list[str] | None = None, *, alpha: float = 1.25) -> None:
        self.feature_columns = feature_columns[:] if feature_columns else FEATURE_COLUMNS[:]
        self.alpha = alpha
        size = len(self.feature_columns)
        self.A = np.eye(size, dtype=float)
        self.b = np.zeros(size, dtype=float)
        self.is_fitted = True

    def _array(self, row: dict[str, float]) -> np.ndarray:
        return StateVectorBuilder.as_array(row, self.feature_columns)

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def mean(self, row: dict[str, float]) -> float:
        vector = self._array(row)
        return float(self.theta() @ vector)

    def bonus(self, row: dict[str, float]) -> float:
        vector = self._array(row)
        solved = np.linalg.solve(self.A, vector)
        return float(self.alpha * np.sqrt(vector @ solved))

    def score(self, row: dict[str, float]) -> float:
        return self.mean(row) + self.bonus(row)

    def update(self, row: dict[str, float], residual_reward: float) -> None:
        vector = self._array(row)
        self.A += np.outer(vector, vector)
        self.b += residual_reward * vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "alpha": self.alpha,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinUCBResidual":
        instance = cls(payload["feature_columns"], alpha=float(payload["alpha"]))
        instance.A = np.array(payload["A"], dtype=float)
        instance.b = np.array(payload["b"], dtype=float)
        return instance


def save_hybrid_controller(
    path: str | Path,
    *,
    config: AdaptivePolicyConfig,
    bandit: LinUCBResidual,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "bandit": bandit.to_dict(),
        "metadata": metadata or {},
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_hybrid_controller(path: str | Path) -> tuple[AdaptivePolicyConfig, LinUCBResidual, dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    config = AdaptivePolicyConfig.from_dict(payload["config"])
    bandit = LinUCBResidual.from_dict(payload["bandit"])
    metadata = dict(payload.get("metadata", {}))
    return config, bandit, metadata
