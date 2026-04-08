from __future__ import annotations

from copy import deepcopy

from .config import DEFAULT_ACTIONS
from .domain_types import ScenarioConfig


SCENARIO_LIBRARY: dict[str, dict[str, object]] = {
    "light": {
        "episode_seconds": 540,
        "arrival_rate_per_minute": {"left": 13.0, "right": 12.0},
        "train_closures": [(90.0, 125.0), (290.0, 330.0)],
    },
    "peak": {
        "episode_seconds": 600,
        "arrival_rate_per_minute": {"left": 18.0, "right": 18.0},
        "train_closures": [(85.0, 145.0), (275.0, 345.0)],
    },
    "chaotic": {
        "episode_seconds": 660,
        "arrival_rate_per_minute": {"left": 22.0, "right": 20.0},
        "train_closures": [(70.0, 155.0), (250.0, 360.0)],
    },
    "peak_left_skew": {
        "episode_seconds": 600,
        "arrival_rate_per_minute": {"left": 24.0, "right": 14.0},
        "train_closures": [(85.0, 145.0), (275.0, 345.0)],
    },
    "peak_right_skew": {
        "episode_seconds": 600,
        "arrival_rate_per_minute": {"left": 14.0, "right": 24.0},
        "train_closures": [(85.0, 145.0), (275.0, 345.0)],
    },
    "chaotic_aggressive": {
        "episode_seconds": 660,
        "arrival_rate_per_minute": {"left": 22.0, "right": 20.0},
        "train_closures": [(70.0, 155.0), (250.0, 360.0)],
        "driver_mix": {
            "cautious": 0.05,
            "compliant": 0.10,
            "opportunistic": 0.20,
            "assertive": 0.25,
            "aggressive": 0.25,
            "reckless": 0.15,
        },
    },
    "chaotic_long_gate": {
        "episode_seconds": 720,
        "arrival_rate_per_minute": {"left": 22.0, "right": 20.0},
        "train_closures": [(70.0, 170.0), (250.0, 390.0)],
    },
}


def list_scenarios() -> list[str]:
    return list(SCENARIO_LIBRARY)


def validate_scenario_config(config: ScenarioConfig) -> None:
    if config.episode_seconds <= 0:
        raise ValueError("episode_seconds must be positive.")
    if config.time_step <= 0:
        raise ValueError("time_step must be positive.")
    if config.decision_interval <= 0:
        raise ValueError("decision_interval must be positive.")
    if config.prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive.")
    if config.pre_close_warning_seconds <= 0:
        raise ValueError("pre_close_warning_seconds must be positive.")
    if config.wrong_side_offset_threshold <= 0:
        raise ValueError("wrong_side_offset_threshold must be positive.")
    if config.approach_length <= 0 or config.crossing_box_length <= 0 or config.recovery_length <= 0:
        raise ValueError("Scenario geometry lengths must be positive.")
    for side, rate in config.arrival_rate_per_minute.items():
        if rate < 0:
            raise ValueError(f"Arrival rate for {side} must be nonnegative.")
    for start, end in config.train_closures:
        if start >= end:
            raise ValueError("Each train closure must satisfy start < end.")
    if abs(sum(config.vehicle_mix.values()) - 1.0) > 1e-6:
        raise ValueError("vehicle_mix must sum to 1.0.")
    if abs(sum(config.driver_mix.values()) - 1.0) > 1e-6:
        raise ValueError("driver_mix must sum to 1.0.")
    for action in config.actions:
        if action.mode not in {"free", "alternating", "priority"}:
            raise ValueError(f"Unsupported action mode: {action.mode}")
        if action.mode in {"alternating", "priority"} and action.burst_seconds <= 0:
            raise ValueError("burst_seconds must be positive for alternating/priority actions.")
        if action.settling_delay < 0 or action.priority_window < 0:
            raise ValueError("Action delays/windows must be nonnegative.")
        if action.mode == "priority" and action.priority_side not in {"left", "right"}:
            raise ValueError("priority actions must set priority_side to left or right.")


def build_scenario(name: str, *, seed: int = 7) -> ScenarioConfig:
    if name not in SCENARIO_LIBRARY:
        raise KeyError(f"Unknown scenario preset: {name}")

    preset = SCENARIO_LIBRARY[name]
    config = ScenarioConfig(
        name=name,
        episode_seconds=int(preset["episode_seconds"]),
        time_step=0.5,
        approach_length=100.0,
        disorder_zone_length=15.0,
        crossing_box_length=12.0,
        recovery_length=80.0,
        decision_interval=5.0,
        prediction_horizon=90.0,
        pre_close_warning_seconds=6.0,
        wrong_side_offset_threshold=0.58,
        arrival_rate_per_minute=deepcopy(preset["arrival_rate_per_minute"]),
        vehicle_mix={"car": 0.60, "bike": 0.25, "auto": 0.15},
        driver_mix=deepcopy(
            preset.get(
                "driver_mix",
                {
                    "cautious": 0.08,
                    "compliant": 0.22,
                    "opportunistic": 0.34,
                    "assertive": 0.16,
                    "aggressive": 0.14,
                    "reckless": 0.06,
                },
            )
        ),
        train_closures=deepcopy(preset["train_closures"]),
        actions=deepcopy(DEFAULT_ACTIONS),
        random_seed=seed,
    )
    validate_scenario_config(config)
    return config
