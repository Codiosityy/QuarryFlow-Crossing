from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)


@dataclass(frozen=True)
class DriverProfileSpec:
    name: str
    desired_speed: float
    acceleration: float
    min_gap: float
    encroachment_bias: float
    aggression: float
    restart_gain: float
    reaction_time_mean: float
    reaction_time_jitter: float
    gate_rush_bias: float
    idling_propensity: float


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    length: float
    width: float
    desired_speed: float
    acceleration: float
    lateral_flexibility: float
    idle_fuel_liters_per_second: float
    idle_co2_kg_per_second: float


@dataclass
class PolicyAction:
    name: str
    mode: str
    burst_seconds: float = 4.0
    settling_delay: float = 0.0
    priority_side: str | None = None
    priority_window: float = 0.0
    discipline_bonus: float = 0.0


@dataclass
class ScenarioConfig:
    name: str
    episode_seconds: int
    time_step: float
    approach_length: float
    disorder_zone_length: float
    crossing_box_length: float
    recovery_length: float
    decision_interval: float
    prediction_horizon: float
    pre_close_warning_seconds: float
    wrong_side_offset_threshold: float
    arrival_rate_per_minute: dict[str, float]
    vehicle_mix: dict[str, float]
    driver_mix: dict[str, float]
    train_closures: list[tuple[float, float]]
    actions: list[PolicyAction]
    random_seed: int = 7


@dataclass
class VehicleAgent:
    vehicle_id: int
    side: str
    vehicle_class: str
    driver_profile: str
    length: float
    width: float
    desired_speed: float
    max_accel: float
    min_gap: float
    aggression: float
    lateral_flexibility: float
    reaction_time_seconds: float
    gate_rush_bias: float
    idling_propensity: float
    idle_fuel_liters_per_second: float
    idle_co2_kg_per_second: float
    progress: float
    speed: float = 0.0
    lateral_offset: float = 0.0
    waiting_time: float = 0.0
    spawned_at: float = 0.0
    entered_crossing_at: float | None = None
    exited_at: float | None = None
    engine_on_when_waiting: bool = True
    restart_delay_remaining: float = 0.0
    needs_restart_delay: bool = False
    total_idling_fuel_liters: float = 0.0
    total_idling_co2_kg: float = 0.0
    finished: bool = False


@dataclass
class CrossingStateSnapshot:
    time: float
    barrier_closed: bool
    time_since_open: float
    time_to_open: float
    time_to_close: float
    queue_lengths: dict[str, float]
    queue_counts: dict[str, int]
    mean_speed: float
    disorder_index: float
    bike_infiltration: float
    wrong_side_queue_share: float
    dilemma_zone_pressure: float
    mean_reaction_time_near_gate: float
    closure_frustration_index: float
    idling_vehicle_share: float
    idling_fuel_rate_lph: float
    idling_co2_rate_kgph: float
    crossing_occupancy: int
    occupancy_risk: float
    pressure_imbalance: float
    aggressive_share_near_gate: float
    current_action: str
    conflict_count: int
    vehicles_spawned: int
    vehicles_cleared: int

    def to_feature_row(self, action_name: str | None = None) -> dict[str, Any]:
        return {
            "time_since_open": self.time_since_open,
            "time_to_open": self.time_to_open,
            "time_to_close": self.time_to_close,
            "barrier_closed": int(self.barrier_closed),
            "queue_left": self.queue_lengths[LEFT],
            "queue_right": self.queue_lengths[RIGHT],
            "queue_count_left": self.queue_counts[LEFT],
            "queue_count_right": self.queue_counts[RIGHT],
            "mean_speed": self.mean_speed,
            "disorder_index": self.disorder_index,
            "bike_infiltration": self.bike_infiltration,
            "wrong_side_queue_share": self.wrong_side_queue_share,
            "dilemma_zone_pressure": self.dilemma_zone_pressure,
            "mean_reaction_time_near_gate": self.mean_reaction_time_near_gate,
            "closure_frustration_index": self.closure_frustration_index,
            "idling_vehicle_share": self.idling_vehicle_share,
            "idling_fuel_rate_lph": self.idling_fuel_rate_lph,
            "idling_co2_rate_kgph": self.idling_co2_rate_kgph,
            "crossing_occupancy": self.crossing_occupancy,
            "occupancy_risk": self.occupancy_risk,
            "pressure_imbalance": self.pressure_imbalance,
            "aggressive_share_near_gate": self.aggressive_share_near_gate,
            "conflict_count": self.conflict_count,
            "vehicles_spawned": self.vehicles_spawned,
            "vehicles_cleared": self.vehicles_cleared,
            "candidate_action": action_name or self.current_action,
        }


@dataclass
class HorizonOutcome:
    average_waiting_time: float
    throughput: float
    max_congestion_length: float
    occupancy_risk_horizon: float
    fairness_gap_horizon: float
    wrong_side_queue_share_horizon: float
    idling_fuel_liters_horizon: float
    mean_clearance_time_horizon: float
    worst_clearance_time_horizon: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class AdaptivePolicyConfig:
    throughput_weight: float = 3.0
    waiting_time_weight: float = 0.7
    congestion_weight: float = 0.08
    occupancy_risk_weight: float = 15.0
    fairness_gap_weight: float = 8.0
    wrong_side_weight: float = 6.0
    idling_fuel_weight: float = 1.2
    worst_clearance_weight: float = 0.04
    uncertainty_penalty_weight: float = 2.5
    linucb_alpha: float = 1.25
    shield_occupancy_threshold: float = 0.88
    shield_fairness_threshold: float = 0.22
    fairness_queue_threshold: int = 6
    free_mode_disorder_penalty: float = 8.0
    free_mode_disorder_threshold: float = 0.45
    free_mode_wrong_side_penalty: float = 7.0
    free_mode_wrong_side_threshold: float = 0.28
    priority_mismatch_penalty: float = 5.0
    imbalance_priority_threshold: float = 0.12
    alternating_high_queue_bonus: float = 2.0
    alternating_high_queue_threshold: int = 18
    settle_reaction_delay_threshold: float = 1.55
    settle_wrong_side_threshold: float = 0.34

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AdaptivePolicyConfig":
        return cls(**values)

    def reward(self, outcome: HorizonOutcome) -> float:
        return (
            self.throughput_weight * outcome.throughput
            - self.waiting_time_weight * outcome.average_waiting_time
            - self.congestion_weight * outcome.max_congestion_length
            - self.occupancy_risk_weight * outcome.occupancy_risk_horizon
            - self.fairness_gap_weight * outcome.fairness_gap_horizon
            - self.wrong_side_weight * outcome.wrong_side_queue_share_horizon
            - self.idling_fuel_weight * outcome.idling_fuel_liters_horizon
            - self.worst_clearance_weight * outcome.worst_clearance_time_horizon
        )


@dataclass
class DecisionTrace:
    time: float
    chosen_action: str
    state_summary: dict[str, float | int | str]
    action_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "chosen_action": self.chosen_action,
            "state_summary": self.state_summary,
            "action_scores": self.action_scores[:],
        }


@dataclass
class EpisodeMetrics:
    average_waiting_time: float
    throughput: float
    max_congestion_length: float
    clearance_time: float
    worst_clearance_time: float
    conflict_count: int
    occupancy_risk: float
    fairness_gap: float
    wrong_side_queue_peak: float
    dilemma_zone_peak: float
    total_idling_fuel_liters: float
    total_idling_co2_kg: float
    vehicles_spawned: int
    vehicles_cleared: int
    disorder_peak: float

    def to_dict(self) -> dict[str, float]:
        return {
            "average_waiting_time": self.average_waiting_time,
            "throughput": self.throughput,
            "max_congestion_length": self.max_congestion_length,
            "clearance_time": self.clearance_time,
            "worst_clearance_time": self.worst_clearance_time,
            "conflict_count": float(self.conflict_count),
            "occupancy_risk": self.occupancy_risk,
            "fairness_gap": self.fairness_gap,
            "wrong_side_queue_peak": self.wrong_side_queue_peak,
            "dilemma_zone_peak": self.dilemma_zone_peak,
            "total_idling_fuel_liters": self.total_idling_fuel_liters,
            "total_idling_co2_kg": self.total_idling_co2_kg,
            "vehicles_spawned": float(self.vehicles_spawned),
            "vehicles_cleared": float(self.vehicles_cleared),
            "disorder_peak": self.disorder_peak,
        }


@dataclass
class SimulationResult:
    metrics: EpisodeMetrics
    history: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[CrossingStateSnapshot] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    decision_traces: list[DecisionTrace] = field(default_factory=list)
