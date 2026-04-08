from __future__ import annotations

import random

from .config import DRIVER_PROFILES, VEHICLE_LIBRARY
from .domain_types import ScenarioConfig, VehicleAgent


def _sample_from_mix(mix: dict[str, float], rng: random.Random) -> str:
    total = sum(mix.values())
    draw = rng.random() * total
    running = 0.0
    for key, weight in mix.items():
        running += weight
        if draw <= running:
            return key
    return next(iter(mix))


def spawn_vehicle(
    side: str,
    vehicle_id: int,
    current_time: float,
    config: ScenarioConfig,
    rng: random.Random,
    spawn_progress: float,
) -> VehicleAgent:
    vehicle_key = _sample_from_mix(config.vehicle_mix, rng)
    driver_key = _sample_from_mix(config.driver_mix, rng)
    vehicle_spec = VEHICLE_LIBRARY[vehicle_key]
    driver_spec = DRIVER_PROFILES[driver_key]
    reaction_time = max(
        0.25,
        rng.gauss(driver_spec.reaction_time_mean, driver_spec.reaction_time_jitter),
    )

    return VehicleAgent(
        vehicle_id=vehicle_id,
        side=side,
        vehicle_class=vehicle_key,
        driver_profile=driver_key,
        length=vehicle_spec.length,
        width=vehicle_spec.width,
        desired_speed=vehicle_spec.desired_speed * driver_spec.desired_speed,
        max_accel=vehicle_spec.acceleration * driver_spec.acceleration,
        min_gap=driver_spec.min_gap,
        aggression=driver_spec.aggression,
        lateral_flexibility=vehicle_spec.lateral_flexibility,
        reaction_time_seconds=reaction_time,
        gate_rush_bias=driver_spec.gate_rush_bias,
        idling_propensity=driver_spec.idling_propensity,
        idle_fuel_liters_per_second=vehicle_spec.idle_fuel_liters_per_second,
        idle_co2_kg_per_second=vehicle_spec.idle_co2_kg_per_second,
        progress=spawn_progress,
        spawned_at=current_time,
        engine_on_when_waiting=rng.random() < driver_spec.idling_propensity,
    )


def update_lateral_offset(
    vehicle: VehicleAgent,
    *,
    barrier_closed: bool,
    blocked: bool,
    approach_length: float,
    disorder_zone_length: float,
    discipline_bonus: float,
    rng: random.Random,
) -> None:
    in_disorder_zone = approach_length - disorder_zone_length <= vehicle.progress < approach_length
    if not in_disorder_zone:
        vehicle.lateral_offset *= 0.75
        return

    behavior_pressure = vehicle.aggression + vehicle.lateral_flexibility * 0.3 - discipline_bonus
    behavior_pressure = max(0.0, min(1.2, behavior_pressure))
    if not barrier_closed and not blocked:
        target = rng.uniform(-0.25, 0.25) * vehicle.lateral_flexibility
    else:
        squeeze_bias = max(0.0, vehicle.gate_rush_bias - 0.2)
        spread = 0.2 + behavior_pressure * 1.1 + squeeze_bias * 0.4
        target = rng.uniform(-spread, spread)

    vehicle.lateral_offset += (target - vehicle.lateral_offset) * 0.35


def bike_infiltration_score(vehicles: list[VehicleAgent], approach_length: float) -> float:
    bikes = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_class == "bike"
        and approach_length - 12.0 <= vehicle.progress < approach_length
    ]
    if not bikes:
        return 0.0
    infiltrating = [vehicle for vehicle in bikes if abs(vehicle.lateral_offset) > 0.55]
    return len(infiltrating) / len(bikes)


def dilemma_zone_pressure(
    vehicles: list[VehicleAgent],
    *,
    approach_length: float,
    time_to_close: float,
    warning_seconds: float,
) -> float:
    if time_to_close <= 0.0 or time_to_close > warning_seconds:
        return 0.0

    in_zone = [
        vehicle
        for vehicle in vehicles
        if approach_length - 10.0 <= vehicle.progress < approach_length
    ]
    if not in_zone:
        return 0.0

    urgency = 1.0 - min(time_to_close / max(warning_seconds, 0.1), 1.0)
    weighted = [
        min(
            1.0,
            0.55 * vehicle.gate_rush_bias
            + 0.25 * vehicle.aggression
            + 0.20 * (1.0 / max(vehicle.reaction_time_seconds, 0.25)),
        )
        for vehicle in in_zone
    ]
    return min(1.0, sum(weighted) / len(weighted) * (0.55 + 0.45 * urgency))


def wrong_side_queue_share(
    vehicles: list[VehicleAgent],
    *,
    approach_length: float,
    threshold: float,
) -> float:
    near_gate = [
        vehicle
        for vehicle in vehicles
        if approach_length - 14.0 <= vehicle.progress < approach_length
    ]
    if not near_gate:
        return 0.0
    wrong_side = [
        vehicle
        for vehicle in near_gate
        if abs(vehicle.lateral_offset) >= threshold
    ]
    wrong_ratio = len(wrong_side) / len(near_gate)
    total_queued = sum(1 for vehicle in vehicles if vehicle.progress < approach_length)
    backlog_multiplier = min(1.8, 0.85 + total_queued / 55.0)
    return min(1.0, wrong_ratio * backlog_multiplier)


def idling_snapshot(
    vehicles: list[VehicleAgent],
    *,
    approach_length: float,
) -> tuple[float, float, float]:
    queued = [
        vehicle
        for vehicle in vehicles
        if vehicle.progress < approach_length and vehicle.speed < 0.15
    ]
    if not queued:
        return 0.0, 0.0, 0.0

    engine_on = [vehicle for vehicle in queued if vehicle.engine_on_when_waiting]
    share = len(engine_on) / len(queued)
    fuel_rate_lph = sum(vehicle.idle_fuel_liters_per_second for vehicle in engine_on) * 3600.0
    co2_rate_kgph = sum(vehicle.idle_co2_kg_per_second for vehicle in engine_on) * 3600.0
    return share, fuel_rate_lph, co2_rate_kgph


def mean_reaction_time(
    vehicles: list[VehicleAgent],
    *,
    approach_length: float,
) -> float:
    near_gate = [
        vehicle
        for vehicle in vehicles
        if approach_length - 16.0 <= vehicle.progress < approach_length
    ]
    if not near_gate:
        return 0.0
    return sum(vehicle.reaction_time_seconds for vehicle in near_gate) / len(near_gate)
