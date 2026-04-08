from __future__ import annotations

import copy
import random

from .behaviors import (
    bike_infiltration_score,
    dilemma_zone_pressure,
    idling_snapshot,
    mean_reaction_time,
    spawn_vehicle,
    update_lateral_offset,
    wrong_side_queue_share,
)
from .policy import FreeFlowPolicy
from .domain_types import (
    DecisionTrace,
    HorizonOutcome,
    LEFT,
    RIGHT,
    SIDES,
    CrossingStateSnapshot,
    EpisodeMetrics,
    PolicyAction,
    ScenarioConfig,
    SimulationResult,
    VehicleAgent,
)
from .scenarios import validate_scenario_config


def _opposite(side: str) -> str:
    return RIGHT if side == LEFT else LEFT


class RailwayCrossingSimulator:
    def __init__(self, config: ScenarioConfig, *, seed: int | None = None) -> None:
        validate_scenario_config(config)
        self.config = config
        self.route_length = (
            config.approach_length + config.crossing_box_length + config.recovery_length
        )
        self.seed = seed if seed is not None else config.random_seed
        self.rng = random.Random(self.seed)
        self.reset(seed=self.seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.spawn_rng = random.Random(self.seed)
        self.behavior_rng = random.Random(self.seed + 10_003)
        self.time = 0.0
        self.barrier_closed = self._barrier_closed_at(0.0)
        self.vehicles: dict[str, list[VehicleAgent]] = {LEFT: [], RIGHT: []}
        self.completed_vehicles: list[VehicleAgent] = []
        self.vehicle_counter = 0
        self.vehicles_spawned = 0
        self.vehicles_cleared = 0
        self.vehicles_cleared_by_side = {LEFT: 0, RIGHT: 0}
        self.conflict_count = 0
        self.current_action = self.config.actions[0]
        self.last_decision_time = -self.config.decision_interval
        self.history: list[dict] = []
        self.snapshots: list[CrossingStateSnapshot] = []
        self.actions_taken: list[dict] = []
        self.decision_traces: list[DecisionTrace] = []
        self.max_disorder = 0.0
        self.max_congestion_length = 0.0
        self.max_occupancy_risk = 0.0
        self.max_wrong_side_queue_share = 0.0
        self.max_dilemma_zone_pressure = 0.0
        self.total_idling_fuel_liters = 0.0
        self.total_idling_co2_kg = 0.0
        self.clearance_durations: list[float] = []
        self._clearance_tracking = False
        self._clearance_started_at: float | None = None
        self._entry_freeze_until = 0.0
        self._conflict_release_side: str | None = None
        self._recorded_initial_state = False

    def clone(self) -> "RailwayCrossingSimulator":
        return copy.deepcopy(self)

    def run_episode(self, policy=None, *, record_history: bool = False) -> SimulationResult:
        self.reset(seed=self.seed)
        if policy is None:
            policy = FreeFlowPolicy()
        if record_history:
            self._record_state()
        while self.time < self.config.episode_seconds:
            self.step(policy, record_history=record_history)
        return self._build_result()

    def run_for(self, duration: float, policy=None, *, record_history: bool = False) -> SimulationResult:
        if policy is None:
            policy = FreeFlowPolicy()
        end_time = min(self.config.episode_seconds, self.time + duration)
        while self.time < end_time:
            self.step(policy, record_history=record_history)
        return self._build_result()

    def evaluate_horizon(self, duration: float, policy=None) -> HorizonOutcome:
        clone = self.clone()
        if policy is None:
            policy = FreeFlowPolicy()

        end_time = min(clone.config.episode_seconds, clone.time + duration)
        initial_cleared = clone.vehicles_cleared
        initial_left = clone.vehicles_cleared_by_side[LEFT]
        initial_right = clone.vehicles_cleared_by_side[RIGHT]
        initial_clearance_count = len(clone.clearance_durations)
        queue_integral = 0.0
        max_congestion = 0.0
        max_occupancy_risk = 0.0
        max_wrong_side = 0.0
        steps = 0
        initial_idling_fuel = clone.total_idling_fuel_liters

        while clone.time < end_time:
            clone.step(policy, record_history=False)
            snapshot = clone.build_snapshot()
            queue_integral += (
                snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT]
            ) * clone.config.time_step
            max_congestion = max(
                max_congestion,
                snapshot.queue_lengths[LEFT],
                snapshot.queue_lengths[RIGHT],
            )
            max_occupancy_risk = max(max_occupancy_risk, snapshot.occupancy_risk)
            max_wrong_side = max(max_wrong_side, snapshot.wrong_side_queue_share)
            steps += 1

        elapsed = max(end_time - self.time, clone.config.time_step)
        avg_wait_proxy = queue_integral / max(steps, 1)
        throughput = (clone.vehicles_cleared - initial_cleared) / elapsed * 3600.0
        newly_cleared = max(clone.vehicles_cleared - initial_cleared, 0)
        fairness_gap_horizon = (
            abs((clone.vehicles_cleared_by_side[LEFT] - initial_left) - (clone.vehicles_cleared_by_side[RIGHT] - initial_right))
            / max(newly_cleared, 1)
        )
        new_clearances = clone.clearance_durations[initial_clearance_count:]
        if new_clearances:
            mean_clearance = sum(new_clearances) / len(new_clearances)
            worst_clearance = max(new_clearances)
        else:
            mean_clearance = elapsed
            worst_clearance = elapsed
        return HorizonOutcome(
            average_waiting_time=avg_wait_proxy,
            throughput=throughput,
            max_congestion_length=max_congestion,
            occupancy_risk_horizon=max_occupancy_risk,
            fairness_gap_horizon=fairness_gap_horizon,
            wrong_side_queue_share_horizon=max_wrong_side,
            idling_fuel_liters_horizon=max(0.0, clone.total_idling_fuel_liters - initial_idling_fuel),
            mean_clearance_time_horizon=mean_clearance,
            worst_clearance_time_horizon=worst_clearance,
        )

    def step(self, policy, *, record_history: bool = False) -> None:
        barrier_before = self.barrier_closed
        self.barrier_closed = self._barrier_closed_at(self.time)

        if barrier_before and not self.barrier_closed:
            self.last_decision_time = self.time - self.config.decision_interval
            snapshot = self.build_snapshot()
            if snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT] > 0:
                self._clearance_tracking = True
                self._clearance_started_at = self.time

        self._spawn_new_vehicles()

        if not self.barrier_closed:
            if self.time - self.last_decision_time >= self.config.decision_interval:
                self.current_action = policy.decide(self)
                self.last_decision_time = self.time
                self.actions_taken.append(
                    {
                        "time": round(self.time, 2),
                        "action": self.current_action.name,
                        "policy": policy.__class__.__name__,
                    }
                )
        else:
            self.current_action = self.config.actions[0]

        entry_blocked = self._entry_blocked_map()
        self._move_side(LEFT, entry_blocked[LEFT])
        self._move_side(RIGHT, entry_blocked[RIGHT])

        self.time += self.config.time_step

        snapshot = self.build_snapshot()
        self.max_disorder = max(self.max_disorder, snapshot.disorder_index)
        self.max_congestion_length = max(
            self.max_congestion_length,
            snapshot.queue_lengths[LEFT],
            snapshot.queue_lengths[RIGHT],
        )
        self.max_occupancy_risk = max(self.max_occupancy_risk, snapshot.occupancy_risk)
        self.max_wrong_side_queue_share = max(
            self.max_wrong_side_queue_share,
            snapshot.wrong_side_queue_share,
        )
        self.max_dilemma_zone_pressure = max(
            self.max_dilemma_zone_pressure,
            snapshot.dilemma_zone_pressure,
        )
        self._update_clearance(snapshot)

        if record_history:
            self._record_state(snapshot=snapshot)

    def build_snapshot(self) -> CrossingStateSnapshot:
        queue_lengths = {
            side: self._queue_length(side)
            for side in SIDES
        }
        queue_counts = {
            side: self._queue_count(side)
            for side in SIDES
        }

        active = self._active_vehicles()
        near_gate = [
            vehicle
            for vehicle in active
            if self.config.approach_length - self.config.disorder_zone_length
            <= vehicle.progress
            < self.config.approach_length
        ]
        mean_speed = sum(vehicle.speed for vehicle in active) / len(active) if active else 0.0
        time_to_close = self._time_to_close(self.time)
        wrong_side_share = wrong_side_queue_share(
            active,
            approach_length=self.config.approach_length,
            threshold=self.config.wrong_side_offset_threshold,
        )
        dilemma_pressure = dilemma_zone_pressure(
            active,
            approach_length=self.config.approach_length,
            time_to_close=time_to_close,
            warning_seconds=self.config.pre_close_warning_seconds,
        )
        mean_reaction = mean_reaction_time(
            active,
            approach_length=self.config.approach_length,
        )
        idling_share, idling_fuel_rate_lph, idling_co2_rate_kgph = idling_snapshot(
            active,
            approach_length=self.config.approach_length,
        )

        lateral_component = (
            sum(abs(vehicle.lateral_offset) for vehicle in near_gate) / len(near_gate)
            if near_gate
            else 0.0
        )
        aggressive_share = (
            sum(1 for vehicle in near_gate if vehicle.aggression > 0.75) / len(near_gate)
            if near_gate
            else 0.0
        )
        density_component = min(1.0, len(near_gate) / 10.0)
        conflict_component = min(1.0, self.conflict_count / 8.0)
        disorder_index = min(
            1.0,
            0.30 * lateral_component
            + 0.18 * aggressive_share
            + 0.14 * density_component
            + 0.14 * conflict_component
            + 0.16 * wrong_side_share
            + 0.08 * dilemma_pressure,
        )

        crossing_occupancy = self._crossing_occupancy_count()
        both_sides_pressing = int(
            self._leader_ready(LEFT) is not None and self._leader_ready(RIGHT) is not None
        )
        occupancy_risk = min(
            1.0,
            0.18 * crossing_occupancy
            + 0.35 * both_sides_pressing
            + 0.20 * disorder_index
            + 0.17 * wrong_side_share
            + 0.02 * self.conflict_count,
        )
        total_pressure = queue_lengths[LEFT] + queue_lengths[RIGHT]
        pressure_imbalance = (
            (queue_lengths[LEFT] - queue_lengths[RIGHT]) / max(total_pressure, 1.0)
            if total_pressure > 0
            else 0.0
        )
        closure_frustration = min(
            1.0,
            0.45 * min((queue_counts[LEFT] + queue_counts[RIGHT]) / 20.0, 1.0)
            + 0.35 * min(
                (self._time_to_open(self.time) if self.barrier_closed else max(0.0, 8.0 - self._time_since_open(self.time)))
                / 60.0,
                1.0,
            )
            + 0.20 * wrong_side_share,
        )

        return CrossingStateSnapshot(
            time=round(self.time, 2),
            barrier_closed=self.barrier_closed,
            time_since_open=0.0 if self.barrier_closed else self._time_since_open(self.time),
            time_to_open=self._time_to_open(self.time),
            time_to_close=time_to_close,
            queue_lengths=queue_lengths,
            queue_counts=queue_counts,
            mean_speed=mean_speed,
            disorder_index=disorder_index,
            bike_infiltration=bike_infiltration_score(active, self.config.approach_length),
            wrong_side_queue_share=wrong_side_share,
            dilemma_zone_pressure=dilemma_pressure,
            mean_reaction_time_near_gate=mean_reaction,
            closure_frustration_index=closure_frustration,
            idling_vehicle_share=idling_share,
            idling_fuel_rate_lph=idling_fuel_rate_lph,
            idling_co2_rate_kgph=idling_co2_rate_kgph,
            crossing_occupancy=crossing_occupancy,
            occupancy_risk=occupancy_risk,
            pressure_imbalance=pressure_imbalance,
            aggressive_share_near_gate=aggressive_share,
            current_action=self.current_action.name,
            conflict_count=self.conflict_count,
            vehicles_spawned=self.vehicles_spawned,
            vehicles_cleared=self.vehicles_cleared,
        )

    def _build_result(self) -> SimulationResult:
        all_vehicles = self._active_vehicles()
        total_vehicles = self.completed_vehicles + all_vehicles
        average_waiting = (
            sum(
                max(
                    0.0,
                    ((vehicle.exited_at if vehicle.exited_at is not None else self.time) - vehicle.spawned_at)
                    - (self.route_length / max(vehicle.desired_speed, 0.1)),
                )
                for vehicle in total_vehicles
            )
            / len(total_vehicles)
            if total_vehicles
            else 0.0
        )
        if self._clearance_tracking and self._clearance_started_at is not None:
            self.clearance_durations.append(self.time - self._clearance_started_at)
            self._clearance_tracking = False
            self._clearance_started_at = None
        throughput = self.vehicles_cleared / max(self.time, 1.0) * 3600.0
        clearance_time = (
            sum(self.clearance_durations) / len(self.clearance_durations)
            if self.clearance_durations
            else float(self.config.episode_seconds)
        )
        worst_clearance_time = (
            max(self.clearance_durations)
            if self.clearance_durations
            else float(self.config.episode_seconds)
        )
        fairness_gap = (
            abs(
                self.vehicles_cleared_by_side[LEFT]
                - self.vehicles_cleared_by_side[RIGHT]
            )
            / max(self.vehicles_cleared, 1)
        )
        metrics = EpisodeMetrics(
            average_waiting_time=average_waiting,
            throughput=throughput,
            max_congestion_length=self.max_congestion_length,
            clearance_time=clearance_time,
            worst_clearance_time=worst_clearance_time,
            conflict_count=self.conflict_count,
            occupancy_risk=self.max_occupancy_risk,
            fairness_gap=fairness_gap,
            wrong_side_queue_peak=self.max_wrong_side_queue_share,
            dilemma_zone_peak=self.max_dilemma_zone_pressure,
            total_idling_fuel_liters=self.total_idling_fuel_liters,
            total_idling_co2_kg=self.total_idling_co2_kg,
            vehicles_spawned=self.vehicles_spawned,
            vehicles_cleared=self.vehicles_cleared,
            disorder_peak=self.max_disorder,
        )
        return SimulationResult(
            metrics=metrics,
            history=self.history[:],
            snapshots=self.snapshots[:],
            actions_taken=self.actions_taken[:],
            decision_traces=self.decision_traces[:],
        )

    def record_decision_trace(self, trace: DecisionTrace) -> None:
        self.decision_traces.append(trace)

    def _record_state(self, snapshot: CrossingStateSnapshot | None = None) -> None:
        snapshot = snapshot or self.build_snapshot()
        self.snapshots.append(snapshot)
        self.history.append(
            {
                "time": snapshot.time,
                "barrier_closed": snapshot.barrier_closed,
                "queue_left": snapshot.queue_lengths[LEFT],
                "queue_right": snapshot.queue_lengths[RIGHT],
                "queue_count_left": snapshot.queue_counts[LEFT],
                "queue_count_right": snapshot.queue_counts[RIGHT],
                "mean_speed": snapshot.mean_speed,
                "disorder_index": snapshot.disorder_index,
                "wrong_side_queue_share": snapshot.wrong_side_queue_share,
                "dilemma_zone_pressure": snapshot.dilemma_zone_pressure,
                "closure_frustration_index": snapshot.closure_frustration_index,
                "idling_vehicle_share": snapshot.idling_vehicle_share,
                "idling_fuel_rate_lph": snapshot.idling_fuel_rate_lph,
                "occupancy_risk": snapshot.occupancy_risk,
                "crossing_occupancy": snapshot.crossing_occupancy,
                "current_action": snapshot.current_action,
                "vehicles": [
                    {
                        "vehicle_id": vehicle.vehicle_id,
                        "side": vehicle.side,
                        "vehicle_class": vehicle.vehicle_class,
                        "progress": round(vehicle.progress, 2),
                        "speed": round(vehicle.speed, 2),
                        "lateral_offset": round(vehicle.lateral_offset, 2),
                    }
                    for vehicle in self._active_vehicles()
                ],
            }
        )

    def _spawn_new_vehicles(self) -> None:
        for side in SIDES:
            expected = self.config.arrival_rate_per_minute[side] / 60.0 * self.config.time_step
            count = int(expected)
            if self.spawn_rng.random() < expected - count:
                count += 1

            for _ in range(count):
                self.vehicle_counter += 1
                vehicle = spawn_vehicle(
                    side,
                    self.vehicle_counter,
                    self.time,
                    self.config,
                    self.spawn_rng,
                    spawn_progress=0.0,
                )
                if self.vehicles[side]:
                    tail = min(self.vehicles[side], key=lambda item: item.progress)
                    vehicle.progress = min(
                        0.0,
                        tail.progress - tail.length - vehicle.min_gap - 1.0,
                    )
                self.vehicles[side].append(vehicle)
                self.vehicles_spawned += 1

    def _entry_blocked_map(self) -> dict[str, bool]:
        blocked = {LEFT: self.barrier_closed, RIGHT: self.barrier_closed}
        if self.barrier_closed:
            return blocked

        if self.time < self._entry_freeze_until:
            return {LEFT: True, RIGHT: True}

        allowed = set(self._allowed_sides())
        if not allowed:
            return {LEFT: True, RIGHT: True}

        occupied_sides = self._crossing_occupied_sides()
        if occupied_sides == {LEFT}:
            blocked[RIGHT] = True
        elif occupied_sides == {RIGHT}:
            blocked[LEFT] = True

        if self.current_action.mode == "free" and occupied_sides == {LEFT} and self._queue_count(RIGHT) > 0:
            blocked[LEFT] = True
        if self.current_action.mode == "free" and occupied_sides == {RIGHT} and self._queue_count(LEFT) > 0:
            blocked[RIGHT] = True

        for side in SIDES:
            if side not in allowed:
                blocked[side] = True

        if self._conflict_release_side and not occupied_sides:
            blocked[_opposite(self._conflict_release_side)] = True
            self._conflict_release_side = None
            return blocked

        if not occupied_sides and allowed == {LEFT, RIGHT}:
            left_ready = self._leader_ready(LEFT)
            right_ready = self._leader_ready(RIGHT)
            if left_ready is not None and right_ready is not None:
                self.conflict_count += 1
                left_pressure = self._queue_length(LEFT) + left_ready.aggression * 10.0
                right_pressure = self._queue_length(RIGHT) + right_ready.aggression * 10.0
                winner = LEFT if left_pressure >= right_pressure else RIGHT
                wrong_side_pressure = min(
                    1.0,
                    (
                        abs(left_ready.lateral_offset)
                        + abs(right_ready.lateral_offset)
                    ) / max(self.config.wrong_side_offset_threshold * 2.4, 0.1),
                )
                queue_pressure = min(
                    1.0,
                    (self._queue_count(LEFT) + self._queue_count(RIGHT)) / 26.0,
                )
                delay = (
                    1.5
                    + ((left_ready.aggression + right_ready.aggression) * 1.2)
                    + wrong_side_pressure * 2.2
                    + queue_pressure * 1.8
                )
                self._entry_freeze_until = self.time + delay
                self._conflict_release_side = winner
                return {LEFT: True, RIGHT: True}

        return blocked

    def _allowed_sides(self) -> tuple[str, ...]:
        action = self.current_action
        if self.barrier_closed:
            return ()

        elapsed = self._time_since_open(self.time)
        if action.mode == "free":
            return SIDES
        if elapsed < action.settling_delay:
            return ()
        if action.mode == "alternating":
            slot = int((elapsed - action.settling_delay) // max(action.burst_seconds, 1.0))
            starting_side = action.priority_side or LEFT
            active = starting_side if slot % 2 == 0 else _opposite(starting_side)
            return (active,)
        if action.mode == "priority":
            priority = action.priority_side or LEFT
            if elapsed < action.priority_window:
                return (priority,)
            slot = int(max(elapsed - action.priority_window, 0.0) // max(action.burst_seconds, 1.0))
            active = priority if slot % 2 == 0 else _opposite(priority)
            return (active,)
        return SIDES

    def _move_side(self, side: str, entry_blocked: bool) -> None:
        vehicles = sorted(
            [vehicle for vehicle in self.vehicles[side] if not vehicle.finished],
            key=lambda item: item.progress,
            reverse=True,
        )
        crossing_start = self.config.approach_length
        time_to_close = self._time_to_close(self.time)
        warning_active = (
            not self.barrier_closed
            and 0.0 < time_to_close <= self.config.pre_close_warning_seconds
        )

        for index, vehicle in enumerate(vehicles):
            leader = vehicles[index - 1] if index > 0 else None
            proposed_speed = min(
                vehicle.desired_speed,
                vehicle.speed + vehicle.max_accel * self.config.time_step,
            )
            near_gate = (
                self.config.approach_length - self.config.disorder_zone_length
                <= vehicle.progress
                < self.config.approach_length
            )
            in_crossing = (
                self.config.approach_length
                <= vehicle.progress
                < self.config.approach_length + self.config.crossing_box_length
            )
            if warning_active and near_gate:
                urgency = 1.0 - min(
                    time_to_close / max(self.config.pre_close_warning_seconds, 0.1),
                    1.0,
                )
                rush_gain = 1.0 + vehicle.gate_rush_bias * (0.18 + 0.34 * urgency)
                proposed_speed = min(vehicle.desired_speed * rush_gain, proposed_speed * rush_gain)
            if (
                self.current_action.mode == "free"
                and near_gate
            ):
                proposed_speed *= max(0.5, 1.0 - abs(vehicle.lateral_offset) * 0.22)
            if self.current_action.mode == "free" and in_crossing and self._queue_count(_opposite(side)) > 0:
                opposing_ready = self._leader_ready(_opposite(side)) is not None
                proposed_speed *= 0.42 if opposing_ready else 0.55
            proposed_progress = vehicle.progress + proposed_speed * self.config.time_step

            if leader is not None:
                safe_progress = leader.progress - leader.length - vehicle.min_gap
                proposed_progress = min(proposed_progress, safe_progress)

            blocked_here = False
            queue_released = (
                not self.barrier_closed
                and not entry_blocked
                and vehicle.progress < crossing_start
            )
            leader_gap_open = leader is None or (
                leader.progress - leader.length - vehicle.min_gap - vehicle.progress
            ) > 0.55
            if vehicle.progress < crossing_start <= proposed_progress and entry_blocked:
                proposed_progress = min(proposed_progress, crossing_start - 0.35)
                blocked_here = True

            if queue_released and leader_gap_open and vehicle.needs_restart_delay:
                if vehicle.restart_delay_remaining <= 0.0:
                    vehicle.restart_delay_remaining = vehicle.reaction_time_seconds
                vehicle.restart_delay_remaining = max(
                    0.0,
                    vehicle.restart_delay_remaining - self.config.time_step,
                )
                if vehicle.restart_delay_remaining > 0.0:
                    proposed_progress = vehicle.progress
                    blocked_here = True
                else:
                    vehicle.needs_restart_delay = False

            proposed_progress = min(proposed_progress, self.route_length)
            actual_move = max(0.0, proposed_progress - vehicle.progress)
            vehicle.speed = actual_move / self.config.time_step

            update_lateral_offset(
                vehicle,
                barrier_closed=self.barrier_closed,
                blocked=blocked_here or vehicle.speed < 0.15,
                approach_length=self.config.approach_length,
                disorder_zone_length=self.config.disorder_zone_length,
                discipline_bonus=self.current_action.discipline_bonus,
                rng=self.behavior_rng,
            )

            if vehicle.progress < crossing_start <= proposed_progress and vehicle.entered_crossing_at is None:
                vehicle.entered_crossing_at = self.time

            vehicle.progress = proposed_progress

            if vehicle.speed < 0.8 and not vehicle.finished and vehicle.progress < crossing_start:
                vehicle.waiting_time += self.config.time_step
                if vehicle.speed < 0.15 or blocked_here:
                    vehicle.needs_restart_delay = True
                if vehicle.speed < 0.15 and vehicle.engine_on_when_waiting:
                    fuel_delta = vehicle.idle_fuel_liters_per_second * self.config.time_step
                    co2_delta = vehicle.idle_co2_kg_per_second * self.config.time_step
                    vehicle.total_idling_fuel_liters += fuel_delta
                    vehicle.total_idling_co2_kg += co2_delta
                    self.total_idling_fuel_liters += fuel_delta
                    self.total_idling_co2_kg += co2_delta
            elif vehicle.speed > 0.8:
                vehicle.restart_delay_remaining = 0.0
                vehicle.needs_restart_delay = False

            if vehicle.progress >= self.route_length:
                vehicle.finished = True
                vehicle.exited_at = self.time
                self.vehicles_cleared += 1
                self.vehicles_cleared_by_side[side] += 1
                self.completed_vehicles.append(vehicle)

        self.vehicles[side] = [vehicle for vehicle in vehicles if not vehicle.finished]

    def _queue_length(self, side: str) -> float:
        queued = [vehicle.progress for vehicle in self.vehicles[side] if vehicle.progress < self.config.approach_length]
        if not queued:
            return 0.0
        return self.config.approach_length - min(queued)

    def _queue_count(self, side: str) -> int:
        return sum(1 for vehicle in self.vehicles[side] if vehicle.progress < self.config.approach_length)

    def _crossing_occupancy_count(self) -> int:
        crossing_start = self.config.approach_length
        crossing_end = crossing_start + self.config.crossing_box_length
        return sum(
            1
            for vehicle in self._active_vehicles()
            if crossing_start <= vehicle.progress < crossing_end
        )

    def _crossing_occupied_sides(self) -> set[str]:
        crossing_start = self.config.approach_length
        crossing_end = crossing_start + self.config.crossing_box_length
        return {
            vehicle.side
            for vehicle in self._active_vehicles()
            if crossing_start <= vehicle.progress < crossing_end
        }

    def _leader_ready(self, side: str) -> VehicleAgent | None:
        if not self.vehicles[side]:
            return None
        leader = max(self.vehicles[side], key=lambda item: item.progress)
        if self.config.approach_length - 4.0 <= leader.progress < self.config.approach_length:
            return leader
        return None

    def _active_vehicles(self) -> list[VehicleAgent]:
        return sorted(
            [vehicle for side in SIDES for vehicle in self.vehicles[side]],
            key=lambda item: (item.side, item.progress),
        )

    def _update_clearance(self, snapshot: CrossingStateSnapshot) -> None:
        total_queued = snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT]
        if self._clearance_tracking and not snapshot.barrier_closed and total_queued <= 1:
            if self._clearance_started_at is not None:
                self.clearance_durations.append(snapshot.time - self._clearance_started_at)
            self._clearance_tracking = False
            self._clearance_started_at = None
        if snapshot.barrier_closed and self._clearance_tracking:
            if self._clearance_started_at is not None:
                self.clearance_durations.append(snapshot.time - self._clearance_started_at)
            self._clearance_tracking = False
            self._clearance_started_at = None

    def _barrier_closed_at(self, value: float) -> bool:
        return any(start <= value < end for start, end in self.config.train_closures)

    def _time_to_open(self, value: float) -> float:
        for start, end in self.config.train_closures:
            if start <= value < end:
                return end - value
        return 0.0

    def _time_to_close(self, value: float) -> float:
        for start, _ in self.config.train_closures:
            if value < start:
                return start - value
        return 0.0

    def _time_since_open(self, value: float) -> float:
        if self._barrier_closed_at(value):
            return 0.0

        last_open = 0.0
        for _, end in self.config.train_closures:
            if end <= value:
                last_open = end
        return value - last_open
