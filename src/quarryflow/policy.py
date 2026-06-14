from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domain_types import (
    AdaptivePolicyConfig,
    CrossingStateSnapshot,
    DecisionTrace,
    HorizonOutcome,
    LEFT,
    PolicyAction,
    RIGHT,
    ScenarioConfig,
)
from .hybrid import LinUCBResidual, StateVectorBuilder


def _opposite(side: str) -> str:
    return RIGHT if side == LEFT else LEFT


def find_action(config: ScenarioConfig, action_name: str) -> PolicyAction:
    for action in config.actions:
        if action.name == action_name:
            return action
    raise KeyError(f"Unknown action: {action_name}")


def _prediction_to_outcome(
    prediction: dict[str, float],
    snapshot: CrossingStateSnapshot,
    default_clearance: float,
) -> HorizonOutcome:
    return HorizonOutcome(
        average_waiting_time=float(prediction.get("average_waiting_time", 0.0)),
        throughput=float(prediction.get("throughput", 0.0)),
        max_congestion_length=float(prediction.get("max_congestion_length", 0.0)),
        occupancy_risk_horizon=float(prediction.get("occupancy_risk_horizon", snapshot.occupancy_risk)),
        fairness_gap_horizon=float(prediction.get("fairness_gap_horizon", 0.0)),
        wrong_side_queue_share_horizon=float(
            prediction.get("wrong_side_queue_share_horizon", snapshot.wrong_side_queue_share)
        ),
        idling_fuel_liters_horizon=float(prediction.get("idling_fuel_liters_horizon", 0.0)),
        mean_clearance_time_horizon=float(prediction.get("mean_clearance_time_horizon", default_clearance)),
        worst_clearance_time_horizon=float(prediction.get("worst_clearance_time_horizon", default_clearance)),
    )


def _utility_std(stds: dict[str, float], config: AdaptivePolicyConfig) -> float:
    return (
        abs(config.throughput_weight) * float(stds.get("throughput", 0.0))
        + abs(config.waiting_time_weight) * float(stds.get("average_waiting_time", 0.0))
        + abs(config.congestion_weight) * float(stds.get("max_congestion_length", 0.0))
        + abs(config.occupancy_risk_weight) * float(stds.get("occupancy_risk_horizon", 0.0))
        + abs(config.fairness_gap_weight) * float(stds.get("fairness_gap_horizon", 0.0))
        + abs(config.wrong_side_weight) * float(stds.get("wrong_side_queue_share_horizon", 0.0))
        + abs(config.idling_fuel_weight) * float(stds.get("idling_fuel_liters_horizon", 0.0))
        + abs(config.worst_clearance_weight) * float(stds.get("worst_clearance_time_horizon", 0.0))
    )


def _priority_adjustment(
    action: PolicyAction,
    snapshot: CrossingStateSnapshot,
    config: AdaptivePolicyConfig,
) -> float:
    score = 0.0
    total_queue = snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT]
    # Penalise free flow ONLY in extreme disorder — light disorder is tolerable
    if action.mode == "free" and snapshot.disorder_index > 0.6:
        score -= config.free_mode_disorder_penalty * 0.35
    if action.mode == "free" and snapshot.wrong_side_queue_share > 0.45:
        score -= config.free_mode_wrong_side_penalty * 0.35
    if action.mode == "priority":
        side = action.priority_side or LEFT
        expected = LEFT if snapshot.pressure_imbalance >= 0 else RIGHT
        if side != expected and abs(snapshot.pressure_imbalance) > config.imbalance_priority_threshold:
            score -= config.priority_mismatch_penalty
    # Only bonus alternating in severe conditions
    if action.mode == "alternating" and total_queue > config.alternating_high_queue_threshold:
        score += config.alternating_high_queue_bonus * 0.5
    # Settle bonus only in the first few seconds with true chaos
    if (
        action.settling_delay > 0.0
        and snapshot.time_since_open < 4.0
        and (
            snapshot.wrong_side_queue_share > 0.45
            or snapshot.disorder_index > 0.55
        )
    ):
        score += 0.3 * config.alternating_high_queue_bonus
    # Free flow gets a throughput bonus when conditions are manageable
    if (
        action.mode == "free"
        and snapshot.time_since_open > 8.0
        and snapshot.occupancy_risk < 0.75
        and snapshot.disorder_index < 0.4
    ):
        score += 0.5 * config.alternating_high_queue_bonus
    return score


@dataclass
class FixedActionPolicy:
    action_name: str

    def decide(self, simulator) -> PolicyAction:
        return find_action(simulator.config, self.action_name)


@dataclass
class FreeFlowPolicy:
    action_name: str = "free_release"

    def decide(self, simulator) -> PolicyAction:
        return find_action(simulator.config, self.action_name)


@dataclass
class StaticAlternatingPolicy:
    action_name: str = "alternating_4s"

    def decide(self, simulator) -> PolicyAction:
        return find_action(simulator.config, self.action_name)


@dataclass
class AdaptivePolicy:
    model: object | None = None
    config: AdaptivePolicyConfig = field(default_factory=AdaptivePolicyConfig)

    def decide(self, simulator) -> PolicyAction:
        snapshot = simulator.build_snapshot()
        if self.model is not None and getattr(self.model, "is_fitted", False):
            return self._decide_with_model(simulator, snapshot)
        if hasattr(simulator, "evaluate_horizon"):
            return self._decide_with_rollout(simulator, snapshot)
        return self._decide_heuristically(simulator, snapshot)

    def _decide_with_model(self, simulator, snapshot: CrossingStateSnapshot) -> PolicyAction:
        best_score = float("-inf")
        best_action = simulator.config.actions[0]
        default_clearance = float(simulator.config.prediction_horizon)
        for action in simulator.config.actions:
            row = StateVectorBuilder.build(snapshot, action.name, simulator.config)
            prediction = self.model.predict_row(row)
            outcome = _prediction_to_outcome(prediction, snapshot, default_clearance)
            score = self.config.reward(outcome)
            score += _priority_adjustment(action, snapshot, self.config)

            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _decide_with_rollout(self, simulator, snapshot: CrossingStateSnapshot) -> PolicyAction:
        candidate_names = self._rollout_candidates(simulator, snapshot)
        best_score = float("-inf")
        best_action = find_action(simulator.config, sorted(candidate_names)[0])
        scored_actions: list[dict[str, Any]] = []

        for action_name in candidate_names:
            action = find_action(simulator.config, action_name)
            outcome = simulator.evaluate_horizon(
                simulator.config.prediction_horizon,
                FixedActionPolicy(action.name),
            )
            base_utility = self.config.reward(outcome)
            score = base_utility + _priority_adjustment(action, snapshot, self.config)
            scored_actions.append(
                {
                    "action": action_name,
                    "base_utility": round(base_utility, 3),
                    "utility_std": 0.0,
                    "linucb_mean": 0.0,
                    "linucb_bonus": 0.0,
                    "score": round(score, 3),
                    "veto_reason": None,
                }
            )
            if score > best_score:
                best_score = score
                best_action = action

        if hasattr(simulator, "record_decision_trace"):
            simulator.record_decision_trace(
                DecisionTrace(
                    time=round(snapshot.time, 2),
                    chosen_action=best_action.name,
                    state_summary=StateVectorBuilder.state_summary(snapshot),
                    action_scores=scored_actions,
                )
            )

        return best_action

    def _rollout_candidates(self, simulator, snapshot: CrossingStateSnapshot) -> set[str]:
        candidates = {"free_release"}
        heuristic_choice = self._decide_heuristically(simulator, snapshot)
        candidates.add(heuristic_choice.name)

        total_queue = snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT]
        left_q = snapshot.queue_counts[LEFT]
        right_q = snapshot.queue_counts[RIGHT]

        # Always evaluate priority when both sides have vehicles —
        # let the rollout scorer decide if it's actually better.
        if left_q > 2 and right_q > 2:
            candidates.add("priority_left")
            candidates.add("priority_right")
        elif total_queue > 6:
            candidates.add("priority_left" if left_q > right_q else "priority_right")

        if total_queue > 16 and snapshot.disorder_index > 0.45:
            candidates.add("alternating_4s")
        if (
            snapshot.wrong_side_queue_share > 0.35
            or snapshot.disorder_index > 0.5
        ) and snapshot.time_since_open < 8.0:
            candidates.add("settle_then_alt")
        if total_queue > 20 and snapshot.occupancy_risk > 0.7:
            candidates.add("alternating_6s")
        return candidates

    def _decide_heuristically(self, simulator, snapshot: CrossingStateSnapshot) -> PolicyAction:
        """Balanced adaptive heuristic: reduce delay and conflicts while
        preserving throughput.

        The simulator's crossing is single-threaded (only one side can
        occupy the box at a time) even in free mode.  Controlled modes
        (alternating/priority) avoid costly conflict freezes (~4-7s each)
        but reduce throughput by blocking one side.  This heuristic
        triggers controlled modes only when the expected conflict cost
        exceeds the throughput cost of intervention.

        Decision logic (in priority order):
        - trivial traffic (queue ≤ 5) → free_release
        - queue imbalance (|imbalance| > 0.16, both sides not pressing) → priority
        - only one side pressing → free_release (no conflict risk)
        - high aggression → free_release with priority fallback for imbalance
        - balanced queues (|imbalance| < 0.18) → free_release (alternating hurts)
        - early reopen + severe wrong side + disorder → settle_then_alt
        - early reopen + extreme overcrowding → alternating_6s
        - transition to free when conditions ease (time > 8s, risk < 0.80)
        - high risk + severe disorder + unbalanced → alternating_6s or 4s
        - default → free_release
        """
        total_queue = snapshot.queue_counts[LEFT] + snapshot.queue_counts[RIGHT]
        left_q = snapshot.queue_counts[LEFT]
        right_q = snapshot.queue_counts[RIGHT]
        imbalance = snapshot.pressure_imbalance
        early_reopen = snapshot.time_since_open < 12.0
        severe_wrong_side = snapshot.wrong_side_queue_share > self.config.settle_wrong_side_threshold
        slow_restart = (
            snapshot.mean_reaction_time_near_gate > self.config.settle_reaction_delay_threshold
        )
        # Conflicts only happen when BOTH sides have ready vehicles.
        # If one side is nearly empty, free flow is safe.
        both_sides_pressing = left_q > 4 and right_q > 4
        # Aggressive drivers ignore discipline, so controlled modes
        # are less effective when the aggressive share is high.
        high_aggression = snapshot.aggressive_share_near_gate > 0.35

        if snapshot.barrier_closed:
            return find_action(simulator.config, "free_release")

        # Trivial traffic: free flow
        if total_queue <= 5:
            return find_action(simulator.config, "free_release")

        # Queue imbalance: priority drain FIRST — this works regardless
        # of whether both sides are pressing.  Priority mode gives the
        # heavier side exclusive access, avoiding conflicts.
        if (
            abs(imbalance) > 0.16
            and total_queue > 8
            and snapshot.wrong_side_queue_share < 0.5
            and not (severe_wrong_side or slow_restart)
        ):
            chosen = "priority_left" if imbalance > 0 else "priority_right"
            return find_action(simulator.config, chosen)

        # If only one side has significant vehicles, free flow is safe
        # (no conflict risk when the other side is nearly empty).
        if not both_sides_pressing:
            return find_action(simulator.config, "free_release")

        # High aggression: controlled modes ineffective, use free flow
        # with occasional priority for strong imbalance.
        if high_aggression:
            if abs(imbalance) > 0.12 and total_queue > 10:
                chosen = "priority_left" if imbalance > 0 else "priority_right"
                return find_action(simulator.config, chosen)
            return find_action(simulator.config, "free_release")

        # ── Balanced-queue guard ──
        # When queues are nearly equal (balanced chaotic scenarios),
        # alternating/settle HURT throughput because they block one
        # side without draining the other faster.  Default to free flow.
        balanced_queues = abs(imbalance) < 0.18

        # Early post-reopening: settle if disorder is severe AND queues unbalanced
        if (
            not balanced_queues
            and early_reopen
            and (
                (severe_wrong_side and snapshot.disorder_index > 0.4)
                or snapshot.disorder_index > 0.55
            )
            and total_queue > 10
        ):
            return find_action(simulator.config, "settle_then_alt")

        # Early post-reopening: alternating if extreme overcrowding AND unbalanced
        if (
            not balanced_queues
            and early_reopen
            and total_queue > 22
            and snapshot.occupancy_risk > 0.78
            and snapshot.disorder_index > 0.4
        ):
            return find_action(simulator.config, "alternating_6s")

        # Transition to free flow when conditions ease
        if snapshot.time_since_open > 8.0 and snapshot.occupancy_risk < 0.80:
            if abs(imbalance) > 0.14 and total_queue > 10:
                chosen = "priority_left" if imbalance > 0 else "priority_right"
                return find_action(simulator.config, chosen)
            return find_action(simulator.config, "free_release")

        # High risk with severe disorder AND unbalanced: brief alternating
        if not balanced_queues and total_queue > 20 and snapshot.occupancy_risk > 0.82:
            return find_action(simulator.config, "alternating_6s")

        if not balanced_queues and snapshot.disorder_index > 0.55 and snapshot.occupancy_risk > 0.80:
            return find_action(simulator.config, "alternating_4s")

        return find_action(simulator.config, "free_release")


@dataclass
class HybridAdaptivePolicy:
    model: object | None
    bandit: LinUCBResidual | None
    config: AdaptivePolicyConfig = field(default_factory=AdaptivePolicyConfig)
    training_mode: bool = False

    def decide(self, simulator) -> PolicyAction:
        snapshot = simulator.build_snapshot()
        if self.model is None or not getattr(self.model, "is_fitted", False) or self.bandit is None:
            fallback = AdaptivePolicy(model=None, config=self.config)
            return fallback._decide_heuristically(simulator, snapshot)

        action_rows: dict[str, dict[str, float]] = {}
        scored_actions: list[dict[str, Any]] = []
        survivors: list[tuple[float, PolicyAction]] = []
        risk_order: list[tuple[float, PolicyAction]] = []
        default_clearance = float(simulator.config.prediction_horizon)

        for action in simulator.config.actions:
            row = StateVectorBuilder.build(snapshot, action.name, simulator.config)
            action_rows[action.name] = row
            mean_prediction, std_prediction = self.model.predict_row_with_uncertainty(row)
            outcome = _prediction_to_outcome(mean_prediction, snapshot, default_clearance)
            base_utility = self.config.reward(outcome)
            utility_std = _utility_std(std_prediction, self.config)
            residual_mean = self.bandit.mean(row)
            residual_bonus = self.bandit.bonus(row)
            score = (
                base_utility
                + residual_mean
                + residual_bonus
                - self.config.uncertainty_penalty_weight * utility_std
                + _priority_adjustment(action, snapshot, self.config)
            )
            veto_reason = self._veto_reason(action, snapshot, outcome)
            scored_actions.append(
                {
                    "action": action.name,
                    "predicted": mean_prediction,
                    "std": std_prediction,
                    "base_utility": round(base_utility, 3),
                    "utility_std": round(utility_std, 3),
                    "linucb_mean": round(residual_mean, 3),
                    "linucb_bonus": round(residual_bonus, 3),
                    "score": round(score, 3),
                    "veto_reason": veto_reason,
                }
            )
            risk_order.append((outcome.occupancy_risk_horizon, action))
            if veto_reason is None:
                survivors.append((score, action))

        if survivors:
            _, chosen_action = max(survivors, key=lambda item: item[0])
        else:
            _, chosen_action = min(risk_order, key=lambda item: item[0])

        trace = DecisionTrace(
            time=round(snapshot.time, 2),
            chosen_action=chosen_action.name,
            state_summary=StateVectorBuilder.state_summary(snapshot),
            action_scores=scored_actions,
        )
        if hasattr(simulator, "record_decision_trace"):
            simulator.record_decision_trace(trace)

        if self.training_mode:
            chosen_row = action_rows[chosen_action.name]
            mean_prediction, _ = self.model.predict_row_with_uncertainty(chosen_row)
            predicted_outcome = _prediction_to_outcome(mean_prediction, snapshot, default_clearance)
            actual_outcome = simulator.evaluate_horizon(
                simulator.config.prediction_horizon,
                FixedActionPolicy(chosen_action.name),
            )
            residual_reward = self.config.reward(actual_outcome) - self.config.reward(predicted_outcome)
            self.bandit.update(chosen_row, residual_reward)

        return chosen_action

    def _veto_reason(
        self,
        action: PolicyAction,
        snapshot: CrossingStateSnapshot,
        outcome: HorizonOutcome,
    ) -> str | None:
        if outcome.occupancy_risk_horizon > self.config.shield_occupancy_threshold:
            return "occupancy-risk"
        total_ready = snapshot.queue_counts[LEFT] > self.config.fairness_queue_threshold and snapshot.queue_counts[RIGHT] > self.config.fairness_queue_threshold
        if total_ready and outcome.fairness_gap_horizon > self.config.shield_fairness_threshold:
            return "fairness-gap"
        if action.mode == "free" and outcome.wrong_side_queue_share_horizon > (
            self.config.free_mode_wrong_side_threshold + 0.1
        ):
            return "wrong-side-risk"
        return None
