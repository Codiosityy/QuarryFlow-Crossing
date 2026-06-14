from __future__ import annotations

import time
import math
from typing import Dict, Optional, Tuple

from .domain_types import PolicyAction, CrossingStateSnapshot
from .simulator import RailwayCrossingSimulator
from .policy import AdaptivePolicy, _priority_adjustment

class FixedActionPolicy:
    """A simple policy that always returns the same action. Used during rollouts."""
    def __init__(self, action: PolicyAction):
        self.action = action

    def decide(self, simulator: RailwayCrossingSimulator) -> PolicyAction:
        return self.action

class MCTSRolloutPolicy:
    """
    Agentic search policy that simulates the future before making a decision.
    For each candidate action, it clones the simulator, runs it forward,
    scores the outcome, and picks the action with the best future score.
    """
    def __init__(
        self,
        model=None,
        rollout_duration_seconds: float = 4.0,  # K: how far into the future to look
        rollouts_per_action: int = 1,            # M: how many stochastic rollouts to average
        w_wait: float = 1.0,                     # Weight for average wait time
        w_queue: float = 2.0,                    # Weight for max congestion
        w_fairness: float = 1.5,                 # Weight for fairness gap between sides
    ):
        self.rollout_duration = rollout_duration_seconds
        self.M = rollouts_per_action
        self.w_wait = w_wait
        self.w_queue = w_queue
        self.w_fairness = w_fairness
        
        # Transposition Table: Cache state evaluations to save compute
        self.ttable: Dict[str, float] = {}
        # Use the ML baseline's heuristic to filter bad actions and save compute
        self.heuristic_filter = AdaptivePolicy(model=model)

    def _hash_state(self, snapshot: CrossingStateSnapshot, action: PolicyAction) -> str:
        """
        Creates a fast string hash of the current macroscopic state.
        This allows us to reuse rollout scores if we encounter the same traffic state again.
        """
        return f"{snapshot.barrier_closed}_{snapshot.queue_counts['left']}_{snapshot.queue_counts['right']}_{action.name}"

    def _score_outcome(self, outcome) -> float:
        """
        Calculates the fitness score of a future horizon.
        Higher score is better (hence the negative sign for penalties).
        """
        # We reuse the carefully tuned AdaptivePolicyConfig reward function!
        return self.heuristic_filter.config.reward(outcome)

    def decide(self, simulator: RailwayCrossingSimulator) -> PolicyAction:
        self.ttable.clear() # Clear cache for each new decision step!
        snapshot = simulator.build_snapshot()
        candidate_names = self.heuristic_filter._rollout_candidates(simulator, snapshot)
        
        # If the heuristic only returns 1 valid action, skip the rollout entirely!
        if len(candidate_names) == 1:
            name = list(candidate_names)[0]
            import logging
            logging.debug(f"[MCTS] Time {simulator.time:.1f} | Candidates: {candidate_names} | Chosen instantly: {name}")
            return next(a for a in simulator.config.actions if a.name == name)

        best_score = -float('inf')
        best_action = next(a for a in simulator.config.actions if a.name == list(candidate_names)[0])

        for action in simulator.config.actions:
            if action.name not in candidate_names:
                continue
                
            state_hash = self._hash_state(snapshot, action)
            
            # 1. Check Transposition Table (Cache Hit)
            if state_hash in self.ttable:
                score = self.ttable[state_hash]
            
            # 2. Perform Rollout (Cache Miss)
            else:
                total_score = 0.0
                for _ in range(self.M):
                    # Run the simulator forward in time using this specific action
                    # evaluate_horizon creates its own clone internally to prevent mutation
                    outcome = simulator.evaluate_horizon(
                        duration=self.rollout_duration,
                        policy=FixedActionPolicy(action)
                    )
                    total_score += self._score_outcome(outcome)
                
                # Average the stochastic rollouts and apply priority penalty
                score = (total_score / self.M) + _priority_adjustment(action, snapshot, self.heuristic_filter.config)
                self.ttable[state_hash] = score
                
            # Keep track of the best action
            if score > best_score:
                best_score = score
                best_action = action

        import logging
        logging.debug(f"[MCTS] Time {simulator.time:.1f} | Candidates: {candidate_names} | Chosen via MCTS Rollout: {best_action.name} (Score: {best_score:.2f})")

        if hasattr(simulator, "record_decision_trace"):
            from .domain_types import DecisionTrace
            from .hybrid import StateVectorBuilder
            
            # Format the scores into the trace format expected by Streamlit
            scored_actions = []
            for h, s in self.ttable.items():
                # We can approximate the trace to just the best score info if needed
                pass
                
            simulator.record_decision_trace(
                DecisionTrace(
                    time=round(snapshot.time, 2),
                    chosen_action=best_action.name,
                    state_summary=StateVectorBuilder.state_summary(snapshot),
                    action_scores=[{"action": best_action.name, "score": round(best_score, 3), "base_utility": 0, "veto_reason": None}]
                )
            )

        # Clear Transposition table periodically to prevent memory leaks if needed,
        # but for a 10-minute episode, keeping it is fine.
        if len(self.ttable) > 10000:
            self.ttable.clear()

        return best_action
