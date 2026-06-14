from __future__ import annotations

import time
import math
from typing import Dict, Optional, Tuple

from .domain_types import PolicyAction, CrossingStateSnapshot
from .simulator import RailwayCrossingSimulator
from .policy import AdaptivePolicy

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
        rollout_duration_seconds: float = 10.0,  # K: how far into the future to look
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
        return -(
            self.w_wait * outcome.average_waiting_time +
            self.w_queue * outcome.max_congestion_length +
            self.w_fairness * outcome.fairness_gap_horizon
        )

    def decide(self, simulator: RailwayCrossingSimulator) -> PolicyAction:
        snapshot = simulator.build_snapshot()
        candidate_names = self.heuristic_filter._rollout_candidates(simulator, snapshot)
        
        # If the heuristic only returns 1 valid action, skip the rollout entirely!
        if len(candidate_names) == 1:
            name = list(candidate_names)[0]
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
                    clone = simulator.clone()
                    # Run the clone forward in time using this specific action
                    outcome = clone.evaluate_horizon(
                        duration=self.rollout_duration,
                        policy=FixedActionPolicy(action)
                    )
                    total_score += self._score_outcome(outcome)
                
                # Average the stochastic rollouts
                score = total_score / self.M
                self.ttable[state_hash] = score
                
            # Keep track of the best action
            if score > best_score:
                best_score = score
                best_action = action

        # Clear Transposition table periodically to prevent memory leaks if needed,
        # but for a 10-minute episode, keeping it is fine.
        if len(self.ttable) > 10000:
            self.ttable.clear()

        return best_action
