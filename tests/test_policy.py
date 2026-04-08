from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.policy import AdaptivePolicy
from quarryflow.scenarios import build_scenario
from quarryflow.domain_types import CrossingStateSnapshot, LEFT, RIGHT


class FakeSimulator:
    def __init__(self):
        self.config = build_scenario("peak", seed=11)

    def build_snapshot(self) -> CrossingStateSnapshot:
        return CrossingStateSnapshot(
            time=120.0,
            barrier_closed=False,
            time_since_open=8.0,
            time_to_open=0.0,
            time_to_close=0.0,
            queue_lengths={LEFT: 45.0, RIGHT: 120.0},
            queue_counts={LEFT: 8, RIGHT: 22},
            mean_speed=1.2,
            disorder_index=0.52,
            bike_infiltration=0.4,
            wrong_side_queue_share=0.14,
            dilemma_zone_pressure=0.0,
            mean_reaction_time_near_gate=1.1,
            closure_frustration_index=0.44,
            idling_vehicle_share=0.9,
            idling_fuel_rate_lph=2.7,
            idling_co2_rate_kgph=5.9,
            crossing_occupancy=0,
            occupancy_risk=0.6,
            pressure_imbalance=-0.45,
            aggressive_share_near_gate=0.35,
            current_action="free_release",
            conflict_count=3,
            vehicles_spawned=100,
            vehicles_cleared=60,
        )


class PolicyTests(unittest.TestCase):
    def test_adaptive_prefers_priority_for_heavy_right_queue(self) -> None:
        action = AdaptivePolicy().decide(FakeSimulator())
        self.assertEqual(action.name, "priority_right")


if __name__ == "__main__":
    unittest.main()
