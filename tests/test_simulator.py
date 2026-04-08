from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy
from quarryflow.scenarios import build_scenario
from quarryflow.simulator import RailwayCrossingSimulator


class SimulatorTests(unittest.TestCase):
    def test_peak_scenario_spawns_and_clears_vehicles(self) -> None:
        config = build_scenario("peak", seed=11)
        result = RailwayCrossingSimulator(config, seed=11).run_episode(FreeFlowPolicy())
        self.assertGreater(result.metrics.vehicles_spawned, 0)
        self.assertGreater(result.metrics.vehicles_cleared, 0)
        self.assertGreater(result.metrics.max_congestion_length, 0.0)

    def test_chaotic_has_more_congestion_than_light(self) -> None:
        light = RailwayCrossingSimulator(build_scenario("light", seed=11), seed=11).run_episode(
            FreeFlowPolicy()
        )
        chaotic = RailwayCrossingSimulator(build_scenario("chaotic", seed=11), seed=11).run_episode(
            FreeFlowPolicy()
        )
        self.assertGreater(
            chaotic.metrics.max_congestion_length,
            light.metrics.max_congestion_length,
        )

    def test_adaptive_outperforms_free_on_peak(self) -> None:
        config = build_scenario("peak", seed=11)
        free = RailwayCrossingSimulator(config, seed=11).run_episode(FreeFlowPolicy())
        adaptive = RailwayCrossingSimulator(config, seed=11).run_episode(AdaptivePolicy())
        self.assertLess(adaptive.metrics.average_waiting_time, free.metrics.average_waiting_time)
        self.assertGreater(adaptive.metrics.throughput, free.metrics.throughput)

    def test_snapshot_exposes_behavioral_crossing_features(self) -> None:
        config = build_scenario("chaotic", seed=11)
        simulator = RailwayCrossingSimulator(config, seed=11)
        simulator.run_for(190.0, FreeFlowPolicy(), record_history=False)
        snapshot = simulator.build_snapshot()
        self.assertGreaterEqual(snapshot.wrong_side_queue_share, 0.0)
        self.assertGreaterEqual(snapshot.closure_frustration_index, 0.0)
        self.assertGreaterEqual(snapshot.idling_vehicle_share, 0.0)

    def test_chaotic_generates_more_wrong_side_pressure_than_light(self) -> None:
        light = RailwayCrossingSimulator(build_scenario("light", seed=11), seed=11).run_episode(
            FreeFlowPolicy()
        )
        chaotic = RailwayCrossingSimulator(build_scenario("chaotic", seed=11), seed=11).run_episode(
            FreeFlowPolicy()
        )
        self.assertGreater(
            chaotic.metrics.wrong_side_queue_peak,
            light.metrics.wrong_side_queue_peak,
        )
        self.assertGreater(
            chaotic.metrics.total_idling_fuel_liters,
            light.metrics.total_idling_fuel_liters,
        )


if __name__ == "__main__":
    unittest.main()
