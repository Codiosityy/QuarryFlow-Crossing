from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.metrics import improvement_summary
from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy
from quarryflow.scenarios import build_scenario
from quarryflow.simulator import RailwayCrossingSimulator


class MetricTests(unittest.TestCase):
    def test_improvement_summary_marks_chaotic_adaptive_gain(self) -> None:
        config = build_scenario("chaotic", seed=11)
        free = RailwayCrossingSimulator(config, seed=11).run_episode(FreeFlowPolicy())
        adaptive = RailwayCrossingSimulator(config, seed=11).run_episode(AdaptivePolicy())
        summary = improvement_summary(free.metrics, adaptive.metrics)
        self.assertGreater(summary["waiting_time_improvement_pct"], 0.0)
        self.assertGreater(summary["throughput_improvement_pct"], 0.0)
        self.assertGreater(summary["congestion_improvement_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
