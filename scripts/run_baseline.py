from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy, StaticAlternatingPolicy
from quarryflow.scenarios import build_scenario
from quarryflow.simulator import RailwayCrossingSimulator


POLICIES = {
    "free": FreeFlowPolicy,
    "alternating": StaticAlternatingPolicy,
    "adaptive": AdaptivePolicy,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="peak")
    parser.add_argument("--policy", choices=POLICIES, default="free")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    config = build_scenario(args.scenario, seed=args.seed)
    simulator = RailwayCrossingSimulator(config, seed=args.seed)
    result = simulator.run_episode(POLICIES[args.policy]())
    print(json.dumps(result.metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
