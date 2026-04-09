"""Pre-compute sensitivity analysis data for the dashboard.

Runs simulations across parameter variations and saves results as CSVs
so the dashboard can load them instantly without live computation.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.policy import AdaptivePolicy, FreeFlowPolicy, StaticAlternatingPolicy
from quarryflow.scenarios import build_scenario
from quarryflow.simulator import RailwayCrossingSimulator


def _run_one(config, seed: int, policy, label: str) -> dict:
    sim = RailwayCrossingSimulator(config, seed=seed)
    result = sim.run_episode(policy, record_history=False)
    m = result.metrics
    return {
        "policy": label,
        "average_waiting_time": round(m.average_waiting_time, 2),
        "throughput": round(m.throughput, 2),
        "max_congestion_length": round(m.max_congestion_length, 2),
        "clearance_time": round(m.clearance_time, 2),
        "fairness_gap": round(m.fairness_gap, 3),
        "conflict_count": m.conflict_count,
        "vehicles_cleared": m.vehicles_cleared,
        "disorder_peak": round(m.disorder_peak, 3),
    }


def arrival_rate_sweep() -> pd.DataFrame:
    """Vary arrival rates and measure delay for Free Flow vs Adaptive."""
    rows = []
    seed = 11
    for left_rate in [10, 14, 18, 22, 26]:
        for right_rate in [10, 14, 18, 22, 26]:
            config = build_scenario("peak", seed=seed)
            config.arrival_rate_per_minute = {"left": float(left_rate), "right": float(right_rate)}
            for label, policy in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
                row = _run_one(config, seed, policy, label)
                row["arrival_left"] = left_rate
                row["arrival_right"] = right_rate
                rows.append(row)
    return pd.DataFrame(rows)


def aggression_sweep() -> pd.DataFrame:
    """Shift driver mix toward aggressive and measure impact."""
    rows = []
    seed = 11
    mixes = [
        ("Calm", {"cautious": 0.30, "compliant": 0.35, "opportunistic": 0.20, "assertive": 0.10, "aggressive": 0.04, "reckless": 0.01}),
        ("Normal", {"cautious": 0.08, "compliant": 0.22, "opportunistic": 0.34, "assertive": 0.16, "aggressive": 0.14, "reckless": 0.06}),
        ("Aggressive", {"cautious": 0.03, "compliant": 0.08, "opportunistic": 0.20, "assertive": 0.24, "aggressive": 0.30, "reckless": 0.15}),
        ("Reckless", {"cautious": 0.01, "compliant": 0.04, "opportunistic": 0.10, "assertive": 0.20, "aggressive": 0.35, "reckless": 0.30}),
    ]
    for mix_label, mix in mixes:
        config = build_scenario("peak", seed=seed)
        config.driver_mix = deepcopy(mix)
        for label, policy in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
            row = _run_one(config, seed, policy, label)
            row["driver_mix"] = mix_label
            rows.append(row)
    return pd.DataFrame(rows)


def closure_duration_sweep() -> pd.DataFrame:
    """Vary gate closure duration and measure delay penalty."""
    rows = []
    seed = 11
    for extra_seconds in [0, 15, 30, 45, 60, 90]:
        config = build_scenario("peak", seed=seed)
        config.train_closures = [
            (85.0, 145.0 + extra_seconds),
            (275.0, 345.0 + extra_seconds),
        ]
        config.episode_seconds = max(config.episode_seconds, int(345 + extra_seconds + 200))
        for label, policy in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
            row = _run_one(config, seed, policy, label)
            row["closure_duration_s"] = 60 + extra_seconds
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ROOT / "artifacts" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[sensitivity] arrival rate sweep ...")
    arrival_rate_sweep().to_csv(out_dir / "arrival_rate_sweep.csv", index=False)

    print("[sensitivity] aggression sweep ...")
    aggression_sweep().to_csv(out_dir / "aggression_sweep.csv", index=False)

    print("[sensitivity] closure duration sweep ...")
    closure_duration_sweep().to_csv(out_dir / "closure_duration_sweep.csv", index=False)

    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
