"""Export simulation results to JSON for the 2D Canvas visualizer.

Usage (from project root):
    python scripts/export_viz.py --scenario peak --out docs/visualizer/data/
    python scripts/export_viz.py --scenario all --out docs/visualizer/data/
    python scripts/export_viz.py --scenario chaotic --policies "Free Flow,Hybrid Adaptive"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from quarryflow.dashboard_data import run_policy_suite
from quarryflow.domain_types import LEFT, RIGHT, SimulationResult
from quarryflow.scenarios import build_scenario, list_scenarios

FORMAT_VERSION = "1.0"

# ── Coordinate mapping (mirrors build_snapshot_figure in streamlit_app.py) ──

APPROACH_LENGTH = 100.0
CROSSING_BOX_LENGTH = 12.0
HALF_BOX = CROSSING_BOX_LENGTH / 2.0


def _progress_to_x(progress: float, side: str) -> float:
    """Map vehicle progress to canvas x-coordinate.

    LEFT vehicles:  progress 0 → x=-106, progress 100 → x=-6, progress 192 → x=+86
    RIGHT vehicles: progress 0 → x=+106, progress 100 → x=+6, progress 192 → x=-86
    """
    offset = progress - APPROACH_LENGTH - HALF_BOX
    return offset if side == LEFT else -offset


def _lateral_to_y(lateral_offset: float, side: str) -> float:
    """Map lateral offset to canvas y-coordinate.

    LEFT:  y = 1.1 + lateral_offset  (upper lane)
    RIGHT: y = -1.1 + lateral_offset (lower lane)
    """
    base = 1.1 if side == LEFT else -1.1
    return base + lateral_offset


def export_frame(
    time: float,
    barrier_closed: bool,
    action: str,
    vehicles: list[dict],
    snapshot: object | None = None,
) -> dict:
    """Convert a single simulation frame to the export format."""
    vehicle_data = []
    for v in vehicles:
        progress = v["progress"]
        side = v["side"]
        vehicle_data.append({
            "id": v["vehicle_id"],
            "side": side,
            "class": v["vehicle_class"],
            "x": round(_progress_to_x(progress, side), 2),
            "y": round(_lateral_to_y(v["lateral_offset"], side), 2),
            "speed": round(v["speed"], 2),
            "lateral_offset": round(v["lateral_offset"], 3),
        })

    metrics = {}
    if snapshot is not None:
        metrics = {
            "queue_left": snapshot.queue_counts[LEFT],
            "queue_right": snapshot.queue_counts[RIGHT],
            "queue_length_left": round(snapshot.queue_lengths[LEFT], 1),
            "queue_length_right": round(snapshot.queue_lengths[RIGHT], 1),
            "disorder_index": round(snapshot.disorder_index, 4),
            "occupancy_risk": round(snapshot.occupancy_risk, 4),
            "wrong_side_queue_share": round(snapshot.wrong_side_queue_share, 4),
            "crossing_occupancy": snapshot.crossing_occupancy,
            "conflict_count": snapshot.conflict_count,
        }

    return {
        "time": round(time, 2),
        "barrier_closed": barrier_closed,
        "action": action,
        "vehicles": vehicle_data,
        "metrics": metrics,
    }


def export_for_visualization(
    result: SimulationResult,
    scenario_name: str,
    policy_label: str,
    *,
    episode_seconds: int = 600,
    time_step: float = 0.5,
) -> dict:
    """Convert a SimulationResult to the JSON export format.

    Args:
        result: The simulation result to export.
        scenario_name: Name of the scenario (e.g. "peak", "chaotic").
        policy_label: Label of the policy (e.g. "Free Flow", "Hybrid Adaptive").
        episode_seconds: Total episode duration for metadata.
        time_step: Simulation time step for metadata.

    Returns:
        Dict ready for JSON serialization.
    """
    frames = []
    history = result.history
    snapshots = result.snapshots

    # Build a lookup from time → snapshot for metrics
    snapshot_by_time = {round(s.time, 2): s for s in snapshots}

    for i, record in enumerate(history):
        t = record["time"]
        snapshot = snapshot_by_time.get(round(t, 2))
        frame = export_frame(
            time=t,
            barrier_closed=record["barrier_closed"],
            action=record.get("current_action", "unknown"),
            vehicles=record.get("vehicles", []),
            snapshot=snapshot,
        )
        frames.append(frame)

    return {
        "format_version": FORMAT_VERSION,
        "scenario": scenario_name,
        "policy": policy_label,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "episode_seconds": episode_seconds,
            "time_step": time_step,
            "total_frames": len(frames),
            "total_vehicles_spawned": result.metrics.vehicles_spawned,
            "total_vehicles_cleared": result.metrics.vehicles_cleared,
        },
        "train_closures": [],  # Filled by caller if needed
        "frames": frames,
    }


def validate_export(data: dict) -> list[str]:
    """Lightweight validation of exported data. Returns list of issues."""
    issues = []
    frames = data.get("frames", [])
    if not frames:
        issues.append("No frames in export")
        return issues

    valid_classes = {"car", "bike", "auto"}
    prev_time = -1.0
    for i, frame in enumerate(frames):
        t = frame.get("time", -1)
        if t < prev_time:
            issues.append(f"Frame {i}: non-monotonic time {t} < {prev_time}")
        prev_time = t

        for v in frame.get("vehicles", []):
            if v.get("class") not in valid_classes:
                issues.append(f"Frame {i}: invalid vehicle class '{v.get('class')}'")
            if v.get("progress", 0) < -10:
                issues.append(f"Frame {i}: negative progress {v.get('progress')}")

    return issues


def export_scenario(
    scenario_name: str,
    output_dir: Path,
    *,
    seed: int = 7,
    policies: list[str] | None = None,
    fast_mode: bool = False,
) -> list[Path]:
    """Run a scenario and export all policy results to JSON files.

    Args:
        scenario_name: Name of the scenario preset.
        output_dir: Directory to write JSON files.
        seed: Random seed for reproducibility.
        policies: Optional list of policy labels to export. None = all.

    Returns:
        List of paths to generated JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_scenario(scenario_name, seed=seed)

    results = run_policy_suite(
        scenario_name,
        seed=seed,
        record_history=True,
        fast_mode=fast_mode,
    )

    # Add train_closures to config for the metadata
    train_closures = config.train_closures
    generated = []

    for label, result in results.items():
        if policies and label not in policies:
            continue

        data = export_for_visualization(
            result,
            scenario_name=scenario_name,
            policy_label=label,
            episode_seconds=config.episode_seconds,
            time_step=config.time_step,
        )
        data["train_closures"] = [[s, e] for s, e in train_closures]

        issues = validate_export(data)
        if issues:
            print(f"  WARN Validation issues for {label}: {issues}", file=sys.stderr)

        filename = f"{scenario_name}_{label.lower().replace(' ', '_')}.json"
        out_path = output_dir / filename
        out_path.write_text(json.dumps(data, indent=None), encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"  OK {label}: {len(data['frames'])} frames, {size_kb:.0f} KB")
        generated.append(out_path)

    return generated


def _build_index(output_dir: Path, scenario_names: list[str]) -> None:
    """Write an index.json listing all available data files."""
    index = {"format_version": FORMAT_VERSION, "scenarios": {}}

    for scenario_dir in output_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        policies = []
        for f in sorted(scenario_dir.glob("*.json")):
            if f.name == "index.json":
                continue
            policies.append(f.stem.replace(f"{scenario}_", ""))
        if policies:
            index["scenarios"][scenario] = policies

    # Also check flat structure (single directory with all files)
    json_files = sorted(output_dir.glob("*.json"))
    if json_files and not index["scenarios"]:
        for f in json_files:
            stem = f.stem
            # Match against known scenario names (longest match first)
            matched = False
            for scenario in sorted(scenario_names, key=len, reverse=True):
                prefix = f"{scenario}_"
                if stem.startswith(prefix):
                    policy = stem[len(prefix):]
                    if policy:
                        index["scenarios"].setdefault(scenario, []).append(policy)
                    matched = True
                    break
            if not matched and stem != "index":
                # Fallback: treat as single scenario
                index["scenarios"].setdefault("unknown", []).append(stem)

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"  Index written to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export simulation results to JSON for the 2D Canvas visualizer."
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario name or 'all' to export all scenarios (default: all)",
    )
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated policy labels to export (default: all)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "visualizer" / "data"),
        help="Output directory for JSON files (default: docs/visualizer/data/)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (shorter episode, larger time step) for quick exports",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    policy_list = [p.strip() for p in args.policies.split(",")] if args.policies else None

    if args.scenario == "all":
        scenarios = list_scenarios()
    else:
        scenarios = [args.scenario]

    print(f"Exporting visualization data to {output_dir}/")
    all_generated = []
    for scenario in scenarios:
        print(f"\n  [{scenario}]")
        try:
            generated = export_scenario(
                scenario, output_dir, seed=args.seed, policies=policy_list,
                fast_mode=args.fast,
            )
            all_generated.extend(generated)
        except Exception as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)

    if all_generated:
        _build_index(output_dir, scenarios)
        print(f"\n  DONE: Exported {len(all_generated)} files to {output_dir}/")
    else:
        print("\n⚠ No files exported.", file=sys.stderr)


if __name__ == "__main__":
    main()
