from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .metrics import compare_results, improvement_summary


def _safe_pct(value: float) -> str:
    return f"{value:.1f}%"


def _safe_num(value: float) -> str:
    return f"{value:.2f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _adaptive_label(results: dict) -> str:
    for candidate in ("MCTS Rollout", "Hybrid Adaptive", "Legacy Adaptive", "Adaptive Staged Release"):
        if candidate in results:
            return candidate
    raise KeyError("No adaptive-style policy found in results.")


def summarize_results(results: dict, scenario_name: str) -> dict[str, object]:
    comparison = pd.DataFrame(compare_results(results))
    baseline = results["Free Flow"].metrics
    improvement_rows = []
    for label, result in results.items():
        if label == "Free Flow":
            continue
        row = {"policy": label}
        row.update(improvement_summary(baseline, result.metrics))
        improvement_rows.append(row)
    improvements = pd.DataFrame(improvement_rows)
    adaptive_label = _adaptive_label(results)
    adaptive = results[adaptive_label].metrics
    adaptive_row = improvements[improvements["policy"] == adaptive_label].iloc[0]
    best_policy = comparison.sort_values(
        by=["throughput", "average_waiting_time"],
        ascending=[False, True],
    ).iloc[0]["policy"]

    scenario_descriptions = {
        "light": "Short closures and manageable demand. Use this to show calm behavior when intervention is unnecessary.",
        "peak": "Balanced, high-demand traffic with visible queue buildup. This is the best primary judging scenario.",
        "chaotic": "Long closures and disorderly release pressure. Use this for the most dramatic before-vs-after demo.",
        "peak_left_skew": "Peak traffic with stronger left-side pressure, useful for fairness and priority demonstrations.",
        "peak_right_skew": "Peak traffic with stronger right-side pressure, useful for fairness and priority demonstrations.",
        "chaotic_aggressive": "Chaotic reopening with more assertive and reckless drivers, useful for robustness evidence.",
        "chaotic_long_gate": "Longer gate closures that stress queue growth and post-open recovery.",
    }

    return {
        "scenario_name": scenario_name,
        "adaptive_label": adaptive_label,
        "best_policy": best_policy,
        "adaptive_delay_gain_pct": float(adaptive_row["waiting_time_improvement_pct"]),
        "adaptive_throughput_gain_pct": float(adaptive_row["throughput_improvement_pct"]),
        "adaptive_congestion_gain_pct": float(adaptive_row["congestion_improvement_pct"]),
        "free_flow_delay": baseline.average_waiting_time,
        "adaptive_delay": adaptive.average_waiting_time,
        "free_flow_throughput": baseline.throughput,
        "adaptive_throughput": adaptive.throughput,
        "free_flow_congestion": baseline.max_congestion_length,
        "adaptive_congestion": adaptive.max_congestion_length,
        "free_flow_risk": baseline.occupancy_risk,
        "adaptive_risk": adaptive.occupancy_risk,
        "scenario_description": scenario_descriptions.get(scenario_name, ""),
        "impact_headline": (
            f"{adaptive_label} cuts delay by {_safe_pct(adaptive_row['waiting_time_improvement_pct'])} "
            f"while increasing throughput by {_safe_pct(adaptive_row['throughput_improvement_pct'])} "
            f"in the {scenario_name} scenario."
        ),
        "technical_headline": (
            "The simulator estimates queue disorder, crossing occupancy pressure, and bidirectional imbalance, "
            "then chooses the next release action using counterfactual prediction, uncertainty, and a safety shield."
        ),
        "judge_hooks": [
            "Impact: the bottleneck remains congested after the barrier opens because human release behavior is unstructured.",
            "Technical: the controller scores counterfactual actions, adds a contextual bandit residual, and vetoes unsafe actions.",
            "Originality: the project combines a railway-crossing microsimulator with a safety-shielded counterfactual bandit rather than generic traffic prediction.",
        ],
        "comparison_frame": comparison,
        "improvement_frame": improvements,
    }


def build_assumptions_markdown() -> str:
    return """# QuarryFlow Assumptions

## Scope

- Single bottleneck type: urban railway crossing with two opposing road approaches and one shared crossing box.
- Data source: simulated, not sensor-collected.
- Goal: improve post-barrier reopening flow, not redesign infrastructure.

## Traffic Rules Assumed

- Barrier closures fully block both directions.
- Vehicles respect physical spacing and cannot overlap leaders.
- Only one side can safely dominate entry to the crossing box at a time.

## Behavior Model

- Mixed vehicles: car, bike, auto-rickshaw.
- Mixed driver archetypes: cautious, compliant, opportunistic, assertive, aggressive, reckless.
- Disorder near the gate is modeled through lateral spread, wrong-side squeezing, gate-rush pressure, reaction-time delay, and conflict pressure.

## What Is Learned

- The surrogate ensemble predicts short-horizon action outcomes.
- The LinUCB residual learns state-dependent action preference from simulator reward.

## What Is Hand-Modeled

- Road geometry, closure schedule, vehicle specifications, driver archetypes, safety shield thresholds, and the action set.
"""


def build_pitch_markdown(results: dict, scenario_name: str, *, model_label: str) -> str:
    summary = summarize_results(results, scenario_name)
    comparison_table = _markdown_table(summary["comparison_frame"])
    improvement_table = _markdown_table(summary["improvement_frame"])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# QuarryFlow Crossing Pitch Brief

Generated: {generated_at}

## Scenario

- Preset: `{scenario_name}`
- Context: {summary['scenario_description']}
- Model mode: `{model_label}`

## One-Line Story

{summary['impact_headline']}

## Comparison

{comparison_table}

## Improvement vs Free Flow

{improvement_table}

## Technical Notes

- Baselines: `Free Flow`, `Static Alternating`
- Visible adaptive benchmark: `Legacy Adaptive`
- Primary hybrid controller: `Hybrid Adaptive`
- Current model backend: `{model_label}`
- Innovation: safety-shielded counterfactual bandit over short-horizon simulator rollouts
"""


def build_judge_packet_markdown(
    results: dict,
    scenario_name: str,
    *,
    model_label: str,
    assumptions_markdown: str,
    learning_curve: pd.DataFrame | None = None,
    holdout_summary: pd.DataFrame | None = None,
) -> str:
    summary = summarize_results(results, scenario_name)
    learning_table = _markdown_table(learning_curve if learning_curve is not None else pd.DataFrame())
    holdout_table = _markdown_table(holdout_summary if holdout_summary is not None else pd.DataFrame())
    return f"""# QuarryFlow Judge Packet

## Project Summary

{summary['impact_headline']}

## Challenge Fit

- Problem 01: single railway-crossing bottleneck with explicit assumptions and measurable congestion metrics
- Problem 02: adaptive controller with measurable learning evidence over curriculum iterations
- Creativity: safety-shielded counterfactual bandit on top of a mixed-traffic microsimulator

## Live Demo Story

- Open with `peak`
- Show `Hybrid Adaptive` versus `Legacy Adaptive` and baselines
- Switch to `chaotic` for the strongest visual contrast

## Holdout Benchmark

{holdout_table}

## Learning Curve

{learning_table}

## Assumptions

{assumptions_markdown}
"""


def write_text_report(content: str, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def write_pitch_report(
    results: dict,
    scenario_name: str,
    output_path: str | Path,
    *,
    model_label: str,
) -> Path:
    return write_text_report(build_pitch_markdown(results, scenario_name, model_label=model_label), output_path)
