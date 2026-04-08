from __future__ import annotations

from .domain_types import EpisodeMetrics, SimulationResult


def compare_results(results: dict[str, SimulationResult]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, result in results.items():
        metrics = result.metrics
        rows.append(
            {
                "policy": label,
                "average_waiting_time": round(metrics.average_waiting_time, 2),
                "throughput": round(metrics.throughput, 2),
                "max_congestion_length": round(metrics.max_congestion_length, 2),
                "clearance_time": round(metrics.clearance_time, 2),
                "worst_clearance_time": round(metrics.worst_clearance_time, 2),
                "conflict_count": metrics.conflict_count,
                "occupancy_risk": round(metrics.occupancy_risk, 3),
                "fairness_gap": round(metrics.fairness_gap, 3),
                "wrong_side_queue_peak": round(metrics.wrong_side_queue_peak, 3),
                "dilemma_zone_peak": round(metrics.dilemma_zone_peak, 3),
                "total_idling_fuel_liters": round(metrics.total_idling_fuel_liters, 3),
            }
        )
    return rows


def improvement_summary(
    baseline: EpisodeMetrics,
    candidate: EpisodeMetrics,
) -> dict[str, float]:
    def safe_improvement(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return (old - new) / old * 100.0

    return {
        "waiting_time_improvement_pct": safe_improvement(
            baseline.average_waiting_time,
            candidate.average_waiting_time,
        ),
        "throughput_improvement_pct": (
            0.0
            if baseline.throughput == 0
            else (candidate.throughput - baseline.throughput) / baseline.throughput * 100.0
        ),
        "congestion_improvement_pct": safe_improvement(
            baseline.max_congestion_length,
            candidate.max_congestion_length,
        ),
        "wrong_side_improvement_pct": safe_improvement(
            baseline.wrong_side_queue_peak,
            candidate.wrong_side_queue_peak,
        ),
        "idling_fuel_improvement_pct": safe_improvement(
            baseline.total_idling_fuel_liters,
            candidate.total_idling_fuel_liters,
        ),
        "clearance_improvement_pct": safe_improvement(
            baseline.clearance_time,
            candidate.clearance_time,
        ),
    }
