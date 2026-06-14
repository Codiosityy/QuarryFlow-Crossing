# QuarryFlow Crossing Pitch Brief

Generated: 2026-06-14 20:41

## Scenario

- Preset: `peak`
- Context: Balanced, high-demand traffic with visible queue buildup. This is the best primary judging scenario.
- Model mode: `sklearn-gbr`

## One-Line Story

MCTS Rollout cuts delay by 25.8% while increasing throughput by 14.1% in the peak scenario.

## Comparison

| policy | average_waiting_time | throughput | max_congestion_length | clearance_time | worst_clearance_time | conflict_count | occupancy_risk | fairness_gap | wrong_side_queue_peak | dilemma_zone_peak | total_idling_fuel_liters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Free Flow | 158.05 | 1020.00 | 459.69 | 192.75 | 255.00 | 5 | 1.00 | 0.02 | 0.72 | 0.71 | 5.50 |
| Static Alternating | 170.41 | 738.00 | 588.63 | 192.75 | 255.00 | 0 | 0.58 | 0.01 | 0.84 | 0.76 | 12.14 |
| MCTS Rollout | 117.31 | 1164.00 | 377.98 | 192.75 | 255.00 | 0 | 0.74 | 0.14 | 0.51 | 0.57 | 8.14 |

## Improvement vs Free Flow

| policy | waiting_time_improvement_pct | throughput_improvement_pct | congestion_improvement_pct | wrong_side_improvement_pct | idling_fuel_improvement_pct | clearance_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Static Alternating | -7.82 | -27.65 | -28.05 | -17.11 | -120.56 | 0.00 |
| MCTS Rollout | 25.78 | 14.12 | 17.77 | 28.57 | -47.95 | 0.00 |

## Technical Notes

- Baselines: `Free Flow`, `Static Alternating`
- Visible adaptive benchmark: `Legacy Adaptive`
- Primary hybrid controller: `Hybrid Adaptive`
- Current model backend: `sklearn-gbr`
- Innovation: safety-shielded counterfactual bandit over short-horizon simulator rollouts
