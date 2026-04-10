import sys
sys.path.insert(0, "src")
from quarryflow.simulator import RailwayCrossingSimulator
from quarryflow.scenarios import build_scenario
from quarryflow.policy import FreeFlowPolicy, AdaptivePolicy
from collections import Counter

for sc in ["chaotic", "chaotic_aggressive", "chaotic_long_gate"]:
    cfg = build_scenario(sc, seed=11)
    sim = RailwayCrossingSimulator(cfg, seed=11)
    pol = AdaptivePolicy()
    r = sim.run_episode(pol, record_history=True)
    m = r.metrics
    actions = [a["action"] for a in r.actions_taken]
    ff = RailwayCrossingSimulator(build_scenario(sc, seed=11), seed=11).run_episode(
        FreeFlowPolicy(), record_history=False
    ).metrics
    dw = (1 - m.average_waiting_time / ff.average_waiting_time) * 100
    dt = (m.throughput / ff.throughput - 1) * 100
    dc = (1 - m.max_congestion_length / ff.max_congestion_length) * 100
    print(f"\n=== {sc} ===")
    print(f"  delay={dw:+.1f}% thru={dt:+.1f}% cong={dc:+.1f}% conf={ff.conflict_count}>{m.conflict_count}")
    print(f"  Actions: {Counter(actions)}")

    # Sample snapshots after first gate opening
    for snap in r.snapshots:
        if not snap.barrier_closed and snap.queue_counts["left"] > 0:
            lq = snap.queue_counts["left"]
            rq = snap.queue_counts["right"]
            bsp = lq > 4 and rq > 4
            print(
                f"  t={snap.time:5.1f} bsp={bsp} aggr={snap.aggressive_share_near_gate:.2f} "
                f"disorder={snap.disorder_index:.3f} occ={snap.occupancy_risk:.3f} "
                f"imb={snap.pressure_imbalance:.3f} q={lq}+{rq} action={snap.current_action}"
            )
            if snap.time > 200:
                break
