import sys
sys.path.insert(0, "src")
from quarryflow.simulator import RailwayCrossingSimulator as Sim
from quarryflow.scenarios import build_scenario as bs
from quarryflow.policy import FreeFlowPolicy as FFP, AdaptivePolicy as AP

print(f"{'Scenario':22s}  {'delay':>7s}  {'thru':>7s}  {'cong':>7s}  {'conf':>5s}")
print("-" * 60)
for sc in ["peak", "chaotic", "chaotic_aggressive", "chaotic_long_gate", "peak_left_skew", "peak_right_skew", "light"]:
    ff = Sim(bs(sc, seed=11), seed=11).run_episode(FFP(), record_history=False).metrics
    ad = Sim(bs(sc, seed=11), seed=11).run_episode(AP(), record_history=False).metrics
    dw = (1 - ad.average_waiting_time / ff.average_waiting_time) * 100
    dt = (ad.throughput / ff.throughput - 1) * 100
    dc = (1 - ad.max_congestion_length / ff.max_congestion_length) * 100
    print(f"{sc:22s}  {dw:+6.1f}%  {dt:+6.1f}%  {dc:+6.1f}%  {ff.conflict_count}>{ad.conflict_count}")
