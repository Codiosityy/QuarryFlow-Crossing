"""Quick 3-seed validation."""
import sys
sys.path.insert(0, "src")
from quarryflow.simulator import RailwayCrossingSimulator as Sim
from quarryflow.scenarios import build_scenario as bs
from quarryflow.policy import FreeFlowPolicy as FFP, AdaptivePolicy as AP

SEEDS = [11, 17, 23]

print(f"{'Scenario':22s}  {'delay':>7s}  {'thru':>7s}  {'cong':>7s}  confl")
print("-" * 65)
for sc in ["peak", "chaotic", "chaotic_aggressive", "chaotic_long_gate"]:
    dws, dts, dcs = [], [], []
    confs = []
    for seed in SEEDS:
        ff = Sim(bs(sc, seed=seed), seed=seed).run_episode(FFP(), record_history=False).metrics
        ad = Sim(bs(sc, seed=seed), seed=seed).run_episode(AP(), record_history=False).metrics
        dws.append((1 - ad.average_waiting_time / ff.average_waiting_time) * 100)
        dts.append((ad.throughput / ff.throughput - 1) * 100)
        dcs.append((1 - ad.max_congestion_length / ff.max_congestion_length) * 100)
        confs.append(f"{ff.conflict_count}>{ad.conflict_count}")
    print(f"{sc:22s}  {sum(dws)/len(dws):+6.1f}%  {sum(dts)/len(dts):+6.1f}%  {sum(dcs)/len(dcs):+6.1f}%  {' '.join(confs)}")
