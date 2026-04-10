"""Trace rollout decisions for chaotic scenario."""
import sys
sys.path.insert(0, "src")
from quarryflow.simulator import RailwayCrossingSimulator as Sim
from quarryflow.scenarios import build_scenario as bs
from quarryflow.policy import AdaptivePolicy as AP

cfg = bs("chaotic", seed=11)
sim = Sim(cfg, seed=11)
pol = AP()
r = sim.run_episode(pol, record_history=False)

print(f"Total decisions: {len(r.decision_traces)}")
for t in r.decision_traces[:5]:
    print(f"\nt={t.time} chosen={t.chosen_action}")
    for a in t.action_scores:
        print(f"  {a['action']:20s}  score={a['score']:8.3f}  base={a['base_utility']:8.3f}")
