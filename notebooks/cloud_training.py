# QuarryFlow Crossing — Cloud Training Notebook
#
# This notebook can be run on Google Colab, Kaggle, or any Jupyter environment
# to train the ML models MUCH faster than on a local machine.
#
# Usage:
#   1. Upload this entire repo to Google Drive or clone from GitHub
#   2. Open this notebook in Colab
#   3. Run all cells — models are saved to artifacts/models/
#   4. Download the artifacts folder back to your local machine
#
# Why cloud is faster:
#   - Colab provides free GPU/TPU (not needed for this, but CPU is still faster)
#   - Colab's CPUs are typically faster than laptop CPUs for batch computation
#   - No background processes competing for resources
#   - Can run full profile (more seeds, more passes) without waiting

# %% [markdown]
# # 🚦 QuarryFlow Crossing — Model Training
#
# Run this notebook to train all ML models for the dashboard.
# Estimated time: ~5-8 minutes on Colab (vs 15-20 minutes locally).

# %%
# === Step 1: Setup ===
import subprocess
import sys
import os

# If running on Colab, clone the repo
if 'google.colab' in str(get_ipython()):
    if not os.path.exists('QuarryFlow-Crossing'):
        subprocess.run(['git', 'clone', 'https://github.com/Codiosityy/QuarryFlow-Crossing.git'], check=True)
    os.chdir('QuarryFlow-Crossing')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    print("✅ Setup complete on Colab")
elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    # Kaggle
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    print("✅ Setup complete on Kaggle")
else:
    print("✅ Running locally")

# %%
# === Step 2: Verify imports ===
sys.path.insert(0, 'src')

from quarryflow.simulator import RailwayCrossingSimulator
from quarryflow.scenarios import build_scenario, list_scenarios
from quarryflow.policy import FreeFlowPolicy, AdaptivePolicy
from quarryflow.model import SurrogateModel, BootstrapSurrogateEnsemble
from quarryflow.evaluation import collect_counterfactual_rows, split_feature_targets

print(f"✅ All imports successful")
print(f"📋 Available scenarios: {list_scenarios()}")

# %%
# === Step 3: Generate training data ===
import time

start = time.time()
all_rows = []
scenarios = ["light", "peak", "chaotic", "peak_left_skew", "peak_right_skew", "chaotic_aggressive", "chaotic_long_gate"]
seeds = [7, 11, 13, 17, 19, 23]  # Full seed set

for scenario in scenarios:
    for seed in seeds:
        rows = collect_counterfactual_rows(scenario, seed)
        all_rows.extend(rows)
        print(f"  ✓ {scenario}/seed={seed}: {len(rows)} rows")

elapsed = time.time() - start
print(f"\n✅ Generated {len(all_rows)} total training rows in {elapsed:.1f}s")

# %%
# === Step 4: Train surrogate ensemble (full quality) ===
from quarryflow.model import TARGET_COLUMNS

features, targets = split_feature_targets(all_rows)

# Full quality: 3 bootstrap models
start = time.time()
ensemble = BootstrapSurrogateEnsemble(n_models=3, random_seed=42).fit(features, targets)
elapsed = time.time() - start
print(f"✅ Ensemble trained ({ensemble.backend}) in {elapsed:.1f}s")

# Also train legacy single model
legacy = SurrogateModel().fit(features, targets)
print(f"✅ Legacy model trained ({legacy.backend})")

# %%
# === Step 5: Train hybrid controller ===
from quarryflow.domain_types import AdaptivePolicyConfig
from quarryflow.hybrid import LinUCBResidual, save_hybrid_controller
from quarryflow.policy import HybridAdaptivePolicy, StaticAlternatingPolicy
from quarryflow.evaluation import evaluate_policy_suite, CURRICULUM_STAGES
import copy

config = AdaptivePolicyConfig()
bandit = LinUCBResidual(alpha=config.linucb_alpha)
best_bandit = copy.deepcopy(bandit.to_dict())
best_reward = float("-inf")

train_seeds = [7, 11, 13, 17, 19, 23]
val_seeds = [29, 31]
stage_passes = 2  # Full quality

print("🎰 Training LinUCB bandit...")
for stage_name, stage_scenarios in CURRICULUM_STAGES:
    for pass_idx in range(stage_passes):
        for scenario in stage_scenarios:
            for seed in train_seeds:
                sim = RailwayCrossingSimulator(build_scenario(scenario, seed=seed), seed=seed)
                policy = HybridAdaptivePolicy(model=ensemble, bandit=bandit, config=config, training_mode=True)
                sim.run_episode(policy, record_history=False)

        # Validate
        policies = {
            "Free Flow": FreeFlowPolicy(),
            "Static Alternating": StaticAlternatingPolicy(),
            "Legacy Adaptive": AdaptivePolicy(model=ensemble, config=config),
            "Hybrid Adaptive": HybridAdaptivePolicy(model=ensemble, bandit=bandit, config=config, training_mode=False),
        }
        val_frame = evaluate_policy_suite(
            scenario_names=[s for _, slist in CURRICULUM_STAGES for s in slist],
            seeds=val_seeds, policies=policies, config=config,
        )
        hybrid_reward = float(val_frame[val_frame["policy"] == "Hybrid Adaptive"]["episode_reward"].mean())
        if hybrid_reward > best_reward:
            best_reward = hybrid_reward
            best_bandit = copy.deepcopy(bandit.to_dict())

        print(f"  ✓ {stage_name} pass {pass_idx+1}: hybrid_reward={hybrid_reward:.1f} (best={best_reward:.1f})")

bandit = LinUCBResidual.from_dict(best_bandit)
print(f"✅ Bandit training complete. Best reward: {best_reward:.1f}")

# %%
# === Step 6: Save all artifacts ===
from pathlib import Path
import json

model_dir = Path("artifacts/models")
model_dir.mkdir(parents=True, exist_ok=True)

ensemble.save(model_dir / "surrogate_ensemble.pkl")
legacy.save(model_dir / "surrogate.pkl")

metadata = {
    "profile": "full",
    "train_seeds": train_seeds,
    "validation_seeds": val_seeds,
    "stage_passes": stage_passes,
    "n_models": 3,
    "best_validation_reward": best_reward,
    "hybrid_default_ok": True,
}
save_hybrid_controller(model_dir / "hybrid_controller.json", config=config, bandit=bandit, metadata=metadata)

print(f"✅ Saved models to {model_dir}")
for f in model_dir.iterdir():
    print(f"  📦 {f.name} ({f.stat().st_size / 1024:.0f} KB)")

# %%
# === Step 7: Run sensitivity analysis ===
from copy import deepcopy
import pandas as pd

analysis_dir = Path("artifacts/analysis")
analysis_dir.mkdir(parents=True, exist_ok=True)

def run_one(cfg, seed, policy, label):
    sim = RailwayCrossingSimulator(cfg, seed=seed)
    r = sim.run_episode(policy, record_history=False)
    m = r.metrics
    return {"policy": label, "average_waiting_time": round(m.average_waiting_time, 2),
            "throughput": round(m.throughput, 2), "max_congestion_length": round(m.max_congestion_length, 2),
            "disorder_peak": round(m.disorder_peak, 3), "vehicles_cleared": m.vehicles_cleared,
            "clearance_time": round(m.clearance_time, 2), "fairness_gap": round(m.fairness_gap, 3),
            "conflict_count": m.conflict_count}

# Arrival rate sweep
print("📊 Running arrival rate sweep...")
rows = []
for lr in [10, 14, 18, 22, 26]:
    for rr in [10, 14, 18, 22, 26]:
        cfg = build_scenario("peak", seed=11)
        cfg.arrival_rate_per_minute = {"left": float(lr), "right": float(rr)}
        for label, pol in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
            row = run_one(cfg, 11, pol, label)
            row["arrival_left"] = lr
            row["arrival_right"] = rr
            rows.append(row)
pd.DataFrame(rows).to_csv(analysis_dir / "arrival_rate_sweep.csv", index=False)
print(f"  ✓ {len(rows)} runs saved")

# Aggression sweep
print("🧠 Running aggression sweep...")
rows = []
mixes = [
    ("Calm", {"cautious": .30, "compliant": .35, "opportunistic": .20, "assertive": .10, "aggressive": .04, "reckless": .01}),
    ("Normal", {"cautious": .08, "compliant": .22, "opportunistic": .34, "assertive": .16, "aggressive": .14, "reckless": .06}),
    ("Aggressive", {"cautious": .03, "compliant": .08, "opportunistic": .20, "assertive": .24, "aggressive": .30, "reckless": .15}),
    ("Reckless", {"cautious": .01, "compliant": .04, "opportunistic": .10, "assertive": .20, "aggressive": .35, "reckless": .30}),
]
for ml, mix in mixes:
    cfg = build_scenario("peak", seed=11)
    cfg.driver_mix = deepcopy(mix)
    for label, pol in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
        row = run_one(cfg, 11, pol, label)
        row["driver_mix"] = ml
        rows.append(row)
pd.DataFrame(rows).to_csv(analysis_dir / "aggression_sweep.csv", index=False)

# Closure sweep
print("⏱️ Running closure sweep...")
rows = []
for extra in [0, 15, 30, 45, 60, 90]:
    cfg = build_scenario("peak", seed=11)
    cfg.train_closures = [(85.0, 145.0 + extra), (275.0, 345.0 + extra)]
    cfg.episode_seconds = max(cfg.episode_seconds, int(345 + extra + 200))
    for label, pol in [("Free Flow", FreeFlowPolicy()), ("Adaptive", AdaptivePolicy())]:
        row = run_one(cfg, 11, pol, label)
        row["closure_duration_s"] = 60 + extra
        rows.append(row)
pd.DataFrame(rows).to_csv(analysis_dir / "closure_duration_sweep.csv", index=False)

print("✅ All analysis data saved!")

# %%
# === Step 8: Download artifacts ===
if 'google.colab' in str(get_ipython()):
    import shutil
    shutil.make_archive('quarryflow_artifacts', 'zip', '.', 'artifacts')
    from google.colab import files
    files.download('quarryflow_artifacts.zip')
    print("📥 Download started — unzip into your local project root")
else:
    print("✅ Artifacts already saved locally in artifacts/")
    print("   Copy them to your project if running remotely.")
