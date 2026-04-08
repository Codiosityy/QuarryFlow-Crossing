from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.dashboard_data import run_policy_suite
from quarryflow.model import BootstrapSurrogateEnsemble, SurrogateModel
from quarryflow.reporting import write_pitch_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="peak")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--model",
        default=str(ROOT / "artifacts" / "models" / "surrogate.pkl"),
    )
    parser.add_argument(
        "--ensemble",
        default=str(ROOT / "artifacts" / "models" / "surrogate_ensemble.pkl"),
    )
    parser.add_argument(
        "--controller",
        default=str(ROOT / "artifacts" / "models" / "hybrid_controller.json"),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional markdown output path. Defaults to artifacts/reports/<scenario>_pitch_brief.md",
    )
    args = parser.parse_args()

    model_path = args.model if Path(args.model).exists() else None
    ensemble_path = args.ensemble if Path(args.ensemble).exists() else None
    controller_path = args.controller if Path(args.controller).exists() else None
    results = run_policy_suite(
        args.scenario,
        seed=args.seed,
        model_path=model_path,
        ensemble_path=ensemble_path,
        controller_path=controller_path,
        record_history=False,
    )
    model_label = "heuristic-only adaptive control"
    if ensemble_path:
        model_label = BootstrapSurrogateEnsemble.load(ensemble_path).backend
    elif model_path:
        model_label = SurrogateModel.load(model_path).backend

    output_path = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "reports" / f"{args.scenario}_pitch_brief.md"
    )
    target = write_pitch_report(
        results,
        args.scenario,
        output_path,
        model_label=model_label,
    )
    print(f"Wrote pitch brief to {target}")


if __name__ == "__main__":
    main()
