from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.dashboard_data import run_policy_suite
from quarryflow.model import BootstrapSurrogateEnsemble
from quarryflow.reporting import (
    build_assumptions_markdown,
    build_judge_packet_markdown,
    write_text_report,
)


def main() -> None:
    report_dir = ROOT / "artifacts" / "reports"
    eval_dir = ROOT / "artifacts" / "eval"
    model_dir = ROOT / "artifacts" / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    assumptions_markdown = build_assumptions_markdown()
    write_text_report(assumptions_markdown, report_dir / "assumptions.md")

    learning_curve = pd.read_csv(eval_dir / "learning_curve.csv") if (eval_dir / "learning_curve.csv").exists() else pd.DataFrame()
    holdout_summary = pd.read_csv(eval_dir / "holdout_summary.csv") if (eval_dir / "holdout_summary.csv").exists() else pd.DataFrame()
    ensemble_path = model_dir / "surrogate_ensemble.pkl"
    controller_path = model_dir / "hybrid_controller.json"
    results = run_policy_suite(
        "peak",
        seed=11,
        model_path=None,
        ensemble_path=str(ensemble_path) if ensemble_path.exists() else None,
        controller_path=str(controller_path) if controller_path.exists() else None,
        record_history=False,
    )
    model_label = "bootstrap-ensemble"
    if ensemble_path.exists():
        model_label = BootstrapSurrogateEnsemble.load(ensemble_path).backend
    judge_packet = build_judge_packet_markdown(
        results,
        "peak",
        model_label=model_label,
        assumptions_markdown=assumptions_markdown,
        learning_curve=learning_curve,
        holdout_summary=holdout_summary,
    )
    write_text_report(judge_packet, report_dir / "judge_packet.md")
    print(f"Wrote judge packet to {report_dir / 'judge_packet.md'}")


if __name__ == "__main__":
    main()
