from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.dashboard_data import comparison_frame, improvement_frame, run_policy_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="peak")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "reports"))
    parser.add_argument("--model", default=None)

    args = parser.parse_args()

    results = run_policy_suite(
        args.scenario,
        seed=args.seed,
        model_path=args.model,

        record_history=False,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comparison = comparison_frame(results)
    improvement = improvement_frame(results)
    comparison.to_csv(outdir / f"{args.scenario}_comparison.csv", index=False)
    improvement.to_csv(outdir / f"{args.scenario}_improvement.csv", index=False)

    with (outdir / f"{args.scenario}_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison.to_dict(orient="records"), handle, indent=2)

    print(comparison.to_string(index=False))
    print()
    print(improvement.to_string(index=False))


if __name__ == "__main__":
    main()
