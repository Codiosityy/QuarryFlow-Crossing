from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.evaluation import collect_counterfactual_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "data" / "training_data.csv"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 13, 17])
    parser.add_argument("--scenarios", nargs="*", default=["light", "peak", "chaotic"])
    args = parser.parse_args()

    rows: list[dict] = []
    for scenario_name in args.scenarios:
        for seed in args.seeds:
            rows.extend(collect_counterfactual_rows(scenario_name, seed))

    frame = pd.DataFrame(rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
