from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.model import TARGET_COLUMNS, SurrogateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "artifacts" / "data" / "training_data.csv"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "models" / "surrogate.pkl"))
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    rows = frame.drop(columns=TARGET_COLUMNS).to_dict(orient="records")
    targets = frame[TARGET_COLUMNS].to_dict(orient="records")

    model = SurrogateModel().fit(rows, targets)
    output = Path(args.out)
    model.save(output)
    print(f"Saved {model.backend} model to {output}")


if __name__ == "__main__":
    main()
