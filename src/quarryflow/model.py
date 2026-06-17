from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when sklearn is missing.
    GradientBoostingRegressor = None
    MultiOutputRegressor = None
    SKLEARN_AVAILABLE = False


TARGET_COLUMNS = [
    "average_waiting_time",
    "throughput",
    "max_congestion_length",
    "occupancy_risk_horizon",
    "fairness_gap_horizon",
    "wrong_side_queue_share_horizon",
    "idling_fuel_liters_horizon",
    "mean_clearance_time_horizon",
    "worst_clearance_time_horizon",
]


class SurrogateModel:
    def __init__(self) -> None:
        self.backend = "numpy-ridge"
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = TARGET_COLUMNS[:]
        self.is_fitted = False
        self._model = None
        self._coefficients: np.ndarray | None = None
        self._training_residual_std: np.ndarray | None = None

    def fit(self, rows: list[dict], targets: list[dict]) -> "SurrogateModel":
        features = pd.DataFrame(rows)
        target_frame = pd.DataFrame(targets)[self.target_columns]
        design = self._encode_features(features, fit=True)

        if SKLEARN_AVAILABLE:
            self.backend = "sklearn-gbr"
            base = GradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.08,
                n_estimators=180,
                random_state=42,
            )
            self._model = MultiOutputRegressor(base)
            self._model.fit(design, target_frame.to_numpy())
        else:
            self.backend = "numpy-ridge"
            self._fit_numpy_ridge(design.to_numpy(dtype=float), target_frame.to_numpy(dtype=float))

        self.is_fitted = True
        return self

    def predict_row(self, row: dict) -> dict[str, float]:
        return self.predict_rows([row])[0]



    def predict_rows(self, rows: list[dict]) -> list[dict[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("SurrogateModel must be fitted before prediction.")

        features = pd.DataFrame(rows)
        design = self._encode_features(features, fit=False)
        if self.backend == "sklearn-gbr":
            predictions = self._model.predict(design)
        else:
            predictions = self._predict_numpy_ridge(design.to_numpy(dtype=float))

        output: list[dict[str, float]] = []
        for vector in predictions:
            output.append(
                {
                    column: float(value)
                    for column, value in zip(self.target_columns, vector)
                }
            )
        return output

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "model": self._model,
            "coefficients": self._coefficients,
            "training_residual_std": self._training_residual_std,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
        import logging

        source = Path(path)
        try:
            with source.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301
        except (ModuleNotFoundError, ImportError) as exc:
            logging.warning(
                "Could not load surrogate model from %s (%s). "
                "Falling back to heuristic-only mode.",
                path,
                exc,
            )
            instance = cls()
            instance.is_fitted = False
            return instance

        instance = cls()
        instance.backend = payload["backend"]
        instance.feature_columns = payload["feature_columns"]
        instance.target_columns = payload["target_columns"]
        instance._model = payload["model"]
        instance._coefficients = payload["coefficients"]
        instance._training_residual_std = payload.get("training_residual_std")
        instance.is_fitted = True
        return instance

    def _encode_features(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        categorical_columns = [
            column
            for column in frame.columns
            if str(frame[column].dtype) in {"object", "category"}
        ]
        encoded = pd.get_dummies(frame, columns=categorical_columns, dtype=float)
        if fit:
            self.feature_columns = list(encoded.columns)
            return encoded.fillna(0.0)

        for column in self.feature_columns:
            if column not in encoded.columns:
                encoded[column] = 0.0
        encoded = encoded[self.feature_columns]
        return encoded.fillna(0.0)

    def _fit_numpy_ridge(self, features: np.ndarray, targets: np.ndarray) -> None:
        intercept = np.ones((features.shape[0], 1))
        design = np.hstack([intercept, features])
        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        self._coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets

    def _predict_numpy_ridge(self, features: np.ndarray) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("Numpy ridge model is not fitted.")
        intercept = np.ones((features.shape[0], 1))
        design = np.hstack([intercept, features])
        return design @ self._coefficients



