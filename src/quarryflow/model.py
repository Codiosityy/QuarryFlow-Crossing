from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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

    def predict_row_with_uncertainty(self, row: dict) -> tuple[dict[str, float], dict[str, float]]:
        prediction = self.predict_row(row)
        return prediction, {column: 0.0 for column in self.target_columns}

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
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)

        instance = cls()
        instance.backend = payload["backend"]
        instance.feature_columns = payload["feature_columns"]
        instance.target_columns = payload["target_columns"]
        instance._model = payload["model"]
        instance._coefficients = payload["coefficients"]
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


class BootstrapSurrogateEnsemble:
    def __init__(self, *, n_models: int = 3, random_seed: int = 42) -> None:
        self.n_models = n_models
        self.random_seed = random_seed
        self.models: list[SurrogateModel] = []
        self.target_columns: list[str] = TARGET_COLUMNS[:]
        self.backend = "bootstrap-ensemble"
        self.is_fitted = False

    def fit(self, rows: list[dict], targets: list[dict]) -> "BootstrapSurrogateEnsemble":
        if not rows:
            raise ValueError("BootstrapSurrogateEnsemble requires non-empty rows.")
        rng = np.random.default_rng(self.random_seed)
        self.models = []
        size = len(rows)
        for index in range(self.n_models):
            sample_ids = rng.integers(0, size, size=size)
            sample_rows = [rows[int(sample_id)] for sample_id in sample_ids]
            sample_targets = [targets[int(sample_id)] for sample_id in sample_ids]
            model = SurrogateModel().fit(sample_rows, sample_targets)
            self.models.append(model)
            if index == 0:
                self.target_columns = model.target_columns[:]
                self.backend = f"bootstrap-ensemble[{model.backend}]"
        self.is_fitted = True
        return self

    def predict_row(self, row: dict) -> dict[str, float]:
        mean, _ = self.predict_row_with_uncertainty(row)
        return mean

    def predict_rows(self, rows: list[dict]) -> list[dict[str, float]]:
        return [self.predict_row(row) for row in rows]

    def predict_row_with_uncertainty(self, row: dict) -> tuple[dict[str, float], dict[str, float]]:
        if not self.is_fitted or not self.models:
            raise RuntimeError("BootstrapSurrogateEnsemble must be fitted before prediction.")
        model_predictions = [model.predict_row(row) for model in self.models]
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for column in self.target_columns:
            values = np.array([prediction[column] for prediction in model_predictions], dtype=float)
            means[column] = float(values.mean())
            stds[column] = float(values.std(ddof=0))
        return means, stds

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted ensemble.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_models": self.n_models,
            "random_seed": self.random_seed,
            "target_columns": self.target_columns,
            "backend": self.backend,
            "models": [
                {
                    "backend": model.backend,
                    "feature_columns": model.feature_columns,
                    "target_columns": model.target_columns,
                    "model": model._model,
                    "coefficients": model._coefficients,
                }
                for model in self.models
            ],
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "BootstrapSurrogateEnsemble":
        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls(n_models=int(payload["n_models"]), random_seed=int(payload["random_seed"]))
        instance.target_columns = list(payload["target_columns"])
        instance.backend = str(payload.get("backend", "bootstrap-ensemble"))
        instance.models = []
        for model_payload in payload["models"]:
            model = SurrogateModel()
            model.backend = model_payload["backend"]
            model.feature_columns = list(model_payload["feature_columns"])
            model.target_columns = list(model_payload["target_columns"])
            model._model = model_payload["model"]
            model._coefficients = model_payload["coefficients"]
            model.is_fitted = True
            instance.models.append(model)
        instance.is_fitted = True
        return instance
