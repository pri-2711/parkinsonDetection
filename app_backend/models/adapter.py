from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelAdapter:
    """Validated wrapper around a saved probability-capable classifier."""

    def __init__(self, artifact_path: str | Path, modality: str):
        self.artifact_path = Path(artifact_path)
        self.modality = modality
        self.bundle: dict[str, Any] | None = None
        self.error_code: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            bundle = joblib.load(self.artifact_path)
            if not isinstance(bundle, dict) or bundle.get("modality") != self.modality:
                raise ValueError("unexpected saved artifact bundle")
            features = bundle.get("features")
            model = bundle.get("model")
            if not isinstance(features, list) or not features or len(features) != len(set(features)):
                raise ValueError("invalid saved feature schema")
            if model is None or not callable(getattr(model, "predict_proba", None)):
                raise ValueError("saved classifier does not provide probabilities")
            classes = np.asarray(getattr(model, "classes_", []))
            if classes.tolist() != [0, 1]:
                raise ValueError("saved classifier has an unexpected class order")
            self.bundle = bundle
        except FileNotFoundError:
            self.error_code = "artifact_missing"
        except Exception:
            # Avoid returning artifact locations or deserialization internals to clients.
            self.error_code = "artifact_unavailable"

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    @property
    def features(self) -> list[str]:
        return list(self.bundle["features"]) if self.bundle else []

    @property
    def family(self) -> str | None:
        return self.bundle.get("family") if self.bundle else None

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("saved model is unavailable")
        if list(features.columns) != self.features:
            raise ValueError("input columns do not match the saved model schema")
        if features.empty:
            raise ValueError("model input must contain at least one row")
        try:
            numeric = features.to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError("model inputs must be numeric") from None
        if np.isinf(numeric).any():
            raise ValueError("model inputs must not contain infinite values")
        probabilities = np.asarray(self.bundle["model"].predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape != (len(features), 2):
            raise ValueError("saved model returned an invalid probability shape")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("saved model returned invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("saved model returned probabilities that are not normalized")
        probability = float(probabilities[0, 1])
        prediction = int(probability >= 0.5)
        return {"prediction": prediction, "probability": probability}
