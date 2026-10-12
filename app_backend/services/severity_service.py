from __future__ import annotations

import joblib
from pathlib import Path


class SeverityService:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.model = None
        if model_path and model_path.exists():
            try:
                artifact = joblib.load(model_path)
                self.model = artifact.get("models", {}).get("ridge") or artifact.get("models", {}).get("random_forest")
                self.features = artifact.get("features", [])
            except Exception:
                self.model = None

    def predict_updrs(self, features) -> dict | None:
        if self.model is None:
            return None
        try:
            prediction = self.model.predict(features)
            return {"updrs_estimate": float(prediction[0]) if hasattr(prediction, "__iter__") else float(prediction)}
        except Exception:
            return None
