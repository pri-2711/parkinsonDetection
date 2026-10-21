from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app_backend.services.acoustic_features import SPEECH_V2_FEATURES


class SpeechV2Model:
    """Optional, mode-specific model trained only on the v2 audio schema."""

    def __init__(self, artifact_dir: str | Path, mode: str):
        self.mode = mode
        self.path = Path(artifact_dir) / f"{mode}.joblib"
        self.bundle: dict[str, Any] | None = None
        self.error_code: str | None = None
        try:
            bundle = joblib.load(self.path)
            if (bundle.get("schema") != "speech_audio_v2" or bundle.get("mode") != mode
                    or bundle.get("features") != SPEECH_V2_FEATURES):
                raise ValueError("v2 model contract mismatch")
            model = bundle.get("model")
            if not callable(getattr(model, "predict_proba", None)) or np.asarray(model.classes_).tolist() != [0, 1]:
                raise ValueError("v2 model must provide probabilities with class order [0, 1]")
            self.bundle = bundle
        except FileNotFoundError:
            self.error_code = "speech_audio_v2_model_not_trained"
        except Exception:
            self.error_code = "speech_audio_v2_model_invalid"

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def predict(self, features: pd.DataFrame) -> dict[str, float | int]:
        if not self.ready:
            raise RuntimeError("speech audio v2 model is not ready")
        if list(features.columns) != SPEECH_V2_FEATURES or features.shape != (1, len(SPEECH_V2_FEATURES)):
            raise ValueError("speech v2 feature order or shape mismatch")
        proba = np.asarray(self.bundle["model"].predict_proba(features), dtype=float)
        if proba.shape != (1, 2) or not np.isfinite(proba).all() or not np.isclose(proba.sum(), 1.0):
            raise ValueError("speech v2 model returned an invalid probability")
        probability = float(proba[0, 1])
        return {"prediction": int(probability >= 0.5), "probability": probability}
