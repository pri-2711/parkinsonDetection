from __future__ import annotations

import json
import math
from pathlib import Path

from src.fusion import fuse_probabilities


class FusionService:
    def __init__(self, weights: dict | None = None):
        self.weights = weights

    @classmethod
    def from_config(cls, config):
        path = Path(config["FUSION_WEIGHTS_FILE"])
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            weights = stored["normalized_weights"]
            if not isinstance(weights, dict) or set(weights) != {"speech", "handwriting", "tremor"}:
                raise ValueError
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights.values()):
                raise ValueError
            if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("validated classification fusion weights are unavailable") from None
        return cls(weights=weights)

    def fuse_predictions(
        self,
        speech_probability: float | None = None,
        handwriting_probability: float | None = None,
        tremor_probability: float | None = None,
        threshold: float = 0.5,
    ) -> dict:
        if speech_probability is None and handwriting_probability is None and tremor_probability is None:
            raise ValueError("At least one modality probability must be provided")

        result = fuse_probabilities(
            speech_probability=speech_probability,
            handwriting_probability=handwriting_probability,
            tremor_probability=tremor_probability,
            weights=self.weights,
            threshold=threshold,
        )
        return result
