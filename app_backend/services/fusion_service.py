from __future__ import annotations

from src.fusion import fuse_probabilities


class FusionService:
    def __init__(self, weights: dict | None = None):
        self.weights = weights

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
