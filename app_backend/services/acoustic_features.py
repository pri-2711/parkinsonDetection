from __future__ import annotations

from app_backend.errors import APIError


class AcousticFeatureExtractor:
    """Explicitly blocked until training-time raw-audio extraction is available."""

    def __init__(self, speech_model):
        self.expected_features = speech_model.features

    @property
    def available(self) -> bool:
        return False

    def extract(self, _audio_bytes: bytes, _audio_metadata: dict):
        raise APIError(
            501,
            "raw_audio_feature_extractor_unavailable",
            "Prediction is disabled: the repository contains precomputed acoustic features but no verified raw-audio feature extractor matching the saved model schema.",
        )
