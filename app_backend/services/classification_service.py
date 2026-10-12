from __future__ import annotations

from app_backend.models.registry import ModelRegistry


class ClassificationService:
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry

    def predict_speech(self, features):
        adapter = self.model_registry.models.get("speech")
        if not adapter or not adapter.ready:
            return None
        return adapter.predict(features)

    def predict_handwriting(self, features):
        adapter = self.model_registry.models.get("handwriting")
        if not adapter or not adapter.ready:
            return None
        return adapter.predict(features)

    def predict_tremor(self, features):
        adapter = self.model_registry.models.get("tremor")
        if not adapter or not adapter.ready:
            return None
        return adapter.predict(features)
