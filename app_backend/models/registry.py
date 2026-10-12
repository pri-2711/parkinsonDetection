from __future__ import annotations

import logging
from pathlib import Path

from app_backend.models.adapter import ModelAdapter


class ModelRegistry:
    def __init__(self, artifact_dir: str | Path, model_files: dict[str, str]):
        self.artifact_dir = Path(artifact_dir)
        self.models = {
            name: ModelAdapter(self.artifact_dir / filename, name)
            for name, filename in model_files.items()
        }
        logger = logging.getLogger("app_backend.models")
        for name, adapter in self.models.items():
            logger.info(
                "model_ready" if adapter.ready else "model_unavailable",
                extra={
                    "event": "model_load",
                    "component": name,
                },
            )

    @classmethod
    def from_config(cls, config):
        return cls(
            config["MODEL_ARTIFACT_DIR"],
            {
                "speech": config["SPEECH_MODEL_FILE"],
                "handwriting": config["HANDWRITING_MODEL_FILE"],
            },
        )

    def readiness(self) -> dict:
        result = {}
        for name, adapter in self.models.items():
            result[name] = {
                "model_loaded": adapter.ready,
                "probability_output": adapter.ready,
                "model_family": adapter.family,
                "expected_feature_count": len(adapter.features),
                "error_code": adapter.error_code,
            }
        result["speech"]["prediction_available"] = False
        result["speech"]["blocker"] = "raw_audio_feature_extractor_unavailable"
        result["speech"]["gender_metadata_required_by_model"] = "gender is an input feature in the trained schema"
        result["handwriting"]["prediction_available"] = False
        result["handwriting"]["blocker"] = "camera_to_tablet_feature_mapping_unvalidated"
        return result
