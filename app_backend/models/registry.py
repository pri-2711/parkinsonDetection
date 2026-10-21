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
                "tremor": config["TREMOR_MODEL_FILE"],
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
        result["speech"]["legacy_baseline_prediction_available"] = False
        result["speech"]["prediction_available"] = False
        result["speech"]["blocker"] = "speech_audio_v2_mode_model_required"
        result["speech"]["input_schema"] = "speech_audio_v2 (separate from legacy 45-feature SVM)"
        result["speech"]["gender_metadata_required"] = True
        result["handwriting"]["prediction_available"] = result["handwriting"]["model_loaded"]
        result["handwriting"]["input_contract"] = "tablet x/y/z, measured pressure, measured grip angle, and timestamp"
        result["handwriting"]["camera_input_prediction_available"] = False
        result["handwriting"]["camera_blocker"] = "camera_to_tablet_feature_mapping_unvalidated"
        result["tremor"]["prediction_available"] = result["tremor"]["model_loaded"]
        result["tremor"]["input_contract"] = "seven UPDRS III items 20/21 ratings, each 0-4"
        return result
