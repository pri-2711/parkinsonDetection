from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ARTIFACTS = (
    REPOSITORY_ROOT / "experiments" / "regularized_svm_rf" / "results" / "artifacts"
)


def _allowed_locations() -> tuple[str, ...]:
    raw = os.getenv(
        "TREMOR_ALLOWED_LOCATIONS",
        "left_hand,right_hand,left_leg,right_leg,jaw,other",
    )
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


class BackendConfig:
    SECRET_KEY = os.getenv("APP_SECRET_KEY")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_REQUEST_BYTES", 22 * 1024 * 1024))
    AUDIO_MAX_BYTES = int(os.getenv("AUDIO_MAX_BYTES", 16 * 1024 * 1024))
    SPIRAL_MAX_BYTES = int(os.getenv("SPIRAL_MAX_BYTES", 20 * 1024 * 1024))
    AUDIO_MIN_SECONDS = float(os.getenv("AUDIO_MIN_SECONDS", 1.0))
    AUDIO_MAX_SECONDS = float(os.getenv("AUDIO_MAX_SECONDS", 30.0))
    SPIRAL_MAX_SECONDS = float(os.getenv("SPIRAL_MAX_SECONDS", 30.0))
    SPIRAL_MAX_FRAMES = int(os.getenv("SPIRAL_MAX_FRAMES", 1200))
    SPIRAL_MAX_DIMENSION = int(os.getenv("SPIRAL_MAX_DIMENSION", 4096))
    UPLOAD_TEMP_DIR = os.getenv("UPLOAD_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "parkinson-assessment-uploads"
    )
    MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", str(DEFAULT_EXPERIMENT_ARTIFACTS))
    SPEECH_MODEL_FILE = os.getenv("SPEECH_MODEL_FILE", "speech_svm_tuned.joblib")
    HANDWRITING_MODEL_FILE = os.getenv("HANDWRITING_MODEL_FILE", "handwriting_svm_tuned.joblib")
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))
    TREMOR_ALLOWED_LOCATIONS = _allowed_locations()
    TREMOR_MAX_INTENSITY = float(os.getenv("TREMOR_MAX_INTENSITY", 10.0))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
