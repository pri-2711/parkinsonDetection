from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def load_repository_environment() -> None:
    """Load root .env values, including when a blank process variable masks one."""
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return

    env_path = REPOSITORY_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)
    # load_dotenv intentionally preserves existing variables. A blank Windows
    # process variable is not a usable override and must not hide the local key.
    for name, value in dotenv_values(env_path).items():
        if value is not None and not os.getenv(name, "").strip():
            os.environ[name] = value


load_repository_environment()

DEFAULT_EXPERIMENT_ARTIFACTS = (
    REPOSITORY_ROOT / "experiments" / "regularized_svm_rf" / "results" / "artifacts"
)
DEFAULT_FUSION_WEIGHTS = (
    REPOSITORY_ROOT / "experiments" / "regularized_svm_rf" / "results" / "fusion_validation_weights.json"
)
DEFAULT_SPEECH_AUDIO_V2_ARTIFACTS = REPOSITORY_ROOT / "experiments" / "speech_audio_v2" / "artifacts"


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
    SPIRAL_MAX_TOTAL_PIXELS = int(os.getenv("SPIRAL_MAX_TOTAL_PIXELS", 2_000_000_000))
    UPLOAD_TEMP_DIR = os.getenv("UPLOAD_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "parkinson-assessment-uploads"
    )
    SPIRAL_MEDIA_DIR = os.getenv("SPIRAL_MEDIA_DIR") or str(REPOSITORY_ROOT / "data" / "spiral_media")
    MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", str(DEFAULT_EXPERIMENT_ARTIFACTS))
    SPEECH_MODEL_FILE = os.getenv("SPEECH_MODEL_FILE", "speech_svm_tuned.joblib")
    SPEECH_AUDIO_V2_ARTIFACT_DIR = os.getenv("SPEECH_AUDIO_V2_ARTIFACT_DIR", str(DEFAULT_SPEECH_AUDIO_V2_ARTIFACTS))
    HANDWRITING_MODEL_FILE = os.getenv("HANDWRITING_MODEL_FILE", "handwriting_svm_tuned.joblib")
    TREMOR_MODEL_FILE = os.getenv("TREMOR_MODEL_FILE", "tremor_svm_tuned.joblib")
    FUSION_WEIGHTS_FILE = os.getenv("FUSION_WEIGHTS_FILE", str(DEFAULT_FUSION_WEIGHTS))
    DATABASE_PATH = os.getenv("DATABASE_PATH", str(REPOSITORY_ROOT / "data" / "parkinson_app.db"))
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))
    TREMOR_ALLOWED_LOCATIONS = _allowed_locations()
    TREMOR_MAX_INTENSITY = float(os.getenv("TREMOR_MAX_INTENSITY", 10.0))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
