from __future__ import annotations

import logging

from flask import Blueprint, current_app, request
from werkzeug.datastructures import FileStorage

from app_backend.errors import APIError
from app_backend.services.acoustic_features import AcousticFeatureExtractor
from app_backend.services.audio_preprocessing import validate_wav

speech_bp = Blueprint("speech", __name__)
logger = logging.getLogger("app_backend.speech")


def _session_id() -> str | None:
    value = request.form.get("session_id")
    return value.strip() if value and value.strip() else None


@speech_bp.post("/api/speech/predict")
def speech_predict():
    sessions = current_app.extensions["assessment_sessions"]
    session_id = _session_id()
    if session_id:
        sessions.assert_current_step(session_id, "speech")
        sessions.begin_step(session_id, "speech")
    try:
        adapter = current_app.extensions["model_registry"].models["speech"]
        if not adapter.ready:
            raise APIError(503, "speech_model_unavailable", "The saved speech model is not ready.")
        file = request.files.get("file")
        if not isinstance(file, FileStorage) or not file.filename:
            raise APIError(400, "missing_audio", "Provide an audio file in the multipart field 'file'.")
        data = file.read(current_app.config["AUDIO_MAX_BYTES"] + 1)
        metadata = validate_wav(data, file.filename, file.content_type, current_app.config)
        extractor = AcousticFeatureExtractor(adapter)
        # Extraction deliberately raises a stable 501 until training-time extraction is recovered.
        features = extractor.extract(data, metadata)
        result = adapter.predict(features)
        if session_id:
            sessions.record_success(session_id, "speech", result)
        return {"status": "completed", **result}
    except APIError as error:
        if session_id:
            sessions.record_failure(session_id, "speech", error.code, error.message)
        raise
    except Exception:
        logger.exception("speech_processing_failed", extra={"event": "speech_processing_error"})
        if session_id:
            sessions.record_failure(session_id, "speech", "speech_processing_failed", "Speech processing failed.")
        raise APIError(500, "speech_processing_failed", "Speech processing failed.") from None
