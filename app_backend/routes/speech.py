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
        file = request.files.get("file")
        if not isinstance(file, FileStorage) or not file.filename:
            raise APIError(400, "missing_audio", "Provide an audio file in the multipart field 'file'.")
        data = file.read(current_app.config["AUDIO_MAX_BYTES"] + 1)
        metadata = validate_wav(data, file.filename, file.content_type, current_app.config)
        mode = request.form.get("mode", "")
        try:
            gender_code = int(request.form.get("gender_code", ""))
        except ValueError:
            raise APIError(422, "invalid_gender_code", "Provide gender_code as 0 or 1.") from None
        extractor = AcousticFeatureExtractor()
        features = extractor.extract(data, metadata, mode=mode, gender_code=gender_code)
        adapter = current_app.extensions["speech_audio_v2_models"][mode]
        if not adapter.ready:
            raise APIError(503, "speech_audio_v2_model_not_trained", "Audio features were extracted with speech_audio_v2, but no model has been trained for this recording mode. The legacy SVM was not used because its feature definitions are not compatible.")
        result = adapter.predict(features)
        result.update({"feature_schema": extractor.schema, "recording_mode": mode, "model_verified": True})
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
