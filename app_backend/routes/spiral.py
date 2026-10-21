from __future__ import annotations

import logging

from flask import Blueprint, current_app, request
from werkzeug.datastructures import FileStorage

from app_backend.errors import APIError
from app_backend.services.spiral_preprocessing import validate_spiral_upload

spiral_bp = Blueprint("spiral", __name__)
logger = logging.getLogger("app_backend.spiral")


@spiral_bp.get("/api/spiral/status")
def spiral_status():
    adapter = current_app.extensions["model_registry"].models["handwriting"]
    return {
        "model_loaded": adapter.ready,
        "prediction_available": False,
        "tablet_trajectory_prediction_available": adapter.ready,
        "blocker": "camera_to_tablet_feature_mapping_unvalidated",
        "tablet_input_contract": "x/y/z, measured pressure, measured grip angle, and timestamp",
        "required_training_signals": [
            "x/y trajectory", "pen-up indicator", "pressure", "grip angle", "timestamp"
        ],
    }


@spiral_bp.post("/api/spiral/predict")
def spiral_predict():
    sessions = current_app.extensions["assessment_sessions"]
    session_id = request.form.get("session_id")
    session_id = session_id.strip() if session_id and session_id.strip() else None
    if session_id:
        sessions.assert_current_step(session_id, "spiral")
        sessions.begin_step(session_id, "spiral")
    try:
        adapter = current_app.extensions["model_registry"].models["handwriting"]
        if not adapter.ready:
            raise APIError(503, "handwriting_model_unavailable", "The saved handwriting model is not ready.")
        file = request.files.get("file")
        if not isinstance(file, FileStorage) or not file.filename:
            raise APIError(400, "missing_spiral", "Provide an image or video in the multipart field 'file'.")
        data = file.read(current_app.config["SPIRAL_MAX_BYTES"] + 1)
        metadata = validate_spiral_upload(data, file.filename, file.content_type, current_app.config)
        raise APIError(
            501,
            "camera_to_tablet_feature_mapping_unvalidated",
            "The upload was validated, but prediction is disabled: camera input cannot currently reproduce the tablet pressure, grip-angle, pen-up, and trajectory features required by the saved handwriting model.",
        )
    except APIError as error:
        if session_id:
            sessions.record_failure(session_id, "spiral", error.code, error.message)
        raise
    except Exception:
        logger.exception("spiral_processing_failed", extra={"event": "spiral_processing_error"})
        if session_id:
            sessions.record_failure(session_id, "spiral", "spiral_processing_failed", "Spiral processing failed.")
        raise APIError(500, "spiral_processing_failed", "Spiral processing failed.") from None
