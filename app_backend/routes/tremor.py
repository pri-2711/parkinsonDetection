from __future__ import annotations

from flask import Blueprint, current_app, request

from app_backend.errors import APIError
from app_backend.services.tremor_placeholder import create_prototype_submission

tremor_bp = Blueprint("tremor", __name__)


@tremor_bp.post("/api/tremor/submit")
def tremor_submit():
    if not request.is_json:
        raise APIError(415, "json_required", "Submit tremor prototype fields as JSON.")
    payload = request.get_json(silent=True)
    result, session_id = create_prototype_submission(
        payload,
        current_app.config["TREMOR_ALLOWED_LOCATIONS"],
        current_app.config["TREMOR_MAX_INTENSITY"],
    )
    if session_id:
        sessions = current_app.extensions["assessment_sessions"]
        sessions.assert_current_step(session_id, "tremor")
        sessions.begin_step(session_id, "tremor")
        sessions.record_success(session_id, "tremor", result)
        result["session_id"] = session_id
    return result, 201
