from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from app_backend.errors import APIError
from app_backend.services.auth import login_required
from app_backend.services.research_insights import build_research_insights

research_bp = Blueprint("research", __name__)


@research_bp.get("/api/research/insights")
@login_required
def research_insights():
    try:
        payload = build_research_insights(
            current_app.config["EXPERIMENT_RESULTS_DIR"],
            current_app.config["MODEL_ARTIFACT_DIR"],
            current_app.config["SPEECH_AUDIO_V2_ARTIFACT_DIR"],
        )
    except (OSError, ValueError, KeyError):
        raise APIError(503, "research_outputs_unavailable", "Saved experiment outputs could not be read.") from None
    return jsonify(payload), 200
