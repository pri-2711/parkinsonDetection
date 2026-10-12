from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify

from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.services.auth import login_required

results_bp = Blueprint("results", __name__)


def _get_db() -> Database:
    if "db" not in current_app.extensions:
        db = Database()
        current_app.extensions["db"] = db
    return current_app.extensions["db"]


@results_bp.get("/api/results/<int:assessment_id>")
@login_required
def get_assessment_results(assessment_id: int):
    db = _get_db()
    assessment = db.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] != "completed":
        raise APIError(409, "assessment_incomplete", "Assessment is not completed yet.")
    results = assessment.get("results")
    if not results:
        raise APIError(404, "results_not_available", "Results are not available for this assessment.")
    return jsonify(results), 200
