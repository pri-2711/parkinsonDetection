from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify

from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.services.auth import login_required

history_bp = Blueprint("history", __name__)


def _get_db() -> Database:
    if "db" not in current_app.extensions:
        db = Database(current_app.config["DATABASE_PATH"])
        current_app.extensions["db"] = db
    return current_app.extensions["db"]


@history_bp.get("/api/history/assessments")
@login_required
def list_assessments():
    db = _get_db()
    limit = 50
    assessments = db.list_assessments(g.user_id, limit)
    return jsonify({"assessments": assessments}), 200


@history_bp.get("/api/history/completed")
@login_required
def list_completed_assessments():
    db = _get_db()
    limit = 100
    assessments = db.get_completed_assessments(g.user_id, limit)
    return jsonify({"assessments": assessments}), 200


@history_bp.get("/api/history/assessments/<int:assessment_id>")
@login_required
def get_assessment(assessment_id: int):
    db = _get_db()
    assessment = db.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    return jsonify(assessment), 200
