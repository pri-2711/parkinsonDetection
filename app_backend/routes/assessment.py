from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify

from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.repositories.assessment_repository import AssessmentRepository
from app_backend.services.auth import login_required
from app_backend.services.fusion_service import FusionService
from app_backend.services.sessions import AssessmentSessionStore

assessment_bp = Blueprint("assessment", __name__)


def _get_db() -> Database:
    if "db" not in current_app.extensions:
        db = Database()
        current_app.extensions["db"] = db
    return current_app.extensions["db"]


def _get_assessment_repo() -> AssessmentRepository:
    if "assessment_repo" not in current_app.extensions:
        current_app.extensions["assessment_repo"] = AssessmentRepository(_get_db())
    return current_app.extensions["assessment_repo"]


# ── In-memory session endpoints (legacy, for backward compatibility) ──

@assessment_bp.post("/api/assessment/sessions")
def create_session():
    session = current_app.extensions["assessment_sessions"].create()
    return session, 201


@assessment_bp.get("/api/assessment/sessions/<session_id>")
def get_session(session_id: str):
    return current_app.extensions["assessment_sessions"].get(session_id)


@assessment_bp.post("/api/assessment/sessions/<session_id>/complete")
def complete_session(session_id: str):
    session = current_app.extensions["assessment_sessions"].complete(session_id)
    return {"session_id": session_id, "status": session["status"], "completed_at": session["completed_at"]}


@assessment_bp.get("/api/assessment/sessions/<session_id>/results")
def get_results(session_id: str):
    return current_app.extensions["assessment_sessions"].results(session_id)


# ── Database-backed assessment endpoints (authenticated) ──

@assessment_bp.post("/api/assessments")
@login_required
def create_assessment():
    repo = _get_assessment_repo()
    assessment = repo.create_assessment(g.user_id)
    return jsonify({
        "assessment_id": assessment["id"],
        "user_id": assessment["user_id"],
        "status": assessment["status"],
        "created_at": assessment["created_at"],
        "steps": assessment["steps"],
    }), 201


@assessment_bp.get("/api/assessments/<int:assessment_id>")
@login_required
def get_assessment(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    return jsonify(assessment), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/complete")
@login_required
def complete_assessment(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    steps = assessment.get("steps", {})
    incomplete = [k for k, v in steps.items() if v.get("status") != "completed"]
    if incomplete:
        raise APIError(409, "assessment_incomplete", f"Cannot complete assessment with incomplete steps: {', '.join(incomplete)}")

    # Extract probabilities from completed steps
    speech_prob = None
    handwriting_prob = None
    tremor_prob = None

    speech_step = steps.get("speech", {})
    if speech_step.get("status") == "completed" and speech_step.get("outcome"):
        speech_prob = speech_step["outcome"].get("probability")

    handwriting_step = steps.get("spiral", {})
    if handwriting_step.get("status") == "completed" and handwriting_step.get("outcome"):
        handwriting_prob = handwriting_step["outcome"].get("probability")

    tremor_step = steps.get("tremor", {})
    if tremor_step.get("status") == "completed" and tremor_step.get("outcome"):
        tremor_prob = tremor_step["outcome"].get("probability")

    # Attempt fusion if at least one probability is available
    fusion_result = None
    if speech_prob is not None or handwriting_prob is not None or tremor_prob is not None:
        try:
            fusion_service = FusionService(weights=None)  # Use default weights
            fusion_result = fusion_service.fuse_predictions(
                speech_probability=speech_prob,
                handwriting_probability=handwriting_prob,
                tremor_probability=tremor_prob,
                threshold=0.5,
            )
        except Exception:
            fusion_result = None

    results = {
        "assessment_id": assessment_id,
        "completed_at": assessment.get("updated_at"),
        "modalities": steps,
        "fusion": fusion_result,
    }
    completed = repo.complete_assessment(assessment_id, g.user_id, results)
    return jsonify({
        "assessment_id": assessment_id,
        "status": completed["status"],
        "completed_at": completed["completed_at"],
    }), 200


@assessment_bp.get("/api/assessments/<int:assessment_id>/results")
@login_required
def get_assessment_results(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] != "completed":
        raise APIError(409, "assessment_incomplete", "Assessment is not completed yet.")
    results = assessment.get("results")
    if not results:
        raise APIError(404, "results_not_available", "Results are not available for this assessment.")
    return jsonify(results), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/speech")
@login_required
def update_speech_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    payload = g.get("json_payload", {})
    outcome = payload.get("outcome")
    error = payload.get("error")

    step_data = {
        "status": "completed" if outcome else "failed",
        "outcome": outcome,
        "error": error,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "speech", step_data)
    return jsonify(updated), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/spiral")
@login_required
def update_spiral_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    payload = g.get("json_payload", {})
    outcome = payload.get("outcome")
    error = payload.get("error")

    step_data = {
        "status": "completed" if outcome else "failed",
        "outcome": outcome,
        "error": error,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "spiral", step_data)
    return jsonify(updated), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/tremor")
@login_required
def update_tremor_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    payload = g.get("json_payload", {})
    outcome = payload.get("outcome")
    error = payload.get("error")

    step_data = {
        "status": "completed" if outcome else "failed",
        "outcome": outcome,
        "error": error,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "tremor", step_data)
    return jsonify(updated), 200
