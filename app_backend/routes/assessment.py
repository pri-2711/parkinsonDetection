from flask import Blueprint, current_app

assessment_bp = Blueprint("assessment", __name__)


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
