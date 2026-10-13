from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, Response
import numpy as np
import pandas as pd

from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.repositories.assessment_repository import AssessmentRepository
from app_backend.services.auth import login_required
from app_backend.services.fusion_service import FusionService
from app_backend.services.pdf_report import generate_assessment_pdf
from app_backend.services.sessions import AssessmentSessionStore
from app_backend.services.trajectory_features import extract_tablet_trajectory_features

assessment_bp = Blueprint("assessment", __name__)


def _get_db() -> Database:
    if "db" not in current_app.extensions:
        db = Database(current_app.config["DATABASE_PATH"])
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
    # Extract probabilities from completed steps
    speech_prob = None
    handwriting_prob = None
    tremor_prob = None

    speech_step = steps.get("speech", {})
    if speech_step.get("status") == "completed" and speech_step.get("outcome") and speech_step["outcome"].get("model_verified") is True:
        speech_prob = speech_step["outcome"].get("probability")

    handwriting_step = steps.get("spiral", {})
    if handwriting_step.get("status") == "completed" and handwriting_step.get("outcome") and handwriting_step["outcome"].get("model_verified") is True:
        handwriting_prob = handwriting_step["outcome"].get("probability")

    tremor_step = steps.get("tremor", {})
    if tremor_step.get("status") == "completed" and tremor_step.get("outcome") and tremor_step["outcome"].get("model_verified") is True:
        tremor_prob = tremor_step["outcome"].get("probability")

    if speech_prob is None and handwriting_prob is None and tremor_prob is None:
        raise APIError(409, "prediction_required", "At least one verified model prediction is required to complete an assessment.")

    # Attempt fusion if at least one probability is available
    fusion_result = None
    if speech_prob is not None or handwriting_prob is not None or tremor_prob is not None:
        try:
            fusion_service = FusionService.from_config(current_app.config)
            fusion_result = fusion_service.fuse_predictions(
                speech_probability=speech_prob,
                handwriting_probability=handwriting_prob,
                tremor_probability=tremor_prob,
                threshold=0.5,
            )
            fusion_result["output_type"] = "research_model_probability_not_clinical_diagnosis"
            fusion_result["patient_level_validation"] = "not_measured_unmatched_modalities"
        except ValueError:
            raise APIError(503, "fusion_weights_unavailable", "Validated classification fusion weights are unavailable.") from None

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
    if isinstance(outcome, dict) and "probability" in outcome:
        raise APIError(422, "prediction_must_use_model_endpoint", "Client-supplied probabilities cannot be saved as model predictions.")

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
    if isinstance(outcome, dict) and "probability" in outcome:
        raise APIError(422, "prediction_must_use_model_endpoint", "Client-supplied probabilities cannot be saved as model predictions.")

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
    if isinstance(outcome, dict) and "probability" in outcome:
        raise APIError(422, "prediction_must_use_model_endpoint", "Client-supplied probabilities cannot be saved as model predictions.")

    step_data = {
        "status": "completed" if outcome else "failed",
        "outcome": outcome,
        "error": error,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "tremor", step_data)
    return jsonify(updated), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/tremor/predict")
@login_required
def predict_tremor_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    payload = g.get("json_payload")
    adapter = current_app.extensions["model_registry"].models.get("tremor")
    if not adapter or not adapter.ready:
        raise APIError(503, "tremor_model_unavailable", "The saved tremor model is not ready.")
    if not isinstance(payload, dict) or set(payload) != set(adapter.features):
        raise APIError(422, "invalid_tremor_features", "Provide exactly the seven trained UPDRS III item 20/21 ratings.")
    try:
        values = [float(payload[name]) for name in adapter.features]
    except (TypeError, ValueError, KeyError):
        raise APIError(422, "invalid_tremor_features", "UPDRS item ratings must be numeric values from 0 to 4.") from None
    if not np.isfinite(values).all() or any(value < 0 or value > 4 or not value.is_integer() for value in values):
        raise APIError(422, "invalid_tremor_features", "UPDRS item ratings must be integer values from 0 to 4.")

    try:
        outcome = adapter.predict(pd.DataFrame([values], columns=adapter.features))
    except ValueError:
        raise APIError(422, "invalid_tremor_features", "Tremor features do not match the trained model input contract.") from None
    outcome["model_family"] = adapter.family
    outcome["output_type"] = "research_model_probability_not_clinical_diagnosis"
    outcome["model_verified"] = True
    outcome["input_features"] = {name: int(payload[name]) for name in adapter.features}
    step_data = {"status": "completed", "outcome": outcome, "error": None, "updated_at": assessment.get("updated_at")}
    updated = repo.update_assessment_step(assessment_id, g.user_id, "tremor", step_data)
    return jsonify({"assessment": updated, "outcome": outcome}), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/spiral/predict")
@login_required
def predict_spiral_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    payload = g.get("json_payload")
    adapter = current_app.extensions["model_registry"].models.get("handwriting")
    if not adapter or not adapter.ready:
        raise APIError(503, "handwriting_model_unavailable", "The saved handwriting model is not ready.")

    if not isinstance(payload, dict):
        raise APIError(400, "invalid_trajectory_payload", "Provide JSON with trajectory telemetry or points array.")

    points_data = payload.get("trajectory") or payload.get("points")
    if not isinstance(points_data, list) or len(points_data) < 2:
        raise APIError(422, "invalid_trajectory", "Trajectory must contain at least 2 coordinate points.")

    # Format points into the PaHaW tablet schema expected by extract_tablet_trajectory_features
    # TABLET_COLUMNS = ("point_index", "x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id")
    try:
        rows = []
        for idx, pt in enumerate(points_data):
            rows.append({
                "point_index": int(pt.get("point_index", idx)),
                "x": float(pt["x"]),
                "y": float(pt["y"]),
                "z": float(pt.get("z", 0.0)),
                "pressure": float(pt.get("pressure", 0.5)),
                "grip_angle": float(pt.get("grip_angle", 45.0)),
                "timestamp": float(pt.get("timestamp", idx * 10)),
                "test_id": int(pt.get("test_id", 1)),
            })
        traj_df = pd.DataFrame(rows)
        extracted_features = extract_tablet_trajectory_features(traj_df, adapter.features)
        outcome = adapter.predict(extracted_features)
    except APIError:
        raise
    except Exception as exc:
        raise APIError(422, "trajectory_processing_failed", f"Failed to extract features from trajectory: {str(exc)}") from None

    outcome["model_family"] = adapter.family
    outcome["output_type"] = "research_model_probability_not_clinical_diagnosis"
    outcome["model_verified"] = True
    outcome["input_mode"] = payload.get("input_mode", "canvas_drawing")
    outcome["points_count"] = len(rows)

    step_data = {"status": "completed", "outcome": outcome, "error": None, "updated_at": assessment.get("updated_at")}
    updated = repo.update_assessment_step(assessment_id, g.user_id, "spiral", step_data)
    return jsonify({"assessment": updated, "outcome": outcome}), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/speech/predict")
@login_required
def predict_speech_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")

    adapter = current_app.extensions["model_registry"].models.get("speech")
    if not adapter or not adapter.ready:
        raise APIError(503, "speech_model_unavailable", "The saved speech model is not ready.")

    payload = g.get("json_payload", {})
    passage_version = payload.get("passage_version", "1.0")
    passage_text = payload.get("passage_text", (
        "Today is a pleasant day. I am speaking at my normal pace and volume. "
        "I will read each sentence clearly and naturally, without rushing. "
        "After finishing the passage, I will pause briefly before ending the recording."
    ))

    # Record the validated reading passage and transparent 501 blocker status
    error_message = (
        "Prediction is disabled: the repository contains precomputed acoustic features "
        "but no verified raw-audio feature extractor matching the saved model's 45-feature schema."
    )
    step_data = {
        "status": "failed",
        "outcome": {
            "passage_version": passage_version,
            "passage_text": passage_text,
            "audio_duration_seconds": payload.get("duration_seconds", 0),
            "sample_rate": payload.get("sample_rate", 44100),
            "model_family": adapter.family,
            "model_verified": False,
        },
        "error": error_message,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "speech", step_data)
    raise APIError(501, "raw_audio_feature_extractor_unavailable", error_message)


@assessment_bp.get("/api/assessments/<int:assessment_id>/report.pdf")
@login_required
def download_assessment_pdf(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")

    pdf_bytes = generate_assessment_pdf(assessment, {"id": g.user_id})
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=assessment_report_{assessment_id}.pdf",
            "Cache-Control": "no-cache",
        },
    )

