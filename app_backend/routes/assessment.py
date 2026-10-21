from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, Response, request, send_file
from werkzeug.datastructures import FileStorage
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
from app_backend.services.audio_preprocessing import validate_wav
from app_backend.services.acoustic_features import AcousticFeatureExtractor
from app_backend.services.spiral_preprocessing import validate_spiral_video
from app_backend.services.spiral_video_analysis import analyze_spiral_video
from app_backend.services.canvas_telemetry import summarize_canvas_telemetry
import re
import uuid
from pathlib import Path

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
            fusion_result["missing_modalities"] = [
                name for name, value in (("speech", speech_prob), ("handwriting", handwriting_prob), ("tremor", tremor_prob))
                if value is None
            ]
            fusion_result["aggregation_type"] = "single_modality_probability" if len(fusion_result["available_modalities"]) == 1 else "auc_weighted_decision_level_score"
            fusion_result["score_interpretation"] = (
                "One model's probability; no multimodal fusion was performed."
                if len(fusion_result["available_modalities"]) == 1 else
                "AUC-weighted mean of independent modality model probabilities. It is not calibrated or validated as a patient-level multimodal probability."
            )
            fusion_result["output_type"] = "research_model_aggregation_not_clinical_diagnosis"
            fusion_result["threshold"] = 0.5
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

    existing_outcome = assessment.get("steps", {}).get("speech", {}).get("outcome") or {}
    if isinstance(outcome, dict) and isinstance(existing_outcome, dict):
        outcome = {**existing_outcome, **outcome}
    elif outcome is None and isinstance(existing_outcome, dict) and existing_outcome:
        outcome = existing_outcome

    step_data = {
        "status": "completed" if outcome and outcome.get("model_verified") is True else ("unavailable" if outcome else "failed"),
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

    existing_outcome = assessment.get("steps", {}).get("spiral", {}).get("outcome") or {}
    if isinstance(outcome, dict) and isinstance(existing_outcome, dict):
        outcome = {**existing_outcome, **outcome}
    elif outcome is None and isinstance(existing_outcome, dict) and existing_outcome:
        outcome = existing_outcome

    step_data = {
        "status": "completed" if outcome and outcome.get("model_verified") is True else ("unavailable" if outcome else "failed"),
        "outcome": outcome,
        "error": error,
        "updated_at": assessment.get("updated_at"),
    }
    updated = repo.update_assessment_step(assessment_id, g.user_id, "spiral", step_data)
    return jsonify(updated), 200


@assessment_bp.post("/api/assessments/<int:assessment_id>/spiral/canvas/observe")
@login_required
def save_canvas_observation(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")
    step = dict(assessment.get("steps", {}).get("spiral") or {})
    outcome = dict(step.get("outcome") or {})
    if not outcome.get("model_verified"):
        step.update({"status": "processing", "outcome": outcome, "error": None})
        repo.update_assessment_step(assessment_id, g.user_id, "spiral", step)
    try:
        observation = summarize_canvas_telemetry(g.get("json_payload"))
    except APIError as error:
        if not outcome.get("model_verified"):
            outcome["canvas_telemetry_error"] = {"code": error.code, "message": error.message}
            completed_video = (outcome.get("video_analysis") or {}).get("status") == "completed"
            step.update({
                "status": "unavailable" if completed_video else "failed",
                "outcome": outcome,
                "error": None if completed_video else error.message,
                "error_code": error.code,
            })
            repo.update_assessment_step(assessment_id, g.user_id, "spiral", step)
        raise
    outcome["canvas_telemetry"] = observation
    outcome.pop("canvas_telemetry_error", None)
    if not outcome.get("model_verified"):
        outcome["prediction_status"] = "unavailable"
        step.update({
            "status": "unavailable",
            "error": "Canvas telemetry does not contain all digitizer measurements required by the trained handwriting model.",
        })
    else:
        step.update({"status": "completed", "error": None})
    step.update({"outcome": outcome, "updated_at": assessment.get("updated_at")})
    updated = repo.update_assessment_step(assessment_id, g.user_id, "spiral", step)
    return jsonify({"assessment": updated, "observation": observation}), 200


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
    existing_outcome = assessment.get("steps", {}).get("spiral", {}).get("outcome") or {}
    if isinstance(existing_outcome, dict):
        outcome = {**existing_outcome, **outcome}
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

    measured_fields = {"x", "y", "z", "pressure", "grip_angle", "timestamp"}
    if any(not isinstance(point, dict) for point in points_data):
        raise APIError(422, "invalid_trajectory", "Each trajectory sample must be a JSON object.")
    missing_measurements = sorted({
        field for point in points_data for field in measured_fields if field not in point
    })
    if missing_measurements:
        raise APIError(
            422,
            "tablet_telemetry_required",
            "The handwriting classifier needs measured tablet x/y/z, pressure, grip angle, and timestamp values. "
            "A browser canvas supplies only 2D coordinates and timing; it cannot provide the missing digitizer measurements.",
        )

    # Format points into the PaHaW tablet schema expected by extract_tablet_trajectory_features
    # TABLET_COLUMNS = ("point_index", "x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id")
    try:
        rows = []
        for idx, pt in enumerate(points_data):
            rows.append({
                "point_index": int(pt.get("point_index", idx)),
                "x": float(pt["x"]),
                "y": float(pt["y"]),
                "z": float(pt["z"]),
                "pressure": float(pt["pressure"]),
                "grip_angle": float(pt["grip_angle"]),
                "timestamp": float(pt["timestamp"]),
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


@assessment_bp.post("/api/assessments/<int:assessment_id>/spiral/video/analyze")
@login_required
def analyze_spiral_video_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")
    uploaded = request.files.get("file")
    if not isinstance(uploaded, FileStorage) or not uploaded.filename:
        raise APIError(400, "missing_video", "Attach the spiral recording as multipart field 'file'.")
    data = uploaded.read(current_app.config["SPIRAL_MAX_BYTES"] + 1)
    steps = assessment.get("steps", {})
    spiral = dict(steps.get("spiral") or {})
    outcome = dict(spiral.get("outcome") or {})
    source_name = Path(uploaded.filename).name[:255]
    outcome["video_analysis"] = {"status": "processing", "source_filename": source_name}
    if not outcome.get("model_verified"):
        spiral["status"] = "processing"
    spiral.update({"outcome": outcome, "updated_at": assessment.get("updated_at")})
    repo.update_assessment_step(assessment_id, g.user_id, "spiral", spiral)
    try:
        metadata = validate_spiral_video(data, uploaded.filename, uploaded.content_type, current_app.config)
        annotated, output_ext, analysis = analyze_spiral_video(data, uploaded.filename, current_app.config)
    except APIError as error:
        outcome["video_analysis"] = {"status": "failed", "source_filename": source_name,
                                     "error": {"code": error.code, "message": error.message}}
        if not outcome.get("model_verified"):
            spiral["status"] = "unavailable"
            spiral["error"] = "The video analysis did not complete; camera video is not a compatible classifier input."
        spiral.update({"outcome": outcome, "updated_at": assessment.get("updated_at")})
        repo.update_assessment_step(assessment_id, g.user_id, "spiral", spiral)
        raise

    video_id = uuid.uuid4().hex
    media_root = Path(current_app.config["SPIRAL_MEDIA_DIR"]).resolve()
    assessment_dir = (media_root / f"assessment_{assessment_id}").resolve()
    if media_root not in assessment_dir.parents:
        raise APIError(500, "media_storage_failed", "Private video storage could not be prepared.")
    assessment_dir.mkdir(parents=True, exist_ok=True)
    source_ext = Path(uploaded.filename).suffix.lower()
    original_path = assessment_dir / f"{video_id}_original{source_ext}"
    annotated_path = assessment_dir / f"{video_id}_annotated.{output_ext}"
    try:
        original_path.write_bytes(data)
        annotated_path.write_bytes(annotated)
        outcome["video_analysis"] = {
            **analysis,
            "status": "completed",
            "source_filename": source_name,
            "source_metadata": metadata,
            "video_id": video_id,
            "annotated_video_url": f"/api/assessments/{assessment_id}/spiral/video/annotated",
            "original_video_url": f"/api/assessments/{assessment_id}/spiral/video/original",
        }
        if not outcome.get("model_verified"):
            outcome["prediction_status"] = "unavailable"
            spiral["status"] = "unavailable"
            spiral["error"] = "Video observations are experimental and are not compatible with the trained digitizer classifier."
        spiral["outcome"] = outcome
        spiral["updated_at"] = assessment.get("updated_at")
        # Keep the classifier's status/probability untouched; video analysis is a separate sidecar.
        updated = repo.update_assessment_step(assessment_id, g.user_id, "spiral", spiral)
        if not updated:
            raise RuntimeError("assessment disappeared during video persistence")
    except Exception:
        original_path.unlink(missing_ok=True)
        annotated_path.unlink(missing_ok=True)
        outcome["video_analysis"] = {"status": "failed", "source_filename": source_name,
                                     "error": {"code": "video_persistence_failed", "message": "Analysis output could not be associated with this assessment."}}
        if not outcome.get("model_verified"):
            spiral["status"] = "unavailable"
            spiral["error"] = "The video analysis output could not be saved; camera video is not a compatible classifier input."
        spiral.update({"outcome": outcome, "updated_at": assessment.get("updated_at")})
        repo.update_assessment_step(assessment_id, g.user_id, "spiral", spiral)
        raise APIError(500, "video_persistence_failed", "Video analysis succeeded but could not be associated with this assessment.") from None
    return jsonify({"assessment_id": assessment_id, "analysis": outcome["video_analysis"]}), 200


def _associated_spiral_video(assessment_id: int, variant: str):
    assessment = _get_assessment_repo().get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    outcome = assessment.get("steps", {}).get("spiral", {}).get("outcome", {})
    video = outcome.get("video_analysis", {}) if isinstance(outcome, dict) else {}
    video_id = video.get("video_id") if isinstance(video, dict) else None
    if not isinstance(video_id, str) or not re.fullmatch(r"[0-9a-f]{32}", video_id):
        raise APIError(404, "video_not_found", "No video is associated with this assessment.")
    suffix = "annotated.mp4" if variant == "annotated" else f"original{Path(video.get('source_filename', '')).suffix.lower()}"
    path = (Path(current_app.config["SPIRAL_MEDIA_DIR"]).resolve() / f"assessment_{assessment_id}" / f"{video_id}_{suffix}").resolve()
    if not path.is_file() or Path(current_app.config["SPIRAL_MEDIA_DIR"]).resolve() not in path.parents:
        raise APIError(404, "video_not_found", "The associated video file is not available.")
    mime = "video/mp4" if variant == "annotated" else {
        ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".webm": "video/webm", ".mp4": "video/mp4"
    }.get(path.suffix.lower(), "application/octet-stream")
    return send_file(path, mimetype=mime, as_attachment=False, conditional=True, max_age=0)


@assessment_bp.get("/api/assessments/<int:assessment_id>/spiral/video/annotated")
@login_required
def get_annotated_spiral_video(assessment_id: int):
    return _associated_spiral_video(assessment_id, "annotated")


@assessment_bp.get("/api/assessments/<int:assessment_id>/spiral/video/original")
@login_required
def get_original_spiral_video(assessment_id: int):
    return _associated_spiral_video(assessment_id, "original")


@assessment_bp.post("/api/assessments/<int:assessment_id>/speech/predict")
@login_required
def predict_speech_step(assessment_id: int):
    repo = _get_assessment_repo()
    assessment = repo.get_assessment(assessment_id, g.user_id)
    if not assessment:
        raise APIError(404, "assessment_not_found", "Assessment not found.")
    if assessment["status"] == "completed":
        raise APIError(409, "assessment_already_completed", "Assessment is already completed.")
    prior_step = dict(assessment.get("steps", {}).get("speech") or {})
    prior_outcome = dict(prior_step.get("outcome") or {})
    prior_step.update({"status": "processing", "outcome": prior_outcome, "error": None})
    repo.update_assessment_step(assessment_id, g.user_id, "speech", prior_step)
    mode = request.form.get("mode", "")
    try:
        uploaded = request.files.get("file")
        if not isinstance(uploaded, FileStorage) or not uploaded.filename:
            raise APIError(400, "missing_audio", "Attach the recording as multipart field 'file'.")
        audio_bytes = uploaded.read(current_app.config["AUDIO_MAX_BYTES"] + 1)
        audio_metadata = validate_wav(audio_bytes, uploaded.filename, uploaded.content_type, current_app.config)
        try:
            gender_code = int(request.form.get("gender_code", ""))
        except ValueError:
            raise APIError(422, "invalid_gender_code", "Provide gender_code as 0 or 1.") from None
        extractor = AcousticFeatureExtractor()
        features = extractor.extract(audio_bytes, audio_metadata, mode=mode, gender_code=gender_code)
        feature_values = {name: float(value) for name, value in features.iloc[0].items()}
        base_outcome = {
            "audio_received": True, "audio_stored": False, "recording_mode": mode,
            "audio_duration_seconds": audio_metadata["duration_seconds"],
            "sample_rate": audio_metadata["sample_rate_hz"], "audio_format": audio_metadata["format"],
            "audio_channels": audio_metadata["channels"], "feature_schema": extractor.schema,
            "feature_extraction": "completed", "feature_values": feature_values,
            "feature_count": len(feature_values), "gender_code": gender_code,
            "passage_version": request.form.get("passage_version") if mode == "reading" else None,
            "model_verified": False,
        }
        v2_model = current_app.extensions["speech_audio_v2_models"][mode]
        if not v2_model.ready:
            error_message = (
                "Speech features were extracted and saved under speech_audio_v2, but no trained model is available "
                "for this recording mode. The incompatible legacy 45-feature SVM was not used."
            )
            outcome = {**base_outcome, "prediction_status": "unavailable"}
            step_data = {"status": "unavailable", "outcome": outcome, "error": error_message,
                         "updated_at": assessment.get("updated_at")}
            updated = repo.update_assessment_step(assessment_id, g.user_id, "speech", step_data)
            return jsonify({"assessment": updated, "outcome": outcome, "status": "unavailable"}), 200

        try:
            result = v2_model.predict(features)
        except Exception:
            raise APIError(500, "speech_prediction_failed", "The trained speech_audio_v2 model could not produce a valid prediction.") from None
        outcome = {**base_outcome, **result, "prediction_status": "completed", "model_family": "speech_audio_v2_svc",
                   "model_verified": True, "output_type": "research_model_probability_not_clinical_diagnosis"}
        step_data = {"status": "completed", "outcome": outcome, "error": None,
                     "updated_at": assessment.get("updated_at")}
        updated = repo.update_assessment_step(assessment_id, g.user_id, "speech", step_data)
        return jsonify({"assessment": updated, "outcome": outcome, "status": "completed"}), 200
    except APIError as error:
        if not prior_outcome.get("model_verified"):
            failure_step = {"status": "failed", "outcome": prior_outcome, "error": error.message,
                            "error_code": error.code, "updated_at": assessment.get("updated_at")}
        else:
            failure_step = {**prior_step, "status": "completed", "outcome": prior_outcome,
                            "error": None, "last_attempt_error": {"code": error.code, "message": error.message}}
        repo.update_assessment_step(assessment_id, g.user_id, "speech", failure_step)
        raise
    except Exception:
        if not prior_outcome.get("model_verified"):
            repo.update_assessment_step(assessment_id, g.user_id, "speech", {
                "status": "failed", "outcome": prior_outcome,
                "error": "Unexpected audio processing failure.", "error_code": "speech_processing_failed",
                "updated_at": assessment.get("updated_at"),
            })
        raise APIError(500, "speech_processing_failed", "Unexpected audio processing failure.") from None


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

