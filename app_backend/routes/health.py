from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/")
@health_bp.get("/api")
def root_status():
    return jsonify({
        "service": "Parkinson's Disease Detection Research Prototype API",
        "version": "1.0.0",
        "status": "online",
        "health_endpoint": "/api/health",
        "modalities": {
            "tremor": "7-item UPDRS III classifier (Calibrated SVM)",
            "handwriting": "Tablet trajectory kinematics classifier (Calibrated SVM)",
            "speech": "Praat-based speech_audio_v2 extraction; mode-specific v2 classifier required",
            "fusion": "Decision-level AUC-weighted late fusion"
        },
        "disclaimer": "Academic research prototype for experimental study. Not a clinical diagnostic tool.",
    }), 200


@health_bp.get("/api/health")
def health():
    registry = current_app.extensions["model_registry"]
    readiness = registry.readiness()
    audio_v2_models = current_app.extensions["speech_audio_v2_models"]
    readiness["speech_audio_v2"] = {
        mode: {"model_loaded": model.ready, "error_code": model.error_code,
               "feature_schema": "speech_audio_v2", "expected_feature_count": 45}
        for mode, model in audio_v2_models.items()
    }
    readiness["speech"]["prediction_available"] = any(model.ready for model in audio_v2_models.values())
    readiness["speech"]["legacy_baseline_loaded"] = readiness["speech"]["model_loaded"]
    readiness["speech"]["probability_output"] = readiness["speech"]["prediction_available"]
    readiness["speech"]["blocker"] = None if readiness["speech"]["prediction_available"] else "speech_audio_v2_model_not_trained"
    all_models_loaded = all(readiness[name]["model_loaded"] for name in ("speech", "handwriting", "tremor"))
    return jsonify({
        "status": "ok" if all_models_loaded else "degraded",
        "models": readiness,
        "capabilities": {
            "speech_prediction": readiness["speech"]["prediction_available"],
            "spiral_prediction": readiness["handwriting"]["camera_input_prediction_available"],
            "tremor_classification": readiness["tremor"]["prediction_available"],
            "manual_tremor_prototype": True,
            "tablet_trajectory_classification": readiness["handwriting"]["prediction_available"],
            "updrs_regression": False,
        },
        "disclaimer": "Research prototype only. Predictions are experimental probabilities, not medical diagnoses."
    }), (200 if all_models_loaded else 503)

