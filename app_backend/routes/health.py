from flask import Blueprint, current_app, jsonify

from app_backend.services.acoustic_features import SPEECH_V2_FEATURES

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
               "feature_schema": "speech_audio_v2", "expected_feature_count": len(SPEECH_V2_FEATURES)}
        for mode, model in audio_v2_models.items()
    }
    # The legacy 45-feature SVM is kept as a saved baseline only. It is never
    # called with speech_audio_v2 values, so it must not count as speech readiness.
    speech = readiness["speech"]
    speech["legacy_baseline_loaded"] = speech["model_loaded"]
    speech["model_loaded"] = any(model.ready for model in audio_v2_models.values())
    speech["prediction_available"] = speech["model_loaded"]
    speech["probability_output"] = speech["prediction_available"]
    speech["status"] = "prediction_available" if speech["prediction_available"] else "extraction_only"
    speech["blocker"] = None if speech["prediction_available"] else "speech_audio_v2_model_not_trained"
    # Overall health covers the modalities that can currently predict. Speech
    # extraction is reported separately so a blocked modality is not a fault.
    enabled_models_loaded = all(readiness[name]["model_loaded"] for name in ("handwriting", "tremor"))
    from app_backend.services.spiral_video_analysis import ANALYSIS_VERSION
    return jsonify({
        "status": "ok" if enabled_models_loaded else "degraded",
        "versions": {"spiral_video_analysis": ANALYSIS_VERSION},
        "models": readiness,
        "capabilities": {
            "speech_feature_extraction": True,
            "speech_prediction": speech["prediction_available"],
            "spiral_prediction": readiness["handwriting"]["camera_input_prediction_available"],
            "tremor_classification": readiness["tremor"]["prediction_available"],
            "manual_tremor_prototype": True,
            "tablet_trajectory_classification": readiness["handwriting"]["prediction_available"],
            "updrs_regression": False,
        },
        "disclaimer": "Research prototype only. Predictions are experimental probabilities, not medical diagnoses."
    }), (200 if enabled_models_loaded else 503)

