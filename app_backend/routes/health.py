from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/")
@health_bp.get("/api")
def root_status():
    return jsonify({
        "service": "Parkinson's Disease Detection Research Prototype API (CSE Final-Year Project)",
        "version": "1.0.0",
        "status": "online",
        "health_endpoint": "/api/health",
        "modalities": {
            "tremor": "7-item UPDRS III classifier (Calibrated SVM)",
            "handwriting": "Tablet trajectory kinematics classifier (Calibrated SVM)",
            "speech": "45 acoustic dysphonia features classifier (Calibrated SVM)",
            "fusion": "Decision-level AUC-weighted late fusion"
        },
        "disclaimer": "Academic research prototype for experimental study. Not a clinical diagnostic tool.",
    }), 200


@health_bp.get("/api/health")
def health():
    registry = current_app.extensions["model_registry"]
    readiness = registry.readiness()
    all_models_loaded = all(readiness[name]["model_loaded"] for name in ("speech", "handwriting", "tremor"))
    return jsonify({
        "status": "ok" if all_models_loaded else "degraded",
        "models": readiness,
        "capabilities": {
            "speech_prediction": readiness["speech"]["prediction_available"],
            "spiral_prediction": readiness["handwriting"]["prediction_available"],
            "tremor_classification": readiness["tremor"]["prediction_available"],
            "manual_tremor_prototype": True,
            "tablet_trajectory_classification": True,
            "updrs_regression": False,
        },
        "disclaimer": "Research prototype only. Predictions are experimental probabilities, not medical diagnoses."
    }), (200 if all_models_loaded else 503)

