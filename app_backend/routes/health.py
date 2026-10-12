from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health():
    registry = current_app.extensions["model_registry"]
    readiness = registry.readiness()
    all_models_loaded = all(readiness[name]["model_loaded"] for name in ("speech", "handwriting"))
    return jsonify({
        "status": "ok" if all_models_loaded else "degraded",
        "models": readiness,
        "capabilities": {
            "speech_prediction": readiness["speech"]["prediction_available"],
            "spiral_prediction": readiness["handwriting"]["prediction_available"],
            "manual_tremor_prototype": True,
            "updrs_regression": False,
        },
    }), (200 if all_models_loaded else 503)
