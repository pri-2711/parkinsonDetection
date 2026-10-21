"""Flask API for the Parkinson's assessment prototype."""

from __future__ import annotations

import logging
import os
import uuid

from flask import Flask, g, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from app_backend.config import BackendConfig, load_repository_environment
from app_backend.errors import APIError, register_error_handlers
from app_backend.logging_utils import configure_logging
from app_backend.models.registry import ModelRegistry
from app_backend.services.sessions import AssessmentSessionStore
from app_backend.services.speech_v2_model import SpeechV2Model


def create_app(test_config: dict | None = None) -> Flask:
    load_repository_environment()
    app = Flask(__name__)
    app.config.from_object(BackendConfig)
    # BackendConfig is a class with import-time defaults; refresh auth settings
    # so app creation always sees the current repository-root .env.
    app.config["APP_SECRET_KEY"] = os.getenv("APP_SECRET_KEY")
    app.config["SECRET_KEY"] = app.config.get("APP_SECRET_KEY")
    if test_config:
        app.config.update(test_config)
    app.config["SECRET_KEY"] = app.config.get("APP_SECRET_KEY") or app.config.get("SECRET_KEY")

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    configure_logging(app.config.get("LOG_LEVEL", "INFO"))
    app.extensions["assessment_sessions"] = AssessmentSessionStore(
        ttl_seconds=app.config["SESSION_TTL_SECONDS"]
    )
    app.extensions["model_registry"] = ModelRegistry.from_config(app.config)
    app.extensions["speech_audio_v2_models"] = {
        mode: SpeechV2Model(app.config["SPEECH_AUDIO_V2_ARTIFACT_DIR"], mode)
        for mode in ("sustained_vowel", "reading")
    }

    from app_backend.routes.assessment import assessment_bp
    from app_backend.routes.auth import auth_bp
    from app_backend.routes.health import health_bp
    from app_backend.routes.history import history_bp
    from app_backend.routes.results import results_bp
    from app_backend.routes.speech import speech_bp
    from app_backend.routes.spiral import spiral_bp
    from app_backend.routes.tremor import tremor_bp

    for blueprint in (health_bp, speech_bp, spiral_bp, tremor_bp, assessment_bp, auth_bp, history_bp, results_bp):
        app.register_blueprint(blueprint)

    @app.before_request
    def assign_request_id():
        g.request_id = uuid.uuid4().hex

    @app.before_request
    def parse_json_payload():
        if request.is_json:
            try:
                g.json_payload = request.get_json()
            except Exception:
                g.json_payload = None

    @app.after_request
    def log_request(response):
        logging.getLogger("app_backend.http").info(
            "request_completed",
            extra={
                "event": "http_request",
                "request_id": getattr(g, "request_id", None),
                "method": request.method,
                "endpoint": request.endpoint or "unmatched",
                "status_code": response.status_code,
            },
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        response.headers["Cache-Control"] = "no-store"
        return response

    register_error_handlers(app)
    return app


__all__ = ["create_app"]
