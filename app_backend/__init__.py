"""Flask API for the Parkinson's assessment prototype."""

from __future__ import annotations

import logging
import uuid

from flask import Flask, g, request
from werkzeug.exceptions import RequestEntityTooLarge

from app_backend.config import BackendConfig
from app_backend.errors import APIError, register_error_handlers
from app_backend.logging_utils import configure_logging
from app_backend.models.registry import ModelRegistry
from app_backend.services.sessions import AssessmentSessionStore


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(BackendConfig)
    if test_config:
        app.config.update(test_config)

    configure_logging(app.config.get("LOG_LEVEL", "INFO"))
    app.extensions["assessment_sessions"] = AssessmentSessionStore(
        ttl_seconds=app.config["SESSION_TTL_SECONDS"]
    )
    app.extensions["model_registry"] = ModelRegistry.from_config(app.config)

    from app_backend.routes.assessment import assessment_bp
    from app_backend.routes.health import health_bp
    from app_backend.routes.speech import speech_bp
    from app_backend.routes.spiral import spiral_bp
    from app_backend.routes.tremor import tremor_bp

    for blueprint in (health_bp, speech_bp, spiral_bp, tremor_bp, assessment_bp):
        app.register_blueprint(blueprint)

    @app.before_request
    def assign_request_id():
        g.request_id = uuid.uuid4().hex

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
