from __future__ import annotations

import logging

from flask import Flask, current_app, g, jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_payload(code: str, message: str, status_code: int):
    return jsonify({
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(g, "request_id", None),
        }
    }), status_code


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return _error_payload(error.code, error.message, error.status_code)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_request(_error):
        return _error_payload("request_too_large", "The request exceeds the configured upload limit.", 413)

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        status = error.code or 500
        messages = {
            404: ("not_found", "The requested API route was not found."),
            405: ("method_not_allowed", "This HTTP method is not supported for the route."),
        }
        code, message = messages.get(status, ("http_error", "The request could not be completed."))
        return _error_payload(code, message, status)

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        current_app.logger.exception(
            "unhandled_request_error",
            extra={"event": "unhandled_error", "request_id": getattr(g, "request_id", None)},
        )
        return _error_payload("internal_error", "The server could not complete the request.", 500)
