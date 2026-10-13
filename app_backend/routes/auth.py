from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, request

from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.services.auth import authenticate_user, login_required, register_user

auth_bp = Blueprint("auth", __name__)


def _get_db() -> Database:
    if "db" not in current_app.extensions:
        db = Database(current_app.config["DATABASE_PATH"])
        current_app.extensions["db"] = db
    return current_app.extensions["db"]


def _require_secret() -> str:
    secret = current_app.config.get("SECRET_KEY")
    if not isinstance(secret, str) or not secret.strip():
        raise APIError(500, "server_misconfigured", "Authentication signing key is not configured. Set APP_SECRET_KEY in the repository-root .env file.")
    return secret


@auth_bp.post("/api/auth/register")
def register():
    _require_secret()
    payload = g.get("json_payload")
    if payload is None:
        try:
            payload = request.get_json()
        except Exception:
            raise APIError(400, "invalid_json", "Request body must be valid JSON.")
    db = _get_db()
    user = register_user(db, payload)
    return jsonify({"id": user["id"], "username": user["username"], "email": user["email"]}), 201


@auth_bp.post("/api/auth/login")
def login():
    payload = g.get("json_payload")
    if payload is None:
        try:
            payload = request.get_json()
        except Exception:
            raise APIError(400, "invalid_json", "Request body must be valid JSON.")
    db = _get_db()
    secret = _require_secret()
    result = authenticate_user(db, payload, secret)
    return jsonify(result), 200


@auth_bp.get("/api/auth/me")
@login_required
def get_current_user():
    db = _get_db()
    user = db.get_user_by_id(g.user_id)
    if not user:
        raise APIError(404, "user_not_found", "User not found.")
    return jsonify({"id": user["id"], "username": user["username"], "email": user["email"]}), 200
