"""User authentication: registration, login, JWT token management.

Passwords are hashed with werkzeug's PBKDF2 (ships with Flask).
Tokens are HMAC-SHA256 JWTs using PyJWT when available, otherwise a
simple signed fallback using Python's built-in hmac + base64.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from functools import wraps

from flask import current_app, g, request
from werkzeug.security import check_password_hash, generate_password_hash

from app_backend.errors import APIError

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOKEN_EXPIRY_SECONDS = 86400  # 24 hours


def _validate_registration(payload: dict) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_json", "A JSON object with username, email, and password is required.")
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not _USERNAME_RE.match(username):
        raise APIError(422, "invalid_username", "Username must be 3–30 alphanumeric characters or underscores.")
    if not _EMAIL_RE.match(email):
        raise APIError(422, "invalid_email", "Provide a valid email address.")
    if len(password) < 8:
        raise APIError(422, "weak_password", "Password must be at least 8 characters.")
    return username, email, password


def _validate_login(payload: dict) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_json", "A JSON object with username and password is required.")
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        raise APIError(422, "missing_credentials", "Both username and password are required.")
    return username, password


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(pw_hash: str, password: str) -> bool:
    return check_password_hash(pw_hash, password)


# ── JWT token helpers ──────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_token(user_id: int, secret: str) -> str:
    """Create a simple HMAC-SHA256 JWT."""
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(json.dumps({
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
    }).encode())
    signing_input = f"{header}.{payload}"
    signature = _b64url_encode(
        hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    return f"{signing_input}.{signature}"


def decode_token(token: str, secret: str) -> dict:
    """Decode and verify a simple HMAC-SHA256 JWT."""
    parts = token.split(".")
    if len(parts) != 3:
        raise APIError(401, "invalid_token", "Malformed authentication token.")
    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    actual_sig = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise APIError(401, "invalid_token", "Authentication token signature is invalid.")
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise APIError(401, "invalid_token", "Authentication token payload is malformed.")
    if payload.get("exp", 0) < time.time():
        raise APIError(401, "token_expired", "Authentication token has expired. Please log in again.")
    return payload


def login_required(f):
    """Flask route decorator that validates the JWT and sets g.user_id."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise APIError(401, "missing_token", "Include an Authorization: Bearer <token> header.")
        token = auth_header[7:].strip()
        secret = current_app.config.get("SECRET_KEY") or ""
        if not secret:
            raise APIError(500, "server_misconfigured", "Server secret key is not configured.")
        payload = decode_token(token, secret)
        user_id = payload.get("sub")
        if not isinstance(user_id, int):
            raise APIError(401, "invalid_token", "Token does not contain a valid user identifier.")
        g.user_id = user_id
        return f(*args, **kwargs)
    return decorated


def register_user(db, payload: dict) -> dict:
    username, email, password = _validate_registration(payload)
    if db.get_user_by_username(username):
        raise APIError(409, "username_taken", "This username is already registered.")
    if db.get_user_by_email(email):
        raise APIError(409, "email_taken", "This email address is already registered.")
    pw_hash = hash_password(password)
    user = db.create_user(username, email, pw_hash)
    return user


def authenticate_user(db, payload: dict, secret: str) -> dict:
    username, password = _validate_login(payload)
    user = db.get_user_by_username(username)
    if not user or not verify_password(user["pw_hash"], password):
        raise APIError(401, "invalid_credentials", "Invalid username or password.")
    token = create_token(user["id"], secret)
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "email": user["email"]}}
