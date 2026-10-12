from __future__ import annotations

from app_backend.errors import APIError


def validate_tremor_payload(payload, allowed_locations, max_intensity):
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_json", "A JSON object is required.")
    allowed = {"location", "intensity", "session_id"}
    if set(payload) - allowed:
        raise APIError(422, "unexpected_fields", "Only location, intensity, and optional session_id are accepted.")
    location = payload.get("location")
    intensity = payload.get("intensity")
    if not isinstance(location, str) or location.strip().lower() not in allowed_locations:
        raise APIError(422, "invalid_tremor_location", "Choose a supported body location.")
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
        raise APIError(422, "invalid_tremor_intensity", "Intensity must be a number from 0 through 10.")
    if not 0 <= float(intensity) <= max_intensity:
        raise APIError(422, "invalid_tremor_intensity", f"Intensity must be between 0 and {max_intensity:g}.")
    session_id = payload.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise APIError(422, "invalid_session_id", "session_id must be a non-empty string when provided.")
    return {
        "location": location.strip().lower(),
        "intensity": float(intensity),
        "session_id": session_id.strip() if session_id else None,
    }
