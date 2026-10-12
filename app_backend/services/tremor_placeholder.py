from __future__ import annotations

from app_backend.schemas import validate_tremor_payload


def create_prototype_submission(payload, allowed_locations, max_intensity):
    values = validate_tremor_payload(payload, allowed_locations, max_intensity)
    session_id = values.pop("session_id")
    result = {
        "status": "submitted",
        "prototype": True,
        "classification": None,
        "updrs_estimate": None,
        "data": values,
    }
    return result, session_id
