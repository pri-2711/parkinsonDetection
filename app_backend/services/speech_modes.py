"""Recording modes for the speech assessment, and the three separate states of a saved speech result.

Every place that names a recording mode (the route, the summary, the PDF, the Results page) reads the mapping
here. A speech result has three independent states, which are never merged into one label:

* feature extraction: completed when speech_audio_v2 features were computed from the recording;
* prediction: completed only when the mode's trained model produced a verified probability;
* prediction unavailable: the features exist, but the model for this mode is not trained (or invalid).
"""
from __future__ import annotations

from typing import Any

from app_backend.errors import APIError

RECORDING_MODES = {
    "reading": {"label": "Reading passage", "model_file": "reading.joblib"},
    "sustained_vowel": {"label": 'Sustained "ah" vowel', "model_file": "sustained_vowel.joblib"},
}

BLOCKER_MESSAGES = {
    "speech_audio_v2_model_not_trained":
        "No trained speech_audio_v2 model exists for this recording mode, so no prediction was made.",
    "speech_audio_v2_model_invalid":
        "The trained model file for this recording mode failed validation, so no prediction was made.",
}


def normalise_recording_mode(value: Any) -> str:
    """The mode exactly as sent, or a 422 error. Nothing is guessed or defaulted."""
    mode = str(value or "").strip()
    if mode not in RECORDING_MODES:
        raise APIError(422, "invalid_speech_mode", "mode must be 'sustained_vowel' or 'reading'.")
    return mode


def recording_mode_label(mode: Any) -> str | None:
    entry = RECORDING_MODES.get(mode) if isinstance(mode, str) else None
    return entry["label"] if entry else None


def speech_result_view(step: dict[str, Any] | None) -> dict[str, Any]:
    """The three states of one saved speech step, for the Results page, the summary and the PDF."""
    step = step or {}
    outcome = step.get("outcome") or {}
    status = step.get("status", "pending")
    mode = outcome.get("recording_mode")

    extraction_state = "completed" if outcome.get("feature_extraction") == "completed" else (
        "failed" if status == "failed" else "not_run")

    probability = outcome.get("probability")
    if status == "completed" and outcome.get("model_verified") is True and isinstance(probability, (int, float)) \
            and not isinstance(probability, bool):
        prediction_state = "completed"
        prediction_reason = None
    elif status == "unavailable" or outcome.get("prediction_status") == "unavailable":
        prediction_state = "unavailable"
        prediction_reason = BLOCKER_MESSAGES.get(
            outcome.get("blocker_code"), "Prediction is unavailable for this recording mode.")
        probability = None
    elif status == "failed":
        prediction_state = "not_run"
        prediction_reason = f"Audio processing failed{': ' + step['error'] if step.get('error') else '.'}"
        probability = None
    else:
        prediction_state = "not_run"
        prediction_reason = "No recording has been processed yet." if status == "pending" else "Processing."
        probability = None

    return {
        "recording_mode": mode if mode in RECORDING_MODES else None,
        "recording_mode_label": recording_mode_label(mode),
        "feature_extraction": extraction_state,
        "feature_count": outcome.get("feature_count") if extraction_state == "completed" else None,
        "feature_schema": outcome.get("feature_schema") if extraction_state == "completed" else None,
        "prediction": prediction_state,
        "prediction_reason": prediction_reason,
        "probability": float(probability) if prediction_state == "completed" else None,
        "prediction_value": outcome.get("prediction") if prediction_state == "completed" else None,
        "model": f"speech_audio_v2 {mode} model" if prediction_state == "completed" and mode else None,
    }
