"""Single source of truth for what an assessment did, predicted, observed and fused.

The API summary endpoint, the Results page and the PDF report all read this
structure, so a modality is only described as a prediction when a validated
model actually produced a probability for it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app_backend.services.speech_modes import recording_mode_label, speech_result_view

MODALITY_ORDER = ("tremor", "spiral", "speech")

MODALITY_LABELS = {
    "tremor": "Tremor (7 UPDRS III items, 20–21)",
    "spiral": "Handwriting / spiral (tablet telemetry)",
    "speech": "Speech (speech_audio_v2 features)",
}

# Results.fusion uses the classifier-side names for the same modalities.
FUSION_NAMES = {"tremor": "tremor", "spiral": "handwriting", "speech": "speech"}

LIMITATIONS = [
    "Academic research prototype. Outputs are experimental model probabilities, not clinical diagnoses.",
    "Speech, handwriting and tremor come from independent participant cohorts. No patient-level multimodal validation exists, so a combined probability is not reported when more than one modality contributes.",
    "Speech prediction is disabled until a mode-specific speech_audio_v2 model is trained on labelled raw recordings with subject-disjoint evaluation. The legacy 45-feature speech SVM is never applied to these features.",
    "Camera and canvas inputs do not reproduce the tablet pressure and grip-angle measurements used by the handwriting classifier, so they are shown only as experimental observations.",
    "Tremor classification uses clinician-rated UPDRS III items; it does not measure tremor from video or audio.",
]


def _verified_probability(step: dict[str, Any]) -> float | None:
    outcome = step.get("outcome") or {}
    value = outcome.get("probability")
    if step.get("status") != "completed" or outcome.get("model_verified") is not True:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _observations(modality: str, step: dict[str, Any]) -> list[str]:
    outcome = step.get("outcome") or {}
    notes: list[str] = []
    if modality == "speech" and outcome.get("feature_extraction") == "completed":
        mode_label = recording_mode_label(outcome.get("recording_mode")) or "unrecorded-mode"
        notes.append(
            f"{outcome.get('feature_count', 0)} {outcome.get('feature_schema', 'speech_audio_v2')} features extracted "
            f"from a {mode_label.lower()} recording (audio not retained)."
        )
    if modality == "spiral":
        canvas = outcome.get("canvas_telemetry")
        if isinstance(canvas, dict):
            notes.append(f"Browser-canvas telemetry saved: {canvas.get('sample_count', 0)} samples, not classifier-compatible.")
        video = outcome.get("video_analysis")
        if isinstance(video, dict) and video.get("status") == "completed":
            notes.append("Spiral video analysis completed: visual observations only, not classifier-compatible.")
    return notes


def _prediction_reason(modality: str, step: dict[str, Any]) -> str:
    status = step.get("status", "pending")
    outcome = step.get("outcome") or {}
    if _verified_probability(step) is not None:
        return "Verified classifier probability saved."
    if modality == "speech":
        view = speech_result_view(step)
        if view["prediction"] == "unavailable":
            return f"Unavailable: {view['prediction_reason']}"
        if status == "failed":
            return "Not predicted: audio processing failed."
    if modality == "spiral" and status == "unavailable":
        return "Not predicted: no compatible digitizer telemetry was saved."
    if status == "failed":
        return "Not predicted: this step failed."
    if status == "pending":
        return "Not yet submitted."
    return "No verified prediction saved."


def build_assessment_summary(assessment: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    steps = assessment.get("steps") or {}
    results = assessment.get("results") or {}
    finalized = assessment.get("status") == "completed"
    fusion_record = results.get("fusion") if finalized else None
    contributors = set((fusion_record or {}).get("available_modalities") or [])

    modalities = []
    observations_all: list[dict[str, str]] = []
    for modality in MODALITY_ORDER:
        step = steps.get(modality) or {"status": "pending", "outcome": None}
        probability = _verified_probability(step)
        observations = _observations(modality, step)
        fusion_name = FUSION_NAMES[modality]
        contributes = finalized and fusion_name in contributors
        modalities.append({
            "modality": modality,
            "label": MODALITY_LABELS[modality],
            "step_status": step.get("status", "pending"),
            "processed": step.get("status") in ("completed", "unavailable"),
            "prediction_available": probability is not None,
            "probability": probability,
            "prediction_note": _prediction_reason(modality, step),
            "observations": observations,
            "speech_result": speech_result_view(step) if modality == "speech" else None,
            "contributes_to_fusion": contributes,
            "error": step.get("error"),
        })
        observations_all.extend({"modality": modality, "text": text} for text in observations)

    predicted = [item for item in modalities if item["prediction_available"]]
    contributing_labels = [item["label"] for item in modalities if item["contributes_to_fusion"]]
    if not finalized:
        fusion_state = "not_finalized"
        fusion_note = "Fusion runs only when the assessment is finalized, and only from verified predictions."
    elif not contributing_labels:
        fusion_state = "no_contributions"
        fusion_note = "Finalized without any verified prediction; no fusion was computed."
    elif len(contributing_labels) == 1:
        fusion_state = "single_modality"
        fusion_note = "One verified prediction contributed; it is that model's probability, not a multimodal result."
    else:
        fusion_state = "multi_modality_aggregate"
        fusion_note = (
            "Several verified predictions contributed to an AUC-weighted score. The modality cohorts are unmatched, "
            "so this combined value is not displayed as a probability."
        )

    single_probability = None
    if fusion_state == "single_modality":
        single_probability = next(item["probability"] for item in modalities if item["contributes_to_fusion"])

    return {
        "assessment_id": assessment.get("id"),
        "status": assessment.get("status"),
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "modalities": modalities,
        "completed_modalities": [item["label"] for item in modalities if item["processed"]],
        "predictions_available": [item["label"] for item in predicted],
        "experimental_observations": observations_all,
        "fusion": {
            "state": fusion_state,
            "contributing_modalities": contributing_labels,
            "fused_probability_displayed": fusion_state == "single_modality",
            "single_modality_probability": single_probability,
            "note": fusion_note,
        },
        "limitations": list(LIMITATIONS),
    }


def fusion_probability_is_displayable(fusion: dict[str, Any] | None) -> bool:
    """A combined probability is shown only when exactly one model contributed."""
    if not fusion:
        return False
    return len(fusion.get("available_modalities") or []) == 1
