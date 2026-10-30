"""Read-only research summaries built from saved experiment outputs.

Every number returned here is loaded from files under
``experiments/regularized_svm_rf/results`` or the speech v2 artifact folder.
Nothing is fitted or copied by hand, so a regenerated experiment flows through
without editing this module.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app_backend.services.acoustic_features import SPEECH_V2_FEATURES, SPEECH_V2_MODEL_FEATURES

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
MODEL_KEYS = ("baseline_svm", "tuned_svm", "baseline_random_forest", "tuned_random_forest", "nested_selected")
SPEECH_V2_MODES = ("sustained_vowel", "reading")

MODALITY_META = {
    "speech": {
        "label": "Speech (legacy precomputed 45-feature descriptors)",
        "dataset": "Replicated acoustic features of the Parkinson speech database (UCI). Precomputed descriptors, not raw audio.",
        "live_use": "Not used for live predictions. The app's speech_audio_v2 features cannot be fed to this model.",
    },
    "handwriting": {
        "label": "Handwriting / spiral (tablet trajectory kinematics)",
        "dataset": "PaHaW tablet drawing telemetry, 16 kinematic and pressure features.",
        "live_use": "Live prediction only from complete measured tablet telemetry (x, y, z, pressure, grip angle, timestamp).",
    },
    "tremor": {
        "label": "Tremor (7 UPDRS III items 20–21)",
        "dataset": "Clinician-rated UPDRS III tremor items, one row per participant evaluation.",
        "live_use": "Live prediction from exact integer ratings 0–4 for the seven items.",
    },
}

ARCHITECTURE = [
    "Flask application factory (app_backend) with SQLite persistence (users, assessments) and HMAC-SHA256 token authentication.",
    "Assessment workflow: speech, spiral and tremor steps persist status, outcome and error per step; results are released only after finalization.",
    "Saved calibrated SVM and Random Forest classifiers per modality, loaded read-only through a validating adapter that checks feature order and class order [0, 1].",
    "Decision-level late fusion: AUC-normalized weighted average of verified modality probabilities, with weights renormalized over the modalities that contributed.",
    "Speech v2 extractor (Parselmouth/Praat plus SciPy) for WAV input, separate from the legacy classifier; a reading-mode model trained on labelled raw audio is experimental; sustained-vowel prediction stays disabled.",
    "React 18 + Vite single-page frontend; PDF reports generated with ReportLab.",
]

VALIDATION = {
    "outer_splitter": "StratifiedGroupKFold, 5 outer folds, seed 42",
    "inner_selection": "Grouped 4-fold grid search selecting by ROC-AUC",
    "calibration": "Sigmoid (Platt) calibration fitted with grouped 3-fold CV inside training folds",
    "fusion_weight_fold": "Fold 1 is reserved for fusion-weight estimation only and excluded from reported per-modality metrics",
    "reported_folds": "Folds 2–5",
    "threshold": 0.5,
    "patient_level_multimodal_validation": "Not performed: the modality cohorts are unmatched.",
    "external_validation": "Pending. No independent local test cohort was identified.",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_summary(folds: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for metric in METRIC_KEYS:
        values = [float(fold[key][metric]) for fold in folds if key in fold and metric in fold[key]]
        summary[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
    return summary


def _modality(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    folds = metrics.get("outer_fold_metrics", [])
    deployment = metrics.get("deployment_artifacts", {})
    deployed_family = metrics.get("deployment_family")
    deployed = deployment.get(deployed_family, {}) if deployed_family else {}
    return {
        "modality": name,
        "label": MODALITY_META[name]["label"],
        "dataset": MODALITY_META[name]["dataset"],
        "live_use": MODALITY_META[name]["live_use"],
        "rows": metrics.get("row_count"),
        "participant_groups": metrics.get("participant_groups"),
        "feature_count": metrics.get("feature_count"),
        "reported_folds": metrics.get("reported_test_folds", []),
        "reported_rows": metrics.get("metrics", {}).get("reported_rows"),
        "pooled_metrics": {key: metrics["metrics"][key] for key in MODEL_KEYS if key in metrics.get("metrics", {})},
        "fold_metrics": {key: _fold_summary(folds, key) for key in MODEL_KEYS if folds and key in folds[0]},
        "deployed_family": deployed_family,
        "deployed_params": deployed.get("best_params"),
        "inner_cv_roc_auc": {family: data.get("inner_cv_best_roc_auc") for family, data in deployment.items()},
        "fusion_validation": metrics.get("fusion_weight_validation"),
        "source": f"{name}_metrics.json",
    }


def build_research_insights(results_dir: str | Path, artifact_dir: str | Path, speech_v2_dir: str | Path) -> dict[str, Any]:
    results = Path(results_dir)
    modalities: list[dict[str, Any]] = []
    for name in ("speech", "handwriting", "tremor"):
        modalities.append(_modality(name, _load(results / f"{name}_metrics.json")))

    fusion = _load(results / "fusion_validation_weights.json")
    weights = fusion.get("normalized_weights", {})
    baseline_weights = fusion.get("baseline_v1_outer_oof_weights", {})

    speech_v2_dir = Path(speech_v2_dir)
    metrics_path = speech_v2_dir / "validation_metrics.json"
    validation = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    modes = {}
    for mode in SPEECH_V2_MODES:
        trained = (speech_v2_dir / f"{mode}.joblib").is_file()
        entry = {"model_trained": trained}
        if mode in validation:
            result = validation[mode]
            svm = result["families"]["svm"]["outer_fold_summary"]["roc_auc"]
            entry.update({
                "recordings": result["n_recordings"],
                "subjects": result["n_subjects"],
                "control_subjects": result["n_control_subjects"],
                "parkinson_subjects": result["n_parkinson_subjects"],
                "majority_class_accuracy": result["majority_class_accuracy"],
                "deployed_family": result["deployed_family"],
                "outer_fold_roc_auc_mean": svm["mean"],
                "outer_fold_roc_auc_sd": svm["sd"],
                "skipped_recordings": len(result["skipped_recordings"]),
                "validation": result["validation"],
            })
        modes[mode] = entry
    reading_ready = modes.get("reading", {}).get("model_trained", False)
    speech_v2 = {
        "schema": "speech_audio_v2",
        "feature_count": len(SPEECH_V2_MODEL_FEATURES),
        "features": list(SPEECH_V2_MODEL_FEATURES),
        "modes": modes,
        "status": "experimental_prediction" if reading_ready else "extraction_only",
        "blocker": None if reading_ready else "No trained speech_audio_v2 model exists for any recording mode.",
        "limitations": [
            "Reading mode only: trained on 37 participants (21 control, 16 PD) from the public MDVR-KCL reading recordings, segmented into 20 s windows.",
            "Sex metadata is unavailable for that cohort, so gender is not a model input.",
            "Recording device, language, passage and microphone differ from the app's own reading passage and devices; the model has not been tested on app recordings.",
            "Subject-disjoint 5-fold estimates have high variance (see fold SD). The result is an experimental research score, not a clinical validation.",
            "Sustained-vowel mode has no training data and remains unavailable.",
        ],
        "validation_protocol": "StratifiedGroupKFold by subject, 5 outer folds; feature selection, scaling, hyperparameter search and calibration inside training folds only.",
    }

    return {
        "disclaimer": "Research prototype. Results are experimental cross-validation estimates on small cohorts, not clinical validation.",
        "modalities": modalities,
        "fusion": {
            "method": fusion.get("method"),
            "validation_auc": fusion.get("validation_auc", {}),
            "weights": weights,
            "weights_sum": fusion.get("weights_sum"),
            "validation_fold": fusion.get("validation_fold"),
            "test_fold_labels_used": fusion.get("test_fold_labels_used"),
            "patient_level_performance": fusion.get("patient_level_performance"),
            "baseline_v1_weights_for_audit": baseline_weights,
            "baseline_v1_weights_note": "Derived from all outer-fold labels; kept only for audit and not used by the application.",
        },
        "speech_audio_v2": speech_v2,
        "architecture": list(ARCHITECTURE),
        "validation": dict(VALIDATION),
        "datasets_note": "Speech, handwriting and tremor come from independent cohorts. Row and participant counts are read from each modality's saved metrics file.",
    }
