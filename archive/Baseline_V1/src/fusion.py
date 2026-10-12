"""Decision-level fusion for independent modality probability estimates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS, REPORTS

MODALITIES = ("speech", "handwriting", "tremor")
FUSION_VERSION = "auc-normalized-late-fusion-v1"
DEFAULT_WEIGHTS = {name: 1.0 / len(MODALITIES) for name in MODALITIES}


def fuse_probabilities(
    speech_probability=None,
    handwriting_probability=None,
    tremor_probability=None,
    weights: Mapping[str, float] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Combine available modality probabilities and renormalize their weights.

    Probability arguments may be scalars or same-shaped numeric arrays. ``None``
    means that modality is unavailable. Returned probabilities/predictions are
    scalars for scalar inputs and NumPy arrays for array inputs.
    """
    supplied = {
        "speech": speech_probability,
        "handwriting": handwriting_probability,
        "tremor": tremor_probability,
    }
    available = [name for name in MODALITIES if supplied[name] is not None]
    if not available:
        raise ValueError("At least one modality probability must be provided")

    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number in [0, 1]")

    supplied_weights = DEFAULT_WEIGHTS if weights is None else weights
    if not isinstance(supplied_weights, Mapping):
        raise ValueError("weights must be a mapping from modality name to non-negative number")
    unknown = set(supplied_weights) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modality weight(s): {', '.join(sorted(unknown))}")

    checked_weights = {}
    for name in MODALITIES:
        value = supplied_weights.get(name, 0.0)
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Weight for {name} must be a finite non-negative number")
        checked_weights[name] = float(value)

    arrays = {}
    expected_shape = None
    for name in available:
        try:
            value = np.asarray(supplied[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Probability for {name} must be numeric") from exc
        if value.size == 0 or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"Probability for {name} must contain finite values in [0, 1]")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError("All available modality probabilities must have the same shape")
        arrays[name] = value

    total_weight = sum(checked_weights[name] for name in available)
    if total_weight <= 0:
        raise ValueError("Available modalities must have a positive total weight")
    effective = {name: checked_weights[name] / total_weight for name in available}
    probability = sum(effective[name] * arrays[name] for name in available)
    prediction = (probability >= float(threshold)).astype(int)

    if expected_shape == ():
        probability = float(probability)
        prediction = int(prediction)
    return {
        "probability": probability,
        "prediction": prediction,
        "available_modalities": available,
        "effective_weights": effective,
    }


def build_fusion_artifacts(
    artifacts_dir: Path = ARTIFACTS,
    reports_dir: Path = REPORTS,
) -> dict:
    """Derive reliability weights from existing modality OOF ensemble outputs.

    This function only reads saved OOF predictions; it does not fit models or
    pair records across the three independent cohorts.
    """
    rows = []
    for modality in MODALITIES:
        path = Path(artifacts_dir) / f"{modality}_oof_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required modality OOF predictions not found: {path}")
        frame = pd.read_csv(path)
        required = {"actual", "ensemble_probability"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f"{path} contains no usable OOF observations or has missing values")
        labels = frame["actual"].to_numpy()
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{path} must contain both classes to calculate ROC-AUC")
        probabilities = frame["ensemble_probability"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{path} contains ensemble probabilities outside [0, 1]")
        auc = float(roc_auc_score(labels, probabilities))
        rows.append({"modality": modality, "validation_ROC_AUC": auc, "raw_weight": auc})

    auc_total = sum(row["raw_weight"] for row in rows)
    if not np.isfinite(auc_total) or auc_total <= 0:
        raise ValueError("Cannot normalize modality weights: total ROC-AUC weight must be positive")
    for row in rows:
        row["normalized_weight"] = row["raw_weight"] / auc_total
    weights = {row["modality"]: row["normalized_weight"] for row in rows}
    if not np.isclose(sum(weights.values()), 1.0):
        raise RuntimeError("Normalized fusion weights do not sum to 1")

    weighting_method = "Each modality's nested group-aware OOF ensemble ROC-AUC, normalized by the sum of the three AUCs."
    config = {
        "methodology_version": FUSION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_ROC_AUC": {row["modality"]: row["validation_ROC_AUC"] for row in rows},
        "raw_weights": {row["modality"]: row["raw_weight"] for row in rows},
        "normalized_weights": weights,
        "weighting_metric": "nested subject-grouped out-of-fold ROC-AUC of each modality's existing SVM/RF ensemble_probability",
        "weighting_formula": "weight_i = ROC_AUC_i / sum(ROC_AUC_speech, ROC_AUC_handwriting, ROC_AUC_tremor)",
        "classification_threshold": 0.5,
        "oof_prediction_source": "models/artifacts/{speech,handwriting,tremor}_oof_predictions.csv; actual and ensemble_probability columns",
        "fusion_level": "decision-level probability fusion; modalities are not joined by participant",
        "experimental_fusion_metrics": None,
        "limitation": "The three modality cohorts are independent and unmatched. Weights use modality-specific OOF reliability; fused patient-level performance has not been measured.",
    }
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    results_dir = reports_dir / "model_results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "fusion_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame(rows, columns=["modality", "validation_ROC_AUC", "raw_weight", "normalized_weight"]).to_csv(
        results_dir / "fusion_weights.csv", index=False
    )
    summary = [
        "# Late-fusion configuration",
        "",
        "The modality probabilities are combined by weighted decision-level fusion:",
        "",
        "`P_final = w_speech * P_speech + w_handwriting * P_handwriting + w_tremor * P_tremor`",
        "",
        "Weights are each modality's nested group-aware OOF ensemble ROC-AUC divided by the sum of the three ROC-AUC values. The default classification threshold is 0.5. Missing modalities are omitted and the remaining weights are renormalized by `src.fusion.fuse_probabilities`.",
        "",
        "| Modality | OOF ROC-AUC | Raw weight | Normalized weight |",
        "|---|---:|---:|---:|",
    ]
    summary.extend(
        f"| {r['modality']} | {r['validation_ROC_AUC']:.6f} | {r['raw_weight']:.6f} | {r['normalized_weight']:.6f} |"
        for r in rows
    )
    summary += [
        "",
        "These cohorts are unmatched. This configuration does not create patient matches and does not establish fused patient-level accuracy, F1, or ROC-AUC. Telemonitoring motor-UPDRS regression remains separate.",
    ]
    (results_dir / "fusion_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return config
