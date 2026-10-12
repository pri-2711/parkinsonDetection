"""Verify Baseline V1 files and saved-model inference without refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent


def _check_manifest() -> int:
    manifest = json.loads((ROOT / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8-sig"))
    checked = 0
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest file is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)


def _classification_data(name: str):
    if name == "speech":
        frame = pd.read_csv(ROOT / "cleaned_dataset/speech_clean.csv")
        return _numeric(frame.drop(columns=["id", "recording", "status"], errors="ignore"))
    if name == "handwriting":
        frame = pd.read_csv(ROOT / "featureEngineering/handwriting_features.csv")
        return _numeric(frame.drop(columns=["participant_id", "test_id", "label"], errors="ignore"))
    frame = pd.read_csv(ROOT / "cleaned_dataset/tremors_clean.csv")
    columns = [c for c in frame if "motor_scale_specific_items_20_" in c or "motor_scale_specific_items_21_" in c]
    return _numeric(frame[columns])


def _verify_classification(name: str) -> dict:
    bundle_path = ROOT / "models" / "artifacts" / f"{name}_ensemble.joblib"
    bundle = joblib.load(bundle_path)
    if set(bundle["models"]) != {"svm", "random_forest"}:
        raise RuntimeError(f"Unexpected model bundle contents: {bundle_path}")
    features = _classification_data(name).loc[:, bundle["features"]].iloc[:3]
    for label, model in bundle["models"].items():
        probabilities = model.predict_proba(features)[:, 1]
        if probabilities.shape != (len(features),) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Saved {name}/{label} model failed inference")

    prediction_path = ROOT / "models" / "artifacts" / f"{name}_oof_predictions.csv"
    oof = pd.read_csv(prediction_path)
    results = json.loads((ROOT / "reports/model_results" / f"{name}_results.json").read_text(encoding="utf-8"))
    actual, probability = oof["actual"], oof["ensemble_probability"]
    predicted = (probability >= 0.5).astype(int)
    reproduced = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probability),
    }
    saved = results["outer_nested_cv_metrics"]["ensemble"]
    for metric, value in reproduced.items():
        if not np.isclose(value, saved[metric], rtol=0, atol=1e-12):
            raise RuntimeError(f"{name} OOF {metric} differs from saved result")
    return {"model_bundle_loaded": True, "inference_rows": len(features), "oof_rows": len(oof), "metrics_reproduced": True}


def _verify_regression() -> dict:
    bundle_path = ROOT / "models/artifacts/telemonitoring_regressors.joblib"
    bundle = joblib.load(bundle_path)
    frame = pd.read_csv(ROOT / "cleaned_dataset/telemonitoring_clean.csv")
    features = _numeric(frame.drop(columns=["subject_number", "motor_updrs", "total_updrs"], errors="ignore"))
    features = features.loc[:, bundle["features"]].iloc[:3]
    if set(bundle["models"]) != {"ridge", "random_forest"}:
        raise RuntimeError("Unexpected telemonitoring model bundle contents")
    for label, model in bundle["models"].items():
        prediction = model.predict(features)
        if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Saved telemonitoring/{label} model failed inference")

    oof = pd.read_csv(ROOT / "models/artifacts/telemonitoring_oof_predictions.csv")
    result = json.loads((ROOT / "reports/model_results/telemonitoring_results.json").read_text(encoding="utf-8"))
    actual = oof["actual"]
    for model_name in ("ridge", "random_forest", "group_mean_baseline"):
        prediction = oof[model_name]
        reproduced = {
            "mae": mean_absolute_error(actual, prediction),
            "rmse": np.sqrt(mean_squared_error(actual, prediction)),
            "r2": r2_score(actual, prediction),
        }
        for metric, value in reproduced.items():
            if not np.isclose(value, result["metrics"][model_name][metric], rtol=0, atol=1e-12):
                raise RuntimeError(f"Telemonitoring OOF {model_name}/{metric} differs from saved result")
    return {"model_bundle_loaded": True, "inference_rows": len(features), "oof_rows": len(oof), "metrics_reproduced": True}


def main() -> None:
    checks = {name: _verify_classification(name) for name in ("speech", "handwriting", "tremor")}
    checks["telemonitoring"] = _verify_regression()
    run_config = json.loads((ROOT / "models/artifacts/run_config.json").read_text(encoding="utf-8"))
    fusion_config = json.loads((ROOT / "models/artifacts/fusion_config.json").read_text(encoding="utf-8"))
    if run_config["seed"] != 42 or abs(sum(fusion_config["normalized_weights"].values()) - 1) > 1e-12:
        raise RuntimeError("Saved random seed or fusion weights failed validation")
    print(json.dumps({"manifest_files_verified": _check_manifest(), "checks": checks,
                      "random_seed": run_config["seed"], "fusion_weights_sum_to_one": True,
                      "model_refitting_performed": False}, indent=2))


if __name__ == "__main__":
    main()
