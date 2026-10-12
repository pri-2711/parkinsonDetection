"""Nested, participant-aware SVM/RF tuning isolated from Baseline V1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / "archive" / "Baseline_V1"
RESULTS = HERE / "results"
CONFIG_PATH = HERE / "experiment_config.json"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
CALIBRATION_FOLDS = 3
WEIGHTS = [round(i / 10, 1) for i in range(11)]
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _read_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _numeric_features(frame, excluded):
    x = frame.drop(columns=excluded, errors="ignore").copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if x.shape[1] == 0:
        raise ValueError("No numeric predictors remain")
    return x


def _duplicate_safe_groups(x, participant_ids):
    ids = np.asarray(participant_ids, dtype=str)
    parent = {p: p for p in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    frame = pd.DataFrame({"signature": pd.util.hash_pandas_object(x, index=False).to_numpy(), "participant": ids})
    for members in frame.groupby("signature").participant.unique():
        for member in members[1:]:
            left, right = find(members[0]), find(member)
            if left != right:
                parent[right] = left
    return np.asarray([find(p) for p in ids])


def load_modality(name):
    if name == "speech":
        raw = pd.read_csv(BASELINE / "cleaned_dataset" / "speech_clean.csv")
        x = _numeric_features(raw, ["status", "id", "recording"])
        y = pd.to_numeric(raw["status"], errors="raise").astype(int).reset_index(drop=True)
        groups = raw["id"].astype(str).to_numpy()
    elif name == "handwriting":
        raw = pd.read_csv(BASELINE / "featureEngineering" / "handwriting_features.csv")
        x = _numeric_features(raw, ["label", "participant_id", "test_id"])
        y = pd.to_numeric(raw["label"], errors="raise").astype(int).reset_index(drop=True)
        groups = _duplicate_safe_groups(x, raw["participant_id"])
    elif name == "tremor":
        raw = pd.read_csv(BASELINE / "cleaned_dataset" / "tremors_clean.csv")
        if "participant_code" not in raw:
            raise ValueError("Tremor input lacks participant_code")
        y = raw["participant_code"].astype(str).str.upper().str.startswith("PD").astype(int).reset_index(drop=True)
        cols = [c for c in raw if "motor_scale_specific_items_20_" in c or "motor_scale_specific_items_21_" in c]
        if not cols:
            raise ValueError("No UPDRS items 20/21 found")
        x = _numeric_features(raw[cols], [])
        groups = raw["participant_code"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown modality: {name}")
    if len(x) != len(y) or pd.isna(groups).any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"Invalid rows, groups, or binary labels for {name}")
    return x.reset_index(drop=True), y, np.asarray(groups), list(x.columns)


def group_splits(x, y, groups, folds):
    if pd.Series(groups).nunique() < folds:
        raise ValueError(f"Need at least {folds} independent groups")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(splitter.split(x, y, groups))
    for train, valid in splits:
        if set(np.asarray(groups)[train]) & set(np.asarray(groups)[valid]):
            raise RuntimeError("Participant leakage across a CV split")
        if np.unique(np.asarray(y)[valid]).size != 2:
            raise ValueError("A validation fold has only one class")
    return splits


def _model_and_grid(family, cfg):
    if family == "svm":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf")),
        ])
        grid = {f"model__{k}": v for k, v in cfg["models"]["svm"]["search_space"].items()}
        return estimator, grid
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])
    grid = {f"model__{k}": v for k, v in cfg["models"]["random_forest"]["search_space"].items()}
    return estimator, grid


def _search(family, x, y, groups, cfg):
    estimator, grid = _model_and_grid(family, cfg)
    splits = group_splits(x, y, groups, INNER_FOLDS)
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=splits, refit=True, n_jobs=1,
                          error_score="raise", return_train_score=False)
    search.fit(x, y)
    return search


def _calibrate(estimator, x, y, groups):
    splits = group_splits(x, y, groups, min(CALIBRATION_FOLDS, pd.Series(groups).nunique()))
    try:
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=splits, ensemble=True)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=splits, ensemble=True)
    return calibrated.fit(x, y)


def _metrics(y, probabilities):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def _baseline_oof(name, y):
    frame = pd.read_csv(BASELINE / "models" / "artifacts" / f"{name}_oof_predictions.csv")
    if len(frame) != len(y) or not np.array_equal(frame["actual"].to_numpy(), y.to_numpy()):
        raise ValueError(f"Archived {name} OOF predictions do not match archive labels and row order")
    for col in ("svm_probability", "random_forest_probability"):
        if col not in frame or not np.isfinite(frame[col]).all():
            raise ValueError(f"Archived OOF predictions missing valid {col}")
    return frame


def reported_test_indices(fold_assignments, excluded_folds=(1,)):
    """Rows eligible for model comparison after reserving validation folds."""
    folds = np.asarray(fold_assignments)
    return np.flatnonzero(~np.isin(folds, np.asarray(excluded_folds)))


def _baseline_fusion_validation(name, x, y, groups, validation_indices):
    """Use only reserved outer-fold 1 validation labels to derive audit weights."""
    train = np.setdiff1d(np.arange(len(y)), validation_indices)
    tx, ty, tg = x.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True), groups[train]
    # Preserve Baseline V1's fixed classifiers and inner-OOF blend-selection rule.
    inner = group_splits(tx, ty, tg, INNER_FOLDS)
    inner_prob = {k: np.full(len(ty), np.nan) for k in ("svm", "random_forest")}
    for tr, va in inner:
        for family in inner_prob:
            estimator, _ = _model_and_grid(family, _read_config())
            # Baseline V1 fixed parameters, not experiment search parameters.
            if family == "svm":
                estimator.set_params(model__C=1.0, model__gamma="scale", model__class_weight=None)
            else:
                estimator.set_params(model__n_estimators=300, model__max_depth=8, model__min_samples_leaf=3,
                                     model__max_features="sqrt", model__class_weight="balanced")
            model = _calibrate(estimator, tx.iloc[tr], ty.iloc[tr].to_numpy(), tg[tr])
            inner_prob[family][va] = model.predict_proba(tx.iloc[va])[:, 1]
    candidates = []
    for w in WEIGHTS:
        p = w * inner_prob["svm"] + (1 - w) * inner_prob["random_forest"]
        m = _metrics(ty, p)
        m["weight"] = w
        m["selection_score"] = (m["roc_auc"] + m["f1"]) / 2
        candidates.append(m)
    selected = max(candidates, key=lambda row: (row["selection_score"], row["recall"], -abs(row["weight"] - 0.5)))
    probabilities = {}
    for family in ("svm", "random_forest"):
        estimator, _ = _model_and_grid(family, _read_config())
        if family == "svm":
            estimator.set_params(model__C=1.0, model__gamma="scale", model__class_weight=None)
        else:
            estimator.set_params(model__n_estimators=300, model__max_depth=8, model__min_samples_leaf=3,
                                 model__max_features="sqrt", model__class_weight="balanced")
        fitted = _calibrate(estimator, tx, ty.to_numpy(), tg)
        probabilities[family] = fitted.predict_proba(x.iloc[validation_indices])[:, 1]
    validation_y = y.iloc[validation_indices].to_numpy()
    blend = selected["weight"] * probabilities["svm"] + (1 - selected["weight"]) * probabilities["random_forest"]
    return {
        "n_validation_rows": int(len(validation_indices)),
        "validation_auc": float(roc_auc_score(validation_y, blend)),
        "selected_inner_oof_svm_weight": float(selected["weight"]),
        "selected_inner_oof_rf_weight": float(1 - selected["weight"]),
        "outer_test_metric_used": False,
    }


def _evaluate_modality(name, x, y, groups, features, cfg):
    outer = group_splits(x, y, groups, OUTER_FOLDS)
    baseline = _baseline_oof(name, y)
    # Fold 1 is reserved solely to estimate the separate, experiment-local
    # validation weights. Its labels are excluded from all reported test metrics.
    fusion_validation_indices = outer[0][1]
    fusion_validation = _baseline_fusion_validation(name, x, y, groups, fusion_validation_indices)
    fold_assignments = np.full(len(y), -1, dtype=int)
    for fold, (_, test) in enumerate(outer, start=1):
        fold_assignments[test] = fold
    if np.any(fold_assignments < 1):
        raise RuntimeError("Outer folds did not assign every row")

    predictions = {"actual": y.to_numpy(), "outer_fold": fold_assignments,
                   "baseline_svm_probability": baseline["svm_probability"].to_numpy(),
                   "baseline_random_forest_probability": baseline["random_forest_probability"].to_numpy()}
    fold_detail = []
    outer_best_params = {"svm": [], "random_forest": []}
    selected_probability = np.full(len(y), np.nan)
    selected_family_by_fold = {}
    for fold, (train, test) in enumerate(outer, start=1):
        print(f"[{name}] outer fold {fold}/{OUTER_FOLDS}: tuning SVM", flush=True)
        tx, ty, tg = x.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True), groups[train]
        searches = {}
        fold_probabilities = {}
        for family in ("svm", "random_forest"):
            if family == "random_forest":
                print(f"[{name}] outer fold {fold}/{OUTER_FOLDS}: tuning Random Forest", flush=True)
            search = _search(family, tx, ty, tg, cfg)
            searches[family] = search
            outer_best_params[family].append({"fold": fold, "best_params": search.best_params_,
                                              "inner_best_roc_auc": float(search.best_score_)})
            model = _calibrate(search.best_estimator_, tx, ty.to_numpy(), tg)
            fold_probabilities[family] = model.predict_proba(x.iloc[test])[:, 1]
        for family, key in (("svm", "svm"), ("random_forest", "random_forest")):
            predictions[f"tuned_{key}_probability"] = predictions.get(f"tuned_{key}_probability", np.full(len(y), np.nan))
            predictions[f"tuned_{key}_probability"][test] = fold_probabilities[family]
        selected = max(searches, key=lambda family: searches[family].best_score_)
        selected_family_by_fold[fold] = selected
        selected_probability[test] = fold_probabilities[selected]
        if fold != 1:
            y_test = y.iloc[test].to_numpy()
            fold_record = {"fold": fold, "rows": int(len(test)), "selected_family": selected,
                           "svm_inner_best_roc_auc": float(searches["svm"].best_score_),
                           "random_forest_inner_best_roc_auc": float(searches["random_forest"].best_score_)}
            for family, key in (("svm", "svm"), ("random_forest", "random_forest")):
                fold_record[f"baseline_{key}"] = _metrics(y_test, baseline[f"{key}_probability"].to_numpy()[test])
                fold_record[f"tuned_{key}"] = _metrics(y_test, fold_probabilities[family])
            fold_record["nested_selected"] = _metrics(y_test, fold_probabilities[selected])
            fold_detail.append(fold_record)
        print(f"[{name}] outer fold {fold}: inner-selected model={selected}; reporting metric={'no (fusion validation fold)' if fold == 1 else 'yes'}", flush=True)

    scored = reported_test_indices(fold_assignments)
    aggregate = {}
    for family in ("svm", "random_forest"):
        key = family
        aggregate[f"baseline_{key}"] = _metrics(y.to_numpy()[scored], baseline[f"{key}_probability"].to_numpy()[scored])
        aggregate[f"tuned_{key}"] = _metrics(y.to_numpy()[scored], predictions[f"tuned_{key}_probability"][scored])
    aggregate["nested_selected"] = _metrics(y.to_numpy()[scored], selected_probability[scored])
    aggregate["reported_rows"] = int(len(scored))
    aggregate["excluded_fusion_validation_rows"] = int(len(fusion_validation_indices))

    # Full-data grouped CV chooses deployment hyperparameters and family.
    print(f"[{name}] selecting full-data deployment models", flush=True)
    final_searches = {}
    for family in ("svm", "random_forest"):
        final_searches[family] = _search(family, x, y, groups, cfg)
    deployment_family = max(final_searches, key=lambda family: final_searches[family].best_score_)
    deployment = {}
    for family, search in final_searches.items():
        calibrated = _calibrate(search.best_estimator_, x, y.to_numpy(), groups)
        bundle = {"modality": name, "family": family, "model": calibrated, "features": features,
                  "best_params": search.best_params_, "inner_cv_best_roc_auc": float(search.best_score_),
                  "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
                  "calibration_folds": CALIBRATION_FOLDS}
        target = RESULTS / "artifacts" / f"{name}_{family}_tuned.joblib"
        joblib.dump(bundle, target)
        reloaded = joblib.load(target)
        sample = x.iloc[:min(3, len(x))].loc[:, features]
        smoke = reloaded["model"].predict_proba(sample)[:, 1]
        if not np.isfinite(smoke).all() or ((smoke < 0) | (smoke > 1)).any():
            raise RuntimeError(f"Inference smoke check failed for {target}")
        deployment[family] = {"best_params": search.best_params_, "inner_cv_best_roc_auc": float(search.best_score_),
                              "artifact": str(target.relative_to(HERE)), "inference_smoke_rows": int(len(sample))}
    result = {
        "modality": name, "row_count": int(len(y)), "feature_count": int(len(features)),
        "participant_groups": int(pd.Series(groups).nunique()), "metrics": aggregate,
        "outer_fold_metrics": fold_detail, "outer_fold_best_params": outer_best_params,
        "selected_family_by_outer_fold": selected_family_by_fold,
        "deployment_selection_metric": "highest mean inner group-CV ROC-AUC on the full available cohort",
        "deployment_family": deployment_family, "deployment_artifacts": deployment,
        "fusion_weight_validation": fusion_validation,
        "fusion_validation_fold": 1,
        "reported_test_folds": [2, 3, 4, 5],
        "baseline_source": f"archive/Baseline_V1/models/artifacts/{name}_oof_predictions.csv",
    }
    prediction_path = RESULTS / "predictions" / f"{name}_oof_comparison.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions["nested_selected_probability"] = selected_probability
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    return result


def _write_reports(results, fusion):
    baseline_fusion = json.loads((BASELINE / "models" / "artifacts" / "fusion_config.json").read_text(encoding="utf-8"))
    report = [
        "# Regularized SVM and Random Forest classification experiment", "",
        "This experiment is isolated under `experiments/regularized_svm_rf`; Baseline V1 remains archived unchanged.", "",
        "## Protocol", "",
        "The archived cleaned inputs/features and archived baseline OOF predictions were used. Five-fold `StratifiedGroupKFold` (seed 42) defines the same outer participant partitions as Baseline V1. Each outer training partition uses four-fold grouped inner grid search by ROC-AUC; sigmoid calibration uses three grouped folds after hyperparameter selection. Median imputation and SVM scaling are in the estimator pipeline and refit inside every inner/calibration training fold. Thresholded metrics use 0.5.", "",
        "Fold 1 in each modality is reserved for validation-only AUC-based fusion-weight estimation. It is excluded from the reported baseline/tuned classification metrics; comparisons therefore use the same four outer test folds (folds 2–5). The fusion-validation fold is not an independent multimodal patient cohort.", "",
        "## Baseline V1 vs tuned results", "",
        "Baseline probabilities come from the archived OOF file; tuned probabilities were generated on the identical row assignments and outer group folds. The nested-selected row chooses SVM or RF within each training fold using inner-CV ROC-AUC.", "",
        "| Modality | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for modality, result in results.items():
        for model_key, label in (("baseline_svm", "Baseline SVM"), ("tuned_svm", "Tuned SVM"),
                                 ("baseline_random_forest", "Baseline RF"), ("tuned_random_forest", "Tuned RF"),
                                 ("nested_selected", "Nested selected SVM/RF")):
            m = result["metrics"][model_key]
            report.append(f"| {modality} | {label} | " + " | ".join(f"{m[k]:.4f}" for k in METRICS) + " |")
    report += ["", "## Fold-level variability", "", "Values below use folds 2–5 only (mean ± sample standard deviation). Fold 1 is reserved for fusion validation.", "",
               "| Modality | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---:|---:|---:|---:|---:|"]
    for modality, result in results.items():
        for metric_key, label in (("baseline_svm", "Baseline SVM"), ("tuned_svm", "Tuned SVM"),
                                  ("baseline_random_forest", "Baseline RF"), ("tuned_random_forest", "Tuned RF"),
                                  ("nested_selected", "Nested selected SVM/RF")):
            per_fold = []
            for fold in result["outer_fold_metrics"]:
                if metric_key.startswith("baseline_"):
                    m = fold[metric_key]
                elif metric_key.startswith("tuned_"):
                    m = fold[metric_key]
                else:
                    m = fold["nested_selected"]
                per_fold.append(m)
            cells = []
            for metric in METRICS:
                values = np.asarray([f[metric] for f in per_fold], dtype=float)
                cells.append(f"{values.mean():.4f} ± {values.std(ddof=1):.4f}")
            report.append(f"| {modality} | {label} | " + " | ".join(cells) + " |")
    report += ["", "## Selected deployment models and parameters", ""]
    for modality, result in results.items():
        report.append(f"- **{modality}: {result['deployment_family']}**. Full-cohort grouped-CV ROC-AUC: "
                      f"{result['deployment_artifacts'][result['deployment_family']]['inner_cv_best_roc_auc']:.4f}. "
                      f"Parameters: `{result['deployment_artifacts'][result['deployment_family']]['best_params']}`.")
    report += ["", "## Fusion validation", "",
               "The existing AUC-normalized late-fusion function preserves probability averaging and renormalizes weights when modalities are missing. Its Baseline V1 weights were derived from all outer OOF `actual` labels and therefore do not meet the stricter requirement that evaluation-fold labels not determine weights. Baseline V1 is preserved unchanged. The isolated experiment estimates AUC-normalized weights from fold 1 only, which was excluded from reported per-modality test metrics; weights are recorded in `fusion_validation_weights.json`. This is a validation-only modality reliability check, not patient-level multimodal validation.", "",
               "The Baseline V1 AUC-normalized weights are reproduced below for audit. Those AUCs came from the complete modality OOF files, including every outer held-out fold, so their labels were part of the fusion-weight calculation. The experiment-local weights use only reserved fold 1; folds 2–5 do not influence those weights.", "",
               "| Modality | Baseline V1 outer-OOF AUC | Baseline V1 normalized weight | Reserved validation AUC | Validation-only normalized weight |", "|---|---:|---:|---:|---:|"]
    for name, auc in fusion["validation_auc"].items():
        old_auc = baseline_fusion["validation_ROC_AUC"][name]
        old_weight = baseline_fusion["normalized_weights"][name]
        report.append(f"| {name} | {old_auc:.6f} | {old_weight:.6f} | {auc:.6f} | {fusion['normalized_weights'][name]:.6f} |")
    report += ["", "## External validation", "",
               "No suitable independent local test cohort was identified. The extra handwriting directory contains Parkinson-only files and lacks controls in the engineered tabular feature schema; the spiral image directories have no aligned participant labels/features; speech and tremor each contain only the cohorts represented in the primary inputs. External validation remains pending. No portal or external data source was searched.", "",
               "## Reproduction", "", "From the repository root:", "", "```powershell", "python experiments/regularized_svm_rf/run_experiments.py", "python -m unittest discover -s tests -v", "python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v", "python archive/Baseline_V1/verify_archive.py", "```", "",
               "The run writes only below `experiments/regularized_svm_rf/results`. The archived model/data/config and manifest are read-only inputs. See `run_metadata.json` for package versions, seeds, and data hashes.", "",
               "## Limitations", "",
               "The first fold is reserved, so reported classification metrics cover four outer folds rather than the full five-fold estimate. Fold-to-fold variability is high for small cohorts. Full-cohort deployment hyperparameters are selected via grouped CV and then calibrated on the full cohort; deployment artifacts are for inference, not an unbiased performance estimate. No patient-level fusion performance is claimed."]
    report_path = RESULTS / "TUNING_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    fusion_path = RESULTS / "fusion_validation_weights.json"
    fusion_path.write_text(json.dumps(fusion, indent=2), encoding="utf-8")


def main():
    cfg = _read_config()
    (RESULTS / "artifacts").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    modalities = list(cfg["classification_datasets"])
    results = {}
    fusion_auc = {}
    for modality in modalities:
        x, y, groups, features = load_modality(modality)
        print(f"[{modality}] rows={len(y)}, features={len(features)}, subject groups={pd.Series(groups).nunique()}", flush=True)
        result = _evaluate_modality(modality, x, y, groups, features, cfg)
        results[modality] = result
        fusion_auc[modality] = result["fusion_weight_validation"]["validation_auc"]
        (RESULTS / f"{modality}_metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    total = sum(fusion_auc.values())
    fusion = {"method": "existing AUC-normalized late-fusion formula; reliability AUC from reserved group validation fold only",
              "validation_auc": fusion_auc,
              "normalized_weights": {name: value / total for name, value in fusion_auc.items()},
              "weights_sum": total / total,
              "test_fold_labels_used": False,
              "test_fold_metrics_used": False,
              "baseline_v1_outer_oof_auc": json.loads((BASELINE / "models" / "artifacts" / "fusion_config.json").read_text(encoding="utf-8"))["validation_ROC_AUC"],
              "baseline_v1_outer_oof_weights": json.loads((BASELINE / "models" / "artifacts" / "fusion_config.json").read_text(encoding="utf-8"))["normalized_weights"],
              "baseline_v1_oof_weights_sum": sum(json.loads((BASELINE / "models" / "artifacts" / "fusion_config.json").read_text(encoding="utf-8"))["normalized_weights"].values()),
              "baseline_v1_oof_labels_were_used_for_weights": True,
              "missing_modalities": "src.fusion.fuse_probabilities omits unavailable modalities and renormalizes remaining weights",
              "validation_fold": 1,
              "patient_level_fusion_performance": None}
    _write_reports(results, fusion)
    data_hashes = {}
    for modality in modalities:
        data_path = BASELINE / "models" / "artifacts" / f"{modality}_oof_predictions.csv"
        data_hashes[str(data_path.relative_to(ROOT))] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    try:
        import sklearn
        import pandas
        import numpy
        versions = {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                    "pandas": pandas.__version__, "numpy": numpy.__version__, "joblib": joblib.__version__}
    except Exception as exc:
        versions = {"version_read_error": str(exc)}
    metadata = {"config": cfg, "seed": SEED, "versions": versions, "baseline_root": str(BASELINE),
                "baseline_manifest": "archive/Baseline_V1/ARCHIVE_MANIFEST.json", "baseline_oof_sha256": data_hashes,
                "outer_test_folds_scored": [2, 3, 4, 5], "fusion_validation_fold": 1}
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Report: {RESULTS / 'TUNING_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
