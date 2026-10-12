from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (ARTIFACTS, CALIBRATION_FOLDS, CLASSIFICATION_DATASETS,
                        FOLDS, REGRESSION_DATASET, REPORTS, ROOT, SEED, WEIGHTS)


def _numeric_features(frame: pd.DataFrame, excluded: list[str]) -> pd.DataFrame:
    x = frame.drop(columns=excluded, errors="ignore").copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.select_dtypes(include=[np.number])
    if x.shape[1] == 0:
        raise ValueError("No numeric predictors remain after excluding identifiers and targets")
    return x.replace([np.inf, -np.inf], np.nan)


def _class_models():
    common = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    svm = Pipeline(common + [("scale", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf"))])
    rf = Pipeline(common + [("model", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1))])
    return {"svm": svm, "random_forest": rf}


def _group_splits(x, y, groups, folds=FOLDS):
    n_groups = pd.Series(groups).nunique()
    if n_groups < folds:
        raise ValueError(f"Need at least {folds} independent groups; found {n_groups}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(splitter.split(x, y, groups))
    for _, test in splits:
        if len(np.unique(np.asarray(y)[test])) < 2:
            raise ValueError("A group test fold contains one class; reduce folds or revise cohort design")
    return splits


def _calibrator(estimator, x, y, groups):
    # Calibration folds are participant-disjoint within the estimator's training data.
    n = min(CALIBRATION_FOLDS, pd.Series(groups).nunique())
    if n < 2:
        raise ValueError("At least two training participants are needed for calibration")
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    calibration_splits = list(splitter.split(x, y, groups))
    if any(len(np.unique(np.asarray(y)[valid])) < 2 for _, valid in calibration_splits):
        raise ValueError("Calibration fold lacks a class; calibration cannot be estimated safely")
    try:
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=calibration_splits, ensemble=True)
    except TypeError:  # sklearn before estimator= rename
        calibrated = CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=calibration_splits, ensemble=True)
    return calibrated.fit(x, y)


def _oof_class(x, y, groups, folds=FOLDS):
    splits = _group_splits(x, y, groups, folds)
    output = {name: np.full(len(y), np.nan) for name in _class_models()}
    for train, valid in splits:
        train_groups = np.asarray(groups)[train]
        for name, estimator in _class_models().items():
            model = _calibrator(estimator, x.iloc[train], y.iloc[train].to_numpy(), train_groups)
            output[name][valid] = model.predict_proba(x.iloc[valid])[:, 1]
    if any(np.isnan(p).any() for p in output.values()):
        raise RuntimeError("Failed to generate a complete out-of-fold probability vector")
    return output


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p))}


def _choose_weight(y, svm_p, rf_p):
    rows = []
    for w in WEIGHTS:
        m = _metrics(y, w * svm_p + (1 - w) * rf_p)
        m.update(svm_weight=w, rf_weight=round(1 - w, 1), selection_score=(m["roc_auc"] + m["f1"]) / 2)
        rows.append(m)
    # Predeclared tie-breaking: score, recall, then closer to equal blend.
    best = max(rows, key=lambda r: (r["selection_score"], r["recall"], -abs(r["svm_weight"] - 0.5)))
    return best, rows


def _nested_classification(name, x, y, groups):
    outer = _group_splits(x, y, groups)
    pred = {"svm": np.full(len(y), np.nan), "random_forest": np.full(len(y), np.nan), "ensemble": np.full(len(y), np.nan)}
    outer_weights = []
    for fold, (train, test) in enumerate(outer, start=1):
        inner_x, inner_y, inner_groups = x.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True), np.asarray(groups)[train]
        inner_oof = _oof_class(inner_x, inner_y, inner_groups, folds=4)
        selected, _ = _choose_weight(inner_y.to_numpy(), inner_oof["svm"], inner_oof["random_forest"])
        outer_weights.append({"outer_fold": fold, "svm_weight": selected["svm_weight"], "rf_weight": selected["rf_weight"]})
        for model_name, estimator in _class_models().items():
            fitted = _calibrator(estimator, x.iloc[train], y.iloc[train].to_numpy(), np.asarray(groups)[train])
            pred[model_name][test] = fitted.predict_proba(x.iloc[test])[:, 1]
        pred["ensemble"][test] = selected["svm_weight"] * pred["svm"][test] + selected["rf_weight"] * pred["random_forest"][test]
    final_oof = _oof_class(x.reset_index(drop=True), y.reset_index(drop=True), np.asarray(groups), folds=FOLDS)
    final_weight, weight_table = _choose_weight(y.to_numpy(), final_oof["svm"], final_oof["random_forest"])
    scores = {name: _metrics(y, p) for name, p in pred.items()}
    # Store fully trained calibrated base estimators and selected ensemble weights.
    fitted = {n: _calibrator(e, x, y.to_numpy(), np.asarray(groups)) for n, e in _class_models().items()}
    joblib.dump({"models": fitted, "weights": {"svm": final_weight["svm_weight"], "random_forest": final_weight["rf_weight"]},
                 "features": list(x.columns), "positive_label": 1}, ARTIFACTS / f"{name}_ensemble.joblib")
    return scores, final_weight, weight_table, outer_weights, pred


def _profile(name, raw, x, y, groups, task):
    d = {"dataset": name, "task": task, "rows": len(raw), "features": x.shape[1], "groups": int(pd.Series(groups).nunique()),
         "missing_cells_predictors": int(x.isna().sum().sum()), "duplicate_rows": int(raw.duplicated().sum()),
         "target_summary": y.value_counts(dropna=False).to_dict() if task == "classification" else {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max())}}
    REPORTS.joinpath("data_profile", f"{name}.json").write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    x.describe().T.to_csv(REPORTS / "feature_analysis" / f"{name}_features.csv")
    x.corr(numeric_only=True).to_csv(REPORTS / "feature_analysis" / f"{name}_correlations.csv")
    return d


def _duplicate_safe_groups(x: pd.DataFrame, participant_ids) -> np.ndarray:
    """Keep participants with identical feature signatures in the same CV fold."""
    ids = np.asarray(participant_ids, dtype=str)
    parent = {participant: participant for participant in ids}
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
    return np.asarray([find(participant) for participant in ids])


def _run_classification(name, path, target_col, group_col):
    raw = pd.read_csv(path)
    if name == "tremor":
        if "participant_code" not in raw:
            raise ValueError("Tremor input lacks participant_code required for subject grouping")
        raw["label"] = raw["participant_code"].astype(str).str.upper().str.startswith("PD").astype(int)
        target_col = "label"
        selected = [c for c in raw if "motor_scale_specific_items_20_" in c or "motor_scale_specific_items_21_" in c]
        if not selected:
            raise ValueError("No UPDRS tremor items (20/21) found")
        x = _numeric_features(raw[selected], [])
    else:
        excluded = [target_col, group_col]
        if name == "speech": excluded.append("recording")
        if name == "handwriting": excluded.append("test_id")
        x = _numeric_features(raw, excluded)
    y = pd.to_numeric(raw[target_col], errors="raise").astype(int)
    participants = raw[group_col].astype(str)
    groups = _duplicate_safe_groups(x, participants) if name == "handwriting" else participants.to_numpy()
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{name}: expected binary target encoded as 0/1")
    if pd.isna(groups).any(): raise ValueError(f"{name}: missing subject grouping identifiers")
    profile = _profile(name, raw, x, y, groups, "classification")
    profile["participants"] = int(participants.nunique())
    profile["split_groups"] = int(pd.Series(groups).nunique())
    REPORTS.joinpath("data_profile", f"{name}.json").write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    scores, selected, weight_rows, fold_weights, oof = _nested_classification(name, x, y, groups)
    pd.DataFrame(weight_rows).to_csv(REPORTS / "model_results" / f"{name}_weight_comparison.csv", index=False)
    pd.DataFrame(fold_weights).to_csv(REPORTS / "model_results" / f"{name}_outer_fold_weights.csv", index=False)
    pd.DataFrame({"actual": y, **{f"{k}_probability": v for k, v in oof.items()}}).to_csv(ARTIFACTS / f"{name}_oof_predictions.csv", index=False)
    result = {"dataset": name, "profile": profile, "outer_nested_cv_metrics": scores,
              "final_weight_selection_from_full_cohort_oof": selected}
    (REPORTS / "model_results" / f"{name}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name}: nested subject-disjoint CV; SVM {scores['svm']}; RF {scores['random_forest']}; ensemble {scores['ensemble']}; selected weights SVM={selected['svm_weight']}, RF={selected['rf_weight']}")
    return result


def _run_regression():
    path, target_col, group_col = REGRESSION_DATASET
    raw = pd.read_csv(path)
    x = _numeric_features(raw, [target_col, "total_updrs", group_col])
    y = pd.to_numeric(raw[target_col], errors="raise").astype(float)
    groups = raw[group_col].astype(str).to_numpy()
    _profile("telemonitoring", raw, x, y, groups, "regression")
    predictions = {k: np.full(len(y), np.nan) for k in ("ridge", "random_forest", "group_mean_baseline")}
    split = GroupKFold(n_splits=FOLDS)
    for train, test in split.split(x, y, groups):
        for key, estimator in {
            "ridge": Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
            "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, max_features=0.8, random_state=SEED, n_jobs=-1))]),
        }.items():
            estimator.fit(x.iloc[train], y.iloc[train])
            predictions[key][test] = estimator.predict(x.iloc[test])
        predictions["group_mean_baseline"][test] = y.iloc[train].mean()
    scores = {name: {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))} for name, p in predictions.items()}
    final_models = {
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, max_features=0.8, random_state=SEED, n_jobs=-1))])}
    for model in final_models.values(): model.fit(x, y)
    joblib.dump({"models": final_models, "features": list(x.columns), "target": target_col}, ARTIFACTS / "telemonitoring_regressors.joblib")
    pd.DataFrame({"actual": y, **predictions}).to_csv(ARTIFACTS / "telemonitoring_oof_predictions.csv", index=False)
    result = {"metrics": scores, "target": target_col, "features": list(x.columns), "group_count": int(pd.Series(groups).nunique()),
              "interpretation": "Negative R2 indicates predictions were worse than predicting each test fold's mean; this is not currently a useful unseen-subject severity model."}
    (REPORTS / "model_results" / "telemonitoring_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"telemonitoring regression grouped CV: {scores}")
    return result


def _handwriting_audit():
    raw = pd.read_csv(ROOT / "cleaned_dataset/handwriting_clean.csv")
    features = pd.read_csv(ROOT / "featureEngineering/handwriting_features.csv")
    per_subject = features.groupby("participant_id").agg(rows=("label", "size"), labels=("label", "nunique"))
    signature_frame = features.drop(columns=["participant_id", "test_id", "label"])
    signatures = pd.util.hash_pandas_object(signature_frame, index=False)
    duplicate_members = []
    for _, block in features.assign(_signature=signatures).groupby("_signature"):
        people = sorted(block.participant_id.unique().tolist())
        if len(people) > 1:
            duplicate_members.append({"participants": people, "rows": int(len(block)), "labels": sorted(block.label.unique().tolist())})
    audits = {
        "trajectory_rows": len(raw), "participants": int(raw.participant_id.nunique()),
        "participant_test_rows": len(features), "duplicate_feature_rows": int(features.drop(columns=["participant_id", "test_id"]).duplicated().sum()),
        "cross_participant_exact_feature_signature_clusters": duplicate_members,
        "split_grouping": "Participants sharing an exact engineered feature signature are kept in one fold.",
        "participants_with_inconsistent_labels": int((per_subject.labels > 1).sum()),
        "tests_per_participant": features.groupby("participant_id").size().value_counts().sort_index().to_dict(),
        "class_counts_by_participant": features.drop_duplicates("participant_id").label.value_counts().sort_index().to_dict(),
        "trajectory_order_preserved": bool((raw.groupby(["participant_id", "test_id"]).point_index.apply(lambda s: s.is_monotonic_increasing)).all()),
        "participant_identifier_excluded_from_model_features": True,
        "test_identifier_excluded_from_model_features": True,
        "folds_grouped_by_participant_id_and_exact_duplicate_component": True,
        "known_limits": ["Only 40 participants; participant-level test rows within a fold are correlated.", "High scores need external participant-level validation."]}
    (REPORTS / "leakage_audit" / "handwriting_audit.json").write_text(json.dumps(audits, indent=2, default=str), encoding="utf-8")
    return audits


def run_all():
    for folder in ("data_profile", "feature_analysis", "leakage_audit", "model_results"):
        (REPORTS / folder).mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run_config.json").write_text(json.dumps({"seed": SEED, "outer_folds": FOLDS, "calibration_folds": CALIBRATION_FOLDS,
        "ensemble_weights": WEIGHTS, "selection_score": "mean(ROC-AUC, F1 at 0.5); ties by recall then closeness to equal weights",
        "classification_models": ["calibrated RBF SVM", "calibrated Random Forest"], "final_fusion": "not trained: no matched cross-modality cohort"}, indent=2), encoding="utf-8")
    # Regenerate deterministic cleaned tables/features without modifying raw data.
    from cleaningPipeline.speech_cleaning import clean_speech
    from cleaningPipeline.telemonitoring_cleaning import clean_telemonitoring
    from cleaningPipeline.tremor_cleaning import clean_tremors
    from cleaningPipeline.handwriting_cleaning import clean_handwriting
    from featureEngineering.handwriting_features import build_handwriting_features
    clean_tremors(); clean_speech(); clean_handwriting(); clean_telemonitoring(); build_handwriting_features()
    audit = _handwriting_audit()
    results = {name: _run_classification(name, *spec) for name, spec in CLASSIFICATION_DATASETS.items()}
    results["telemonitoring"] = _run_regression()
    (REPORTS / "leakage_audit" / "handwriting_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    from src.fusion import build_fusion_artifacts
    fusion_config = build_fusion_artifacts()
    report = ["# Pipeline results", "", "All classification estimates use nested five-fold subject-disjoint cross-validation. SVM/RF probabilities are sigmoid-calibrated using group-disjoint folds. Ensemble weights are chosen within each outer training partition; reported ensemble scores use outer held-out folds. The deployable ensemble weight is selected from full-cohort grouped OOF predictions and is therefore not a final independent test estimate.", "", "## Modality classification", "", "| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, res in results.items():
        if name == "telemonitoring": continue
        for model, m in res["outer_nested_cv_metrics"].items(): report.append(f"| {name} | {model} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |")
    prior = {
        "speech": {"svm": {"accuracy": .8000, "precision": .8273, "recall": .7583, "f1": .7913, "roc_auc": .8501}, "random_forest": {"accuracy": .7750, "precision": .7946, "recall": .7417, "f1": .7672, "roc_auc": .8312}},
        "handwriting": {"svm": {"accuracy": .8545, "precision": .8519, "recall": .9452, "f1": .8961, "roc_auc": .8663}, "random_forest": {"accuracy": .9727, "precision": .9605, "recall": 1.0000, "f1": .9799, "roc_auc": .9582}},
        "tremor": {"svm": {"accuracy": .8308, "precision": .7000, "recall": .4667, "f1": .5600, "roc_auc": .8188}, "random_forest": {"accuracy": .8154, "precision": .7500, "recall": .3000, "f1": .4286, "roc_auc": .8228}},
    }
    baseline_rows = []
    for dataset, models in prior.items():
        for model, metrics in models.items():
            for metric, old in metrics.items():
                new = results[dataset]["outer_nested_cv_metrics"][model][metric]
                baseline_rows.append({"dataset": dataset, "model": model, "metric": metric, "prior_reported": old, "current_nested_grouped": new})
    pd.DataFrame(baseline_rows).to_csv(REPORTS / "model_results" / "baseline_comparison.csv", index=False)
    report += ["", "## Classification comparison with prior reported scores", "", "Current results use nested CV and calibrated probabilities. Earlier scores are included as historical reference; small changes in preprocessing and evaluation nesting mean they should not be interpreted as a controlled model-only comparison.", "", "| Dataset | Model | Prior F1 | Current F1 | Prior ROC-AUC | Current ROC-AUC |", "|---|---|---:|---:|---:|---:|"]
    for dataset, models in prior.items():
        for model, old in models.items():
            now = results[dataset]["outer_nested_cv_metrics"][model]
            report.append(f"| {dataset} | {model} | {old['f1']:.3f} | {now['f1']:.3f} | {old['roc_auc']:.3f} | {now['roc_auc']:.3f} |")
    report += ["", "## Telemonitoring regression", "", "| Model | MAE | RMSE | R2 |", "|---|---:|---:|---:|"]
    for model, m in results["telemonitoring"]["metrics"].items(): report.append(f"| {model} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} |")
    report += ["", "## Selected deployable ensemble weights", "", "Weights were selected from full-cohort grouped OOF predictions. Use nested outer-CV scores above to estimate weight selection performance.", "", "| Modality | SVM weight | RF weight |", "|---|---:|---:|"]
    for name, res in results.items():
        if name != "telemonitoring":
            chosen = res["final_weight_selection_from_full_cohort_oof"]
            report.append(f"| {name} | {chosen['svm_weight']:.1f} | {chosen['rf_weight']:.1f} |")
    report += ["", "## Data, features, and split strategy", "", "| Dataset | Cohort | Features used | Split group |", "|---|---|---|---|"]
    report += ["| Speech | 240 recordings, 80 participants | 45 numeric acoustic features; ID, recording, and status excluded | `id` |",
               "| Handwriting | 110 tests, 40 participants (15 control, 25 PD) | 16 aggregates: points, path length, velocity, acceleration, jerk, pressure, grip angle, duration, pen-up fraction, spatial spread, bounding box | `participant_id`, joined across exact duplicate feature signatures |"]
    report.append("| Tremor | 130 participants (100 control, 30 PD) | 7 numeric UPDRS items from items 20/21; missing values imputed within training folds | `participant_code` |")
    report.append("| Telemonitoring | 5,875 recordings, 42 participants | 19 predictors: age, sex, test time, jitter/shimmer, NHR/HNR, RPDE, DFA, PPE; subject ID and total UPDRS excluded | `subject_number` |")
    report += ["", "All classification estimates use nested five-fold `StratifiedGroupKFold`; regression uses five-fold `GroupKFold`. Imputation, scaling, calibration, and weight choice are training-fold-only. The handwriting source point order is used during aggregation. The 7-row control signature and 2-row PD signature shared across participants are held in single fold groups.", "", "## Regression baseline comparison", "", "| Model | Prior MAE / RMSE / R2 | Current MAE / RMSE / R2 |", "|---|---:|---:|"]
    prior_reg = {"ridge": (7.0712, 8.2978, -0.0421), "random_forest": (7.4860, 9.1917, -0.2787)}
    for model, old in prior_reg.items():
        now = results["telemonitoring"]["metrics"][model]
        report.append(f"| {model} | {old[0]:.3f} / {old[1]:.3f} / {old[2]:.3f} | {now['mae']:.3f} / {now['rmse']:.3f} / {now['r2']:.3f} |")
    report += ["", "## Late decision-level fusion", "", "Modality OOF ensemble ROC-AUC values determine reliability weights. These weights support probability combination at inference; fusion has no experimentally measured patient-level performance because the three cohorts are unmatched.", "", "| Modality | OOF ROC-AUC | Normalized weight |", "|---|---:|---:|"]
    for modality, auc in fusion_config["validation_ROC_AUC"].items():
        report.append(f"| {modality} | {auc:.6f} | {fusion_config['normalized_weights'][modality]:.6f} |")
    report += ["", "## Interpretation and limitations", "", "Telemonitoring does not currently generalize usefully to unseen participants because R2 is negative. The handwriting audit found exact feature signatures shared across participants; handwriting folds keep matching signatures together to avoid cross-fold contamination. Scores remain exploratory because there are only 40 participants and external validation is absent. No patient-level fusion performance is claimed because the classification cohorts are unmatched. See per-dataset JSON results, feature summaries, and the handwriting leakage audit for details."]
    (REPORTS / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Pipeline completed. Reports: reports/FINAL_REPORT.md")
