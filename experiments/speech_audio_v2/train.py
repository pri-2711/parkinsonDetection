"""Train mode-specific speech_audio_v2 classifiers from labelled raw WAV recordings.

Manifest CSV columns: audio_path, subject_id, mode, gender_code, status.
`status` is 0=control, 1=Parkinson's; `gender_code` is 0=man, 1=woman.

Protocol (all learned steps are fitted inside training folds only):
  * Features are extracted by app_backend.services.acoustic_features, the same code used at
    prediction time, so training and inference share one definition.
  * Outer evaluation: StratifiedGroupKFold over subjects (5 folds, seed 42). Every recording of a
    subject lies in one fold, so no subject is in both training and test data.
  * Inside each outer training fold: standardization, univariate feature selection (SelectKBest)
    and the classifier hyperparameters are chosen by grouped inner CV (ROC-AUC).
  * Probabilities come from sigmoid calibration fitted on grouped calibration folds of the training data.
  * The deployed family (SVM or random forest) is chosen by grouped CV on the full training set,
    then refitted on all recordings with the same grouped calibration.

The UCI precomputed table is not used: it has no raw audio to verify extraction against.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app_backend.errors import APIError
from app_backend.services.acoustic_features import AcousticFeatureExtractor, SPEECH_V2_FEATURES, SPEECH_V2_MODEL_FEATURES

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
CALIBRATION_FOLDS = 3
MIN_SUBJECTS = 8
MODES = ("sustained_vowel", "reading")
REQUIRED_COLUMNS = {"audio_path", "subject_id", "mode", "gender_code", "status"}


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(missing)}")
    if manifest[sorted(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Manifest contains missing required values")
    if not set(manifest["mode"]).issubset(set(MODES)):
        raise ValueError("mode must be sustained_vowel or reading")
    if not set(manifest["status"].astype(int)).issubset({0, 1}) or not set(manifest["gender_code"].astype(int)).issubset({0, 1}):
        raise ValueError("status and gender_code must use the documented 0/1 codes")
    base = path.resolve().parent
    manifest["audio_path"] = manifest["audio_path"].map(
        lambda value: str((base / value).resolve()) if not Path(value).is_absolute() else value)
    absent = [value for value in manifest["audio_path"] if not Path(value).is_file()]
    if absent:
        raise ValueError(f"Audio file not found: {absent[0]}")
    manifest["status"] = manifest["status"].astype(int)
    manifest["gender_code"] = manifest["gender_code"].astype(int)
    if (manifest.groupby("subject_id")["status"].nunique() > 1).any():
        raise ValueError("Conflicting labels for a subject")
    return manifest


CACHE_COLUMNS = ["audio_path", "subject_id", "mode", "gender_code", "status"]
CACHE_HEADER = CACHE_COLUMNS + SPEECH_V2_MODEL_FEATURES


def _drop_partial_tail(cache: Path) -> None:
    """A process killed mid-write can leave a final line without its newline. Remove it; that recording is re-extracted."""
    data = cache.read_bytes()
    newline = bytes([10])
    if data and not data.endswith(newline):
        cut = data.rfind(newline)
        cache.write_bytes(data[: cut + 1] if cut >= 0 else b"")


def _read_cache(cache: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Load a partial cache and verify it belongs to this manifest and this feature schema."""
    _drop_partial_tail(cache)
    if cache.stat().st_size == 0:
        return pd.DataFrame(columns=CACHE_HEADER)
    table = pd.read_csv(cache)
    if list(table.columns) != CACHE_HEADER:
        raise ValueError(f"Feature cache {cache} does not match the expected {len(SPEECH_V2_MODEL_FEATURES)}-feature "
                         "schema; move it aside to re-extract")
    if table["audio_path"].duplicated().any():
        raise ValueError(f"Feature cache {cache} contains duplicate recordings")
    values = table[SPEECH_V2_MODEL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Feature cache {cache} contains non-finite feature values")
    manifest = rows.assign(audio_path=rows["audio_path"].astype(str)).set_index("audio_path")
    unknown = set(table["audio_path"].astype(str)) - set(manifest.index)
    if unknown:
        raise ValueError(f"Feature cache {cache} holds recordings that are not in this manifest")
    for record in table.itertuples(index=False):
        expected = manifest.loc[str(record.audio_path)]
        if str(record.subject_id) != str(expected["subject_id"]) or int(record.status) != int(expected["status"]):
            raise ValueError(f"Feature cache {cache} has a subject or label that differs from the manifest for {record.audio_path}")
    return table


def extract_features(rows: pd.DataFrame, extractor: AcousticFeatureExtractor, mode: str,
                     cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Extract every recording, resuming from `cache` when given.

    Each successful recording is appended and flushed to the cache immediately, so an interrupted run
    keeps its completed work. Returns features, the matching subject/label rows in manifest order,
    and a log of recordings that could not be extracted.
    """
    total = len(rows)
    records: dict[str, dict] = {}
    if cache is not None and cache.is_file():
        for record in _read_cache(cache, rows).to_dict("records"):
            records[str(record["audio_path"])] = record
    cached_count = len(records)
    skipped: list[dict] = []
    handle = None
    writer = None
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fresh = not cache.is_file() or cache.stat().st_size == 0
        handle = cache.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=CACHE_HEADER)
        if fresh:
            writer.writeheader()
            handle.flush()
    try:
        for index, row in enumerate(rows.itertuples(index=False), 1):
            path = str(row.audio_path)
            name = Path(path).name
            if path in records:
                print(f"[{index}/{total}] {name} cached", flush=True)
                continue
            try:
                frame = extractor.extract(Path(path).read_bytes(), {}, mode=mode, gender_code=int(row.gender_code))
            except APIError as error:
                skipped.append({"audio_path": name, "subject_id": str(row.subject_id), "code": error.code})
                print(f"[{index}/{total}] {name} skipped ({error.code})", flush=True)
                continue
            except Exception:  # extractor failure on an unusual recording; logged, never silently used
                skipped.append({"audio_path": name, "subject_id": str(row.subject_id), "code": "unexpected_extraction_error"})
                print(f"[{index}/{total}] {name} skipped (unexpected_extraction_error)", flush=True)
                continue
            record = {"audio_path": path, "subject_id": str(row.subject_id), "mode": mode,
                      "gender_code": int(row.gender_code), "status": int(row.status),
                      **{name_: float(frame.iloc[0][name_]) for name_ in SPEECH_V2_MODEL_FEATURES}}
            if writer is not None:
                writer.writerow(record)
                handle.flush()
                os.fsync(handle.fileno())
            records[path] = record
            print(f"[{index}/{total}] {name} extracted", flush=True)
    finally:
        if handle is not None:
            handle.close()
    print(f"extraction done: {len(records)} recordings ({cached_count} from cache), {len(skipped)} skipped", flush=True)
    ordered = [records[str(path)] for path in rows["audio_path"].astype(str) if str(path) in records]
    table = pd.DataFrame(ordered, columns=CACHE_HEADER)
    features = table[SPEECH_V2_MODEL_FEATURES].reset_index(drop=True)
    kept = table[CACHE_COLUMNS].reset_index(drop=True)
    kept["gender_code"] = kept["gender_code"].astype(int)
    kept["status"] = kept["status"].astype(int)
    return features, kept, skipped


def outer_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(groups)) < MIN_SUBJECTS:
        raise ValueError(f"Each mode needs at least {MIN_SUBJECTS} distinct subjects before subject-level evaluation")
    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in folds:
        if set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
            raise ValueError("A subject-disjoint fold does not contain both classes; add more subjects")
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("subject leakage between training and test folds")
    return folds


def _inner_splits(y: np.ndarray, groups: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-disjoint folds that keep both classes in every fold where the data allow it."""
    subjects_per_class = min(pd.Series(groups).groupby(y).nunique())
    n_splits = max(2, min(folds, subjects_per_class))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _base_pipeline(family: str) -> tuple[Pipeline, dict]:
    if family == "svm":
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
        ])
        grid = {"select__k": [10, 20, len(SPEECH_V2_FEATURES)], "clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}
    else:
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1)),
        ])
        grid = {"select__k": [10, 20, len(SPEECH_V2_FEATURES)], "clf__max_depth": [4, 8], "clf__min_samples_leaf": [3, 6]}
    return pipeline, grid


def fit_family(family: str, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[CalibratedClassifierCV, dict, float]:
    """Inner grouped search and grouped sigmoid calibration. Both use training rows only."""
    pipeline, grid = _base_pipeline(family)
    inner = _inner_splits(y, groups, INNER_FOLDS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = GridSearchCV(pipeline, grid, cv=inner, scoring="roc_auc", refit=False, n_jobs=1)
        search.fit(x, y)
        tuned = clone(pipeline).set_params(**search.best_params_)
        calibration_splits = _inner_splits(y, groups, CALIBRATION_FOLDS)
        calibrated = CalibratedClassifierCV(tuned, method="sigmoid", cv=calibration_splits)
        calibrated.fit(x, y)
    return calibrated, search.best_params_, float(search.best_score_)


def metrics_for(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def _summary(values: list[dict]) -> dict:
    keys = values[0].keys()
    return {key: {"mean": float(np.mean([v[key] for v in values])),
                  "sd": float(np.std([v[key] for v in values], ddof=1)) if len(values) > 1 else 0.0} for key in keys}


def train_mode(mode: str, rows: pd.DataFrame, extractor: AcousticFeatureExtractor, output_dir: Path,
               cache: Path | None = None) -> dict:
    x, kept, skipped = extract_features(rows, extractor, mode, cache)
    if kept.empty:
        raise ValueError(f"No {mode} recording could be extracted")
    y = kept["status"].to_numpy(dtype=int)
    groups = kept["subject_id"].astype(str).to_numpy()
    folds = outer_folds(y, groups)

    per_family: dict[str, dict] = {}
    for family in ("svm", "random_forest"):
        fold_metrics = []
        for train, test in folds:
            model, _, _ = fit_family(family, x.iloc[train], y[train], groups[train])
            fold_metrics.append(metrics_for(y[test], model.predict_proba(x.iloc[test])[:, 1]))
        per_family[family] = {"folds": fold_metrics, "summary": _summary(fold_metrics)}

    # Deployment family is chosen from grouped CV on the training data, not from the outer test folds.
    deployed_family, deployed_params, deployed_inner_auc = None, None, -np.inf
    for family in ("svm", "random_forest"):
        _, params, score = fit_family(family, x, y, groups)
        if not np.isfinite(score):
            raise ValueError(f"Grouped inner validation produced no finite ROC-AUC for {family}; check the data")
        per_family[family]["full_data_inner_roc_auc"] = score
        if score > deployed_inner_auc:
            deployed_family, deployed_params, deployed_inner_auc = family, params, score
    final_model, _, _ = fit_family(deployed_family, x, y, groups)

    if set(final_model.classes_) != {0, 1}:
        raise ValueError("Final model did not learn both classes")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": "speech_audio_v2", "mode": mode, "features": SPEECH_V2_MODEL_FEATURES, "model": final_model,
        "family": deployed_family, "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in deployed_params.items()},
    }
    joblib.dump(bundle, output_dir / f"{mode}.joblib")
    return {
        "n_recordings": int(len(kept)),
        "n_subjects": int(len(np.unique(groups))),
        "n_control_subjects": int(kept.drop_duplicates("subject_id")["status"].eq(0).sum()),
        "n_parkinson_subjects": int(kept.drop_duplicates("subject_id")["status"].eq(1).sum()),
        "skipped_recordings": skipped,
        "outer_folds": OUTER_FOLDS,
        "subject_disjoint_folds": True,
        "majority_class_accuracy": float(max(np.mean(y), 1 - np.mean(y))),
        "families": {family: {"outer_fold_metrics": data["folds"], "outer_fold_summary": data["summary"],
                              "full_data_inner_roc_auc": data.get("full_data_inner_roc_auc")}
                     for family, data in per_family.items()},
        "deployed_family": deployed_family,
        "deployed_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in deployed_params.items()},
        "threshold": 0.5,
        "validation": "StratifiedGroupKFold by subject, 5 outer folds; feature selection, scaling, tuning and calibration inside training folds",
    }


def train(manifest_path: Path, output_dir: Path, cache_dir: Path | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    extractor = AcousticFeatureExtractor()
    results: dict[str, dict] = {}
    for mode in MODES:
        rows = manifest.loc[manifest["mode"] == mode].reset_index(drop=True)
        if rows.empty:
            continue
        cache = cache_dir / f"features_{mode}.csv" if cache_dir else None
        results[mode] = train_mode(mode, rows, extractor, output_dir, cache)
    if not results:
        raise ValueError("Manifest contains no supported recording mode")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--cache-dir", type=Path, default=None, help="reuse extracted features from this folder")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.cache_dir), indent=2))
