"""Train mode-specific models from labeled, reproducible raw WAV recordings.

Manifest CSV columns: audio_path, subject_id, mode, gender_code, status.
`status` is 0=control, 1=Parkinson's; `gender_code` is the source-dataset
encoding 0=man, 1=woman. This script deliberately does not consume the legacy
precomputed feature table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app_backend.services.acoustic_features import AcousticFeatureExtractor, SPEECH_V2_FEATURES


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"audio_path", "subject_id", "mode", "gender_code", "status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(missing)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing required values")
    if not set(manifest["mode"]).issubset({"sustained_vowel", "reading"}):
        raise ValueError("mode must be sustained_vowel or reading")
    if not set(manifest["status"].astype(int)).issubset({0, 1}) or not set(manifest["gender_code"].astype(int)).issubset({0, 1}):
        raise ValueError("status and gender_code must use the documented 0/1 codes")
    base = path.resolve().parent
    manifest["audio_path"] = manifest["audio_path"].map(lambda value: str((base / value).resolve()) if not Path(value).is_absolute() else value)
    absent = [value for value in manifest["audio_path"] if not Path(value).is_file()]
    if absent:
        raise ValueError(f"Audio file not found: {absent[0]}")
    return manifest


def _subject_split(rows: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    y = rows["status"].astype(int).to_numpy()
    groups = rows["subject_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 8:
        raise ValueError("Each mode needs at least 8 distinct subjects before subject-level evaluation")
    for candidate_seed in range(seed, seed + 100):
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=candidate_seed).split(rows, y, groups))
        if set(y[train_idx]) == {0, 1} and set(y[test_idx]) == {0, 1}:
            return train_idx, test_idx
    raise ValueError("Could not create a subject-disjoint split containing both classes")


def train(manifest_path: Path, output_dir: Path) -> dict:
    manifest = load_manifest(manifest_path)
    extractor = AcousticFeatureExtractor()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for mode in ("sustained_vowel", "reading"):
        rows = manifest.loc[manifest["mode"] == mode].reset_index(drop=True)
        if rows.empty:
            continue
        feature_rows = []
        for row in rows.itertuples(index=False):
            audio_bytes = Path(row.audio_path).read_bytes()
            metadata = {"sample_rate_hz": None}
            feature_rows.append(extractor.extract(audio_bytes, metadata, mode=mode, gender_code=int(row.gender_code)).iloc[0])
        x = pd.DataFrame(feature_rows, columns=SPEECH_V2_FEATURES)
        y = rows["status"].astype(int).to_numpy()
        groups = rows["subject_id"].astype(str).to_numpy()
        if any(rows.groupby("subject_id")["status"].nunique() != 1):
            raise ValueError(f"Conflicting labels for a subject in {mode} mode")
        train_idx, test_idx = _subject_split(rows)
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        model.fit(x.iloc[train_idx], y[train_idx])
        probability = model.predict_proba(x.iloc[test_idx])[:, 1]
        predictions = (probability >= 0.5).astype(int)
        bundle = {"schema": "speech_audio_v2", "mode": mode, "features": SPEECH_V2_FEATURES, "model": model}
        joblib.dump(bundle, output_dir / f"{mode}.joblib")
        results[mode] = {
            "n_recordings": int(len(rows)), "n_subjects": int(len(np.unique(groups))),
            "train_subjects": int(len(np.unique(groups[train_idx]))),
            "test_subjects": int(len(np.unique(groups[test_idx]))),
            "accuracy": float(accuracy_score(y[test_idx], predictions)),
            "f1": float(f1_score(y[test_idx], predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[test_idx], probability)),
            "threshold": 0.5,
            "validation": "one fixed subject-disjoint 75/25 holdout; exploratory, dataset-dependent",
        }
    if not results:
        raise ValueError("Manifest contains no supported recording mode")
    (output_dir / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output), indent=2))
