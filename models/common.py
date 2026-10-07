from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    print(f"\n{title}")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _save_metrics(dataset_name: str, task: str, results: dict[str, dict[str, Any]]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, np.number))})
        rows.append(row)
    output = ARTIFACT_DIR / f"{dataset_name}_{task}_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def print_classification_summary(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    results: dict[str, dict[str, Any]],
) -> None:
    class_counts = ", ".join(f"{label}={count}" for label, count in target.value_counts().sort_index().items())
    print(
        f"Data: {len(target)} rows, {features.shape[1]} features, {groups.nunique()} subjects, "
        f"class counts ({class_counts}); 5-fold subject-disjoint cross-validation."
    )
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows = [
        [model_name, *(f"{metrics.get(key, float('nan')):.4f}" for key in metric_keys)]
        for model_name, metrics in results.items()
    ]
    _print_table(f"{dataset_name.title()} classification scores (out-of-fold)", headers, rows)
    print(f"Metrics CSV: {_save_metrics(dataset_name, 'classification', results)}")


def plot_classification_results(dataset_name: str, results: dict[str, dict[str, Any]]) -> list[Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(results)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    positions = np.arange(len(model_names))
    width = 0.15

    figure, axis = plt.subplots(figsize=(10, 5))
    for index, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [results[name].get(key, 0.0) for name in model_names]
        axis.bar(positions + (index - 2) * width, values, width, label=label)
    axis.set_title(f"{dataset_name.title()} classifier comparison")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions, model_names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, frameon=False)
    figure.tight_layout()
    score_plot = ARTIFACT_DIR / f"{dataset_name}_classification_scores.png"
    figure.savefig(score_plot, dpi=160)
    plt.close(figure)

    best_name = max(results, key=lambda name: results[name]["f1"])
    matrix = np.asarray(results[best_name]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(f"{dataset_name.title()} best model: {best_name}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks([0, 1], ["Healthy", "Parkinson's"])
    axis.set_yticks([0, 1], ["Healthy", "Parkinson's"])
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    confusion_plot = ARTIFACT_DIR / f"{dataset_name}_best_confusion_matrix.png"
    figure.savefig(confusion_plot, dpi=160)
    plt.close(figure)
    print(f"Plots: {score_plot}, {confusion_plot} (best by F1: {best_name})")
    return [score_plot, confusion_plot]


def print_regression_summary(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    results: dict[str, dict[str, Any]],
) -> None:
    print(
        f"Data: {len(target)} rows, {features.shape[1]} features, {groups.nunique()} subjects; "
        f"target motor_UPDRS mean={target.mean():.3f}, std={target.std():.3f}, "
        f"range={target.min():.3f}..{target.max():.3f}; 5-fold subject-disjoint cross-validation."
    )
    headers = ["Model", "MAE", "RMSE", "R2"]
    rows = [
        [model_name, f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['r2']:.4f}"]
        for model_name, metrics in results.items()
    ]
    _print_table("Telemonitoring regression scores (out-of-fold)", headers, rows)
    print(f"Metrics CSV: {_save_metrics('telemonitoring', 'regression', results)}")


def plot_regression_results(results: dict[str, dict[str, Any]]) -> list[Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(results)
    positions = np.arange(len(model_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width / 2, [results[name]["mae"] for name in model_names], width, label="MAE")
    axis.bar(positions + width / 2, [results[name]["rmse"] for name in model_names], width, label="RMSE")
    axis.set_title("Telemonitoring regression error comparison")
    axis.set_ylabel("Error (Motor UPDRS points; lower is better)")
    axis.set_xticks(positions, model_names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    error_plot = ARTIFACT_DIR / "telemonitoring_regression_errors.png"
    figure.savefig(error_plot, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(model_names, [results[name]["r2"] for name in model_names], color="#4287a8")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Telemonitoring regression R2 comparison")
    axis.set_ylabel("R2 (higher is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    r2_plot = ARTIFACT_DIR / "telemonitoring_regression_r2.png"
    figure.savefig(r2_plot, dpi=160)
    plt.close(figure)
    print(f"Plots: {error_plot}, {r2_plot}")
    return [error_plot, r2_plot]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical_features,
        ))

    return ColumnTransformer(transformers=transformers)


def class_report(name: str, actual: pd.Series, predicted: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "dataset": name,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }
    if probabilities is not None and len(np.unique(actual)) > 1:
        metrics["roc_auc"] = roc_auc_score(actual, probabilities)
    return metrics


def regression_report(name: str, actual: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    mse = mean_squared_error(actual, predicted)
    return {
        "dataset": name,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mse),
        "r2": r2_score(actual, predicted),
    }


def train_classification_model(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    model_name: str,
    model: Any,
) -> dict[str, Any]:
    if len(features) != len(target) or len(target) != len(groups):
        raise ValueError("Features, targets, and subject groups must have the same number of rows")

    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    predictions = np.empty(len(target), dtype=target.dtype)
    probabilities = np.empty(len(target), dtype=float) if hasattr(model, "predict_proba") else None
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_indices, test_indices in splitter.split(features, target, groups):
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(features.iloc[train_indices])),
            ("model", model),
        ])
        pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
        predictions[test_indices] = pipeline.predict(features.iloc[test_indices])
        if probabilities is not None:
            probabilities[test_indices] = pipeline.predict_proba(features.iloc[test_indices])[:, 1]

    report = class_report(dataset_name, target, predictions, probabilities)
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(features)),
        ("model", model),
    ])
    final_pipeline.fit(features, target)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, ARTIFACT_DIR / f"{dataset_name}_{model_name}_classifier.joblib")
    return report


def train_regression_model(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    model_name: str,
    model: Any,
) -> dict[str, Any]:
    if len(features) != len(target) or len(target) != len(groups):
        raise ValueError("Features, targets, and subject groups must have the same number of rows")

    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    predictions = np.empty(len(target), dtype=float)
    splitter = GroupKFold(n_splits=5)

    for train_indices, test_indices in splitter.split(features, target, groups):
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(features.iloc[train_indices])),
            ("model", model),
        ])
        pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
        predictions[test_indices] = pipeline.predict(features.iloc[test_indices])

    report = regression_report(dataset_name, target, predictions)
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(features)),
        ("model", model),
    ])
    final_pipeline.fit(features, target)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, ARTIFACT_DIR / f"{dataset_name}_{model_name}_regressor.joblib")
    return report