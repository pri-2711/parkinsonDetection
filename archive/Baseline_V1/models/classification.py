from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from models.common import (
    plot_classification_results,
    print_classification_summary,
    train_classification_model,
)


def run_classifiers(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> dict[str, dict[str, Any]]:
    classifiers = (
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ("random_forest", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3, max_features="sqrt", random_state=42
        )),
        ("gradient_boosting", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, min_samples_leaf=3,
            subsample=0.8, random_state=42
        )),
    )
    results = {
        model_name: train_classification_model(dataset_name, features, target, groups, model_name, model)
        for model_name, model in classifiers
    }
    print_classification_summary(dataset_name, features, target, groups, results)
    plot_classification_results(dataset_name, results)
    return results