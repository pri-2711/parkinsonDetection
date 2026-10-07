import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from models.common import (
    ROOT,
    plot_regression_results,
    print_regression_summary,
    train_regression_model,
)


def run_telemonitoring_models():
    telemonitoring = pd.read_csv(ROOT / "cleaned_dataset" / "telemonitoring_clean.csv")
    features = telemonitoring.drop(
        columns=["subject_number", "motor_updrs", "total_updrs"], errors="ignore"
    )
    target = telemonitoring["motor_updrs"].astype(float)
    regressors = (
        ("ridge", Ridge(alpha=10.0)),
        ("random_forest", RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5, max_features=0.8, random_state=42
        )),
        ("gradient_boosting", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2, min_samples_leaf=5,
            subsample=0.8, random_state=42
        )),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.2)),
    )
    results = {
        model_name: train_regression_model(
            "telemonitoring", features, target, telemonitoring["subject_number"], model_name, model
        )
        for model_name, model in regressors
    }
    print_regression_summary(features, target, telemonitoring["subject_number"], results)
    plot_regression_results(results)
    return results