from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(frame: pd.DataFrame, excluded_columns: list[str] | None = None) -> ColumnTransformer:
    excluded = set(excluded_columns or [])
    features = frame.drop(columns=[column for column in excluded if column in frame.columns])
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipeline, numeric_columns), ("categorical", categorical_pipeline, categorical_columns)])


def fit_on_training_data(train_frame: pd.DataFrame, excluded_columns: list[str] | None = None) -> ColumnTransformer:
    preprocessor = build_preprocessor(train_frame, excluded_columns)
    preprocessor.fit(train_frame)
    return preprocessor


def transform(preprocessor: ColumnTransformer, frame: pd.DataFrame):
    return preprocessor.transform(frame)


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)