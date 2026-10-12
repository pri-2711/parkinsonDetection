from __future__ import annotations

import numpy as np
import pandas as pd

from app_backend.errors import APIError
from featureEngineering.handwriting_features import extract_features

TABLET_COLUMNS = (
    "point_index", "x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id"
)


def extract_tablet_trajectory_features(
    trajectory: pd.DataFrame,
    model_feature_names: list[str],
) -> pd.DataFrame:
    """Apply the repository's training extractor to compatible tablet points.

    This does not convert camera images/video to tablet measurements. It exists
    to reuse the exact training-time feature code if compatible tablet telemetry
    is supplied by a future, separately validated input device.
    """
    missing = set(TABLET_COLUMNS) - set(trajectory.columns)
    if missing:
        raise APIError(422, "tablet_trajectory_schema_mismatch", "Tablet trajectory is missing required training-schema fields.")
    if trajectory.empty or trajectory["test_id"].nunique(dropna=False) != 1:
        raise APIError(422, "invalid_tablet_trajectory", "Provide one non-empty tablet test trajectory at a time.")
    try:
        points = trajectory.loc[:, list(TABLET_COLUMNS)].copy()
        for column in TABLET_COLUMNS:
            points[column] = pd.to_numeric(points[column], errors="raise")
    except (TypeError, ValueError):
        raise APIError(422, "invalid_tablet_trajectory", "Tablet trajectory fields must be numeric.") from None
    values = points.loc[:, ["x", "y", "z", "pressure", "grip_angle", "timestamp"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(points) < 2:
        raise APIError(422, "invalid_tablet_trajectory", "Trajectory measurements must be finite and contain at least two points.")
    points["participant_id"] = "backend_input"
    # The training extractor carries labels in its returned metadata. A dummy
    # label is added only for that metadata field, then removed before inference.
    points["label"] = 0
    try:
        output = points.groupby(["participant_id", "test_id"], sort=True).apply(
            extract_features, include_groups=False
        ).reset_index(drop=True)
    except (KeyError, ValueError, IndexError, FloatingPointError):
        raise APIError(422, "invalid_tablet_trajectory", "The trajectory could not be converted with the training feature extractor.") from None
    metadata_columns = {"participant_id", "label", "test_id"}
    result = output.drop(columns=[name for name in metadata_columns if name in output], errors="ignore")
    if list(result.columns) != list(model_feature_names):
        raise APIError(422, "tablet_feature_schema_mismatch", "Extracted tablet features do not match the saved handwriting model schema.")
    result = result.apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise APIError(422, "invalid_tablet_trajectory", "The training extractor produced incomplete feature values.")
    return result
