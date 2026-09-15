from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "cleaned_dataset" / "handwriting_clean.csv"
OUTPUT = Path(__file__).resolve().parent / "handwriting_features.csv"


def _safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) else 0.0


def extract_features(group: pd.DataFrame) -> pd.Series:
    participant_id, test_id = group.name
    group = group.sort_values("timestamp").reset_index(drop=True)
    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)
    pressure = group["pressure"].to_numpy(dtype=float)
    grip = group["grip_angle"].to_numpy(dtype=float)
    timestamp = group["timestamp"].to_numpy(dtype=float)
    dt = np.diff(timestamp, prepend=timestamp[0]) / 1000.0
    positive_dt = dt[1:][dt[1:] > 0]
    fallback_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.001
    dt[0] = fallback_dt
    dt[dt <= 0] = fallback_dt
    time_axis = np.cumsum(dt)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    distance = np.hypot(dx, dy)
    velocity = distance / dt
    acceleration = np.gradient(velocity, time_axis)
    jerk = np.gradient(acceleration, time_axis)
    duration = max(float(timestamp[-1] - timestamp[0]) / 1000.0, 0.0) if len(timestamp) else 0.0
    pen_up = group["z"].fillna(0).to_numpy(dtype=float) != 0
    center_x, center_y = np.mean(x), np.mean(y)
    radial_distance = np.hypot(x - center_x, y - center_y)
    return pd.Series({
        "participant_id": participant_id,
        "label": int(group["label"].iloc[0]),
        "test_id": int(test_id),
        "n_points": int(len(group)),
        "path_length": float(distance.sum()),
        "mean_velocity": float(np.mean(velocity)),
        "max_velocity": float(np.max(velocity)) if len(velocity) else 0.0,
        "mean_acceleration": float(np.mean(np.abs(acceleration))),
        "max_acceleration": float(np.max(np.abs(acceleration))) if len(acceleration) else 0.0,
        "mean_jerk": float(np.mean(np.abs(jerk))),
        "max_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
        "pressure_mean": float(np.mean(pressure)),
        "pressure_std": _safe_std(pd.Series(pressure)),
        "pressure_range": float(np.max(pressure) - np.min(pressure)) if len(pressure) else 0.0,
        "grip_angle_std": _safe_std(pd.Series(grip)),
        "movement_duration": duration,
        "pen_up_fraction": float(np.mean(pen_up)) if len(pen_up) else 0.0,
        "spatial_std": _safe_std(pd.Series(radial_distance)),
        "bounding_box_diagonal": float(np.hypot(np.ptp(x), np.ptp(y))) if len(x) else 0.0,
    })


def build_handwriting_features() -> pd.DataFrame:
    trajectory = pd.read_csv(INPUT)
    features = trajectory.groupby(["participant_id", "test_id"], sort=True).apply(extract_features, include_groups=False).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)
    print(f"Handwriting feature report: trajectory_rows={len(trajectory)}, participant_tests={len(features)}, features={len(features.columns) - 3}")
    print(f"Saved: {OUTPUT}")
    return features


if __name__ == "__main__":
    build_handwriting_features()