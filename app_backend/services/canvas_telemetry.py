"""Validation and summary of browser canvas telemetry (not digitizer-compatible)."""
from __future__ import annotations

import math
from statistics import pstdev

from app_backend.errors import APIError


def summarize_canvas_telemetry(payload: object) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise APIError(400, "invalid_canvas_telemetry", "Provide canvas telemetry with a samples array.")
    samples = payload["samples"]
    if len(samples) < 2 or len(samples) > 50_000:
        raise APIError(422, "invalid_canvas_telemetry", "Canvas telemetry must contain 2 to 50,000 samples.")
    try:
        width, height = int(payload.get("canvas_width")), int(payload.get("canvas_height"))
    except (TypeError, ValueError):
        raise APIError(422, "invalid_canvas_telemetry", "Canvas dimensions must be integers.") from None
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise APIError(422, "invalid_canvas_telemetry", "Canvas dimensions are outside the supported range.")

    clean = []
    previous_time = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise APIError(422, "invalid_canvas_telemetry", f"Sample {index} must be an object.")
        try:
            x, y, timestamp = float(sample["x"]), float(sample["y"]), float(sample["timestampMs"])
            stroke_id = int(sample["strokeId"])
        except (KeyError, TypeError, ValueError):
            raise APIError(422, "invalid_canvas_telemetry", f"Sample {index} is missing valid coordinates, time, or stroke ID.") from None
        phase = sample.get("phase")
        if (not all(math.isfinite(value) for value in (x, y, timestamp)) or phase not in {"down", "move", "up"}
                or timestamp < previous_time or timestamp > 600_000 or stroke_id < 0
                or x < 0 or y < 0 or x > width or y > height):
            raise APIError(422, "invalid_canvas_telemetry", f"Sample {index} has invalid coordinates, ordering, phase, or timing.")
        previous_time = timestamp
        pressure = sample.get("pressure")
        pressure_available = sample.get("pressureAvailable") is True
        if pressure_available:
            if not isinstance(pressure, (int, float)) or not math.isfinite(float(pressure)) or not 0 <= float(pressure) <= 1:
                raise APIError(422, "invalid_canvas_telemetry", f"Sample {index} has invalid browser pressure data.")
        else:
            pressure = None
        tilt_available = sample.get("tiltAvailable") is True
        tilt_x, tilt_y = sample.get("tiltX"), sample.get("tiltY")
        if tilt_available:
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -90 <= float(value) <= 90 for value in (tilt_x, tilt_y)):
                raise APIError(422, "invalid_canvas_telemetry", f"Sample {index} has invalid browser tilt data.")
        else:
            tilt_x = tilt_y = None
        clean.append({"x": x, "y": y, "timestamp": timestamp, "stroke_id": stroke_id,
                      "phase": phase, "pressure": float(pressure) if pressure_available else None,
                      "pressure_available": pressure_available, "tilt_available": tilt_available})

    speeds, accelerations, jerks, turns, pauses = [], [], [], [], []
    path_length = 0.0
    for index in range(1, len(clean)):
        a, b = clean[index - 1], clean[index]
        elapsed_ms = b["timestamp"] - a["timestamp"]
        if elapsed_ms >= 250:
            pauses.append(round(elapsed_ms, 2))
        if a["stroke_id"] != b["stroke_id"]:
            continue
        distance = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        path_length += distance
        dt = elapsed_ms / 1000
        if dt > 0:
            speeds.append(distance / dt)
        if index >= 2:
            first, middle = clean[index - 2], a
            if first["stroke_id"] == middle["stroke_id"] == b["stroke_id"]:
                v1 = math.hypot(middle["x"] - first["x"], middle["y"] - first["y"])
                v2 = distance
                dt1 = (middle["timestamp"] - first["timestamp"]) / 1000
                if dt > 0 and dt1 > 0:
                    acceleration = (v2 / dt - v1 / dt1) / ((dt + dt1) / 2)
                    accelerations.append(acceleration)
                    if len(accelerations) > 1:
                        jerk_dt = (b["timestamp"] - first["timestamp"]) / 1000
                        if jerk_dt > 0:
                            jerks.append((accelerations[-1] - accelerations[-2]) / jerk_dt)
                u = (middle["x"] - first["x"], middle["y"] - first["y"])
                v = (b["x"] - middle["x"], b["y"] - middle["y"])
                denom = math.hypot(*u) * math.hypot(*v)
                if denom:
                    cosine = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / denom))
                    turns.append(math.degrees(math.acos(cosine)))

    x_values = [point["x"] for point in clean]
    y_values = [point["y"] for point in clean]
    pressure = [point["pressure"] for point in clean if point["pressure_available"]]
    tilt_count = sum(point["tilt_available"] for point in clean)
    duration_ms = max(0.0, clean[-1]["timestamp"] - clean[0]["timestamp"])

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "feature_schema": "browser_canvas_2d_v1",
        "model_compatible": False,
        "prediction_available": False,
        "sample_count": len(clean),
        "stroke_count": len({point["stroke_id"] for point in clean if point["phase"] == "down"}),
        "path_length_px": round(path_length, 3),
        "mean_speed_px_per_second": round(mean(speeds), 3) if speeds else None,
        "max_speed_px_per_second": round(max(speeds), 3) if speeds else None,
        "mean_acceleration_px_per_second2": round(mean(accelerations), 3) if accelerations else None,
        "mean_jerk_px_per_second3": round(mean(jerks), 3) if jerks else None,
        "mean_direction_change_degrees": round(mean(turns), 3) if turns else None,
        "bounding_box_diagonal_px": round(math.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values)), 3),
        "spatial_std_px": round(math.hypot(pstdev(x_values), pstdev(y_values)), 3),
        "duration_ms": round(duration_ms, 3),
        "pause_durations_ms": pauses,
        "pressure_samples": len(pressure),
        "pressure_coverage": round(len(pressure) / len(clean), 4),
        "mean_browser_pressure_normalized": round(mean(pressure), 4) if pressure else None,
        "tilt_samples": int(tilt_count),
        "training_schema_coverage": {
            "direct_browser_count_only": ["n_points"],
            "approximate_canvas_units": ["path_length", "velocity", "acceleration", "jerk", "movement_duration", "spatial_std", "bounding_box_diagonal"],
            "unavailable_or_incompatible": ["z_axis_pen_up", "calibrated_pressure", "grip_angle_std"],
        },
        "limitations": [
            "Canvas coordinates use CSS-scaled browser pixels and are not physical or digitizer-calibrated units.",
            "Browser pressure is normalized device-reported data only; it is not calibrated tablet pressure.",
            "Pen tilt is not grip angle, and pointer stroke boundaries are not digitizer Z-axis pen-up measurements.",
            "These observations are not fed to the PaHaW handwriting classifier and produce no disease prediction.",
        ],
    }
