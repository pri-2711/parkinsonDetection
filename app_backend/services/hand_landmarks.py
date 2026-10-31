"""Hand landmarks for spiral videos, from MediaPipe Hands run in a separate Python environment.

MediaPipe pins its own NumPy and OpenCV, which conflict with this server's packages, so it is never
imported here. The interpreter is ``HAND_LANDMARK_PYTHON``, or the repository's ``.venv-mediapipe`` when that
exists. When neither is available, or the worker fails, the reason is reported and the ink measurements still
run. A hand that the worker did not run on is never reported as 0% detected.

Measurements are geometry of the 21 MediaPipe landmarks, in image pixels divided by the hand scale
(wrist to middle-finger knuckle). They describe what is visible in the video and are not a grip force,
a pressure or a diagnosis. Finger flexion is an angle at the joint in the image plane, so it is an
approximation that depends on the camera view.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

WORKER_PATH = Path(__file__).with_name("hand_landmark_worker.py")
WORKER_TIMEOUT_SECONDS = 600
MIN_HAND_SCORE = 0.5          # handedness score below this is not drawn or measured
MIN_FRAMES_FOR_GRIP = 30      # usable frames needed before a grip is classified
MIN_USABLE_SHARE_FOR_GRIP = 0.25
EDGE_MARGIN_PX = 4            # a landmark this close to the frame edge is treated as cut off
MIN_HAND_SCALE_PX = 8.0

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
# Landmarks the pen-holding estimate needs: thumb tip, index tip and knuckle, middle tip and knuckle.
GRIP_LANDMARKS = (4, 8, 5, 12, 9)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
NOT_MEASURED_PEN_TIP = ("Not measured separately: the pen body and tip are not detected apart from the hand in "
                        "this pipeline. Newly drawn ink is measured instead.")
NOT_MEASURED_PRESSURE = "Not measured: pressure needs a digitizer. Camera-based posture is not force."


def worker_python(config) -> str | None:
    """The MediaPipe interpreter: HAND_LANDMARK_PYTHON when it exists, otherwise None."""
    value = (config.get("HAND_LANDMARK_PYTHON") or "").strip()
    if not value or not Path(value).is_file():
        return None
    return value


def landmark_status(config) -> dict:
    if worker_python(config) is None:
        return {"available": False, "reason": "HAND_LANDMARK_PYTHON is not set to a Python with MediaPipe installed."}
    return {"available": True, "reason": None}


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def run_worker(video_path: str, config) -> tuple[list | None, dict | None, str | None]:
    """Run the worker on the video. Returns (frames, video info, reason the hands were not measured)."""
    python = worker_python(config)
    if python is None:
        return None, None, "Hand landmarks did not run: HAND_LANDMARK_PYTHON is missing or not a file."
    with tempfile.TemporaryDirectory(prefix="pd-hand-landmarks-") as work:
        output = Path(work) / "landmarks.json"
        try:
            completed = subprocess.run(
                [python, "-I", str(WORKER_PATH), video_path, str(output)],
                capture_output=True, text=True, timeout=WORKER_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired:
            return None, None, f"Hand landmarks did not run: the worker took longer than {WORKER_TIMEOUT_SECONDS} s."
        except OSError as error:
            return None, None, f"Hand landmarks did not run: the worker could not start ({error.strerror})."
        if completed.returncode != 0 or not output.exists():
            detail = _last_line(completed.stderr)
            reason = "Hand landmarks did not run: the worker failed."
            return None, None, f"{reason} {detail}".strip()
        data = json.loads(output.read_text(encoding="utf-8"))
    return data["frames"], {"width": data["width"], "height": data["height"]}, None


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    u, v = a - vertex, b - vertex
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 0:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def _confidence(detection_rate: float, mean_score: float) -> str:
    if detection_rate >= 0.8 and mean_score >= 0.8:
        return "high"
    if detection_rate >= 0.5:
        return "moderate"
    return "low"


def hand_frames(frames: list, width: int, height: int) -> list:
    """Per frame: None, or a dict with pixel landmarks (21 x 2), handedness, score, hand scale and which
    landmarks lie inside the frame."""
    out = []
    for entry in frames:
        if entry is None or entry["score"] < MIN_HAND_SCORE:
            out.append(None)
            continue
        pts = np.array([[x * width, y * height] for x, y, _ in entry["landmarks"]], dtype=float)
        scale = float(np.linalg.norm(pts[WRIST] - pts[9]))
        if scale < MIN_HAND_SCALE_PX:
            out.append(None)
            continue
        inside = ((pts[:, 0] >= EDGE_MARGIN_PX) & (pts[:, 0] <= width - EDGE_MARGIN_PX)
                  & (pts[:, 1] >= EDGE_MARGIN_PX) & (pts[:, 1] <= height - EDGE_MARGIN_PX))
        out.append({"pts": pts, "handedness": entry["handedness"], "score": entry["score"],
                    "scale": scale, "inside": inside})
    return out


def not_run_metrics(reason: str) -> tuple[dict, dict, dict]:
    """Hand values when the worker did not run. Nothing is measured, so no detection rate is reported."""
    return {}, {}, {
        "hand_landmarks": reason,
        "pen_tip_tracking": NOT_MEASURED_PEN_TIP,
        "finger_pressure": NOT_MEASURED_PRESSURE,
    }


def _frame_geometry(h) -> dict:
    pts, scale = h["pts"], h["scale"]
    flex = {}
    for name, (mcp, pip, dip, _tip) in FINGERS.items():
        # Flexion at the middle joint: 0 when straight, larger when bent.
        flex[name] = 180.0 - _angle_deg(pts[mcp], pts[pip], pts[dip])
    thumb_cmc, thumb_mcp, thumb_ip, _thumb_tip = THUMB
    flex["thumb"] = 180.0 - _angle_deg(pts[thumb_cmc], pts[thumb_mcp], pts[thumb_ip])
    vector = pts[9] - pts[WRIST]
    return {
        "flexion": flex,
        "thumb_index": float(np.linalg.norm(pts[4] - pts[8])) / scale,
        "orientation": math.degrees(math.atan2(vector[0], -vector[1])),
        "wrist": pts[WRIST] / scale,
    }


def _grip_estimate(usable, frame_count, mean_score):
    """Pen-holding estimate from the per-frame geometry of frames where the essential landmarks are inside."""
    usable_share = len(usable) / frame_count
    base = {"usable_frames": len(usable), "usable_share_of_frames": round(usable_share, 3)}
    if len(usable) < MIN_FRAMES_FOR_GRIP or usable_share < MIN_USABLE_SHARE_FOR_GRIP:
        return {
            **base,
            "label": "Insufficient visibility",
            "estimate_confidence": "none",
            "note": ("Not classified: the thumb, index and middle landmarks were not inside the frame in enough "
                     "frames. Only frames where they are visible are used."),
        }
    pinch = float(np.median([g["thumb_index"] for g in usable]))
    middle = float(np.median([g["flexion"]["middle"] for g in usable]))
    index = float(np.median([g["flexion"]["index"] for g in usable]))
    if pinch < 0.45 and middle >= 15.0:
        label = "Tripod-like grip"
    elif pinch < 0.45:
        label = "Pinch grip without visible support"
    else:
        label = "Other or uncertain grip"
    # Heuristic from one camera view, so it is never reported as high confidence.
    confidence = "moderate" if usable_share >= 0.5 and mean_score >= 0.8 else "low"
    return {
        **base,
        "label": label,
        "estimate_confidence": confidence,
        "thumb_index_share_of_hand": round(pinch, 3),
        "middle_flexion_deg": round(middle, 1),
        "index_flexion_deg": round(index, 1),
        "note": "Heuristic from one camera view. Not a validated grip classification.",
    }


def hand_metrics(hands: list, frame_count: int) -> tuple[dict, dict, dict]:
    """Measured values, estimated values and reasons for values that are not measured.

    ``hands`` holds one entry per video frame (None when no hand was found), as returned by hand_frames.
    """
    detected = [h for h in hands if h is not None]
    measured: dict = {}
    estimated: dict = {}
    unavailable: dict = {"pen_tip_tracking": NOT_MEASURED_PEN_TIP, "finger_pressure": NOT_MEASURED_PRESSURE}
    if frame_count <= 0:
        return measured, estimated, {**unavailable, "hand_landmarks": "The video has no frames."}
    detection_rate = len(detected) / frame_count
    mean_score = float(np.mean([h["score"] for h in detected])) if detected else 0.0
    confidence = _confidence(detection_rate, mean_score)
    measured["frames_with_hand_detected"] = round(detection_rate, 3)
    if not detected:
        unavailable["hand_landmarks"] = "No hand was detected with enough confidence in this video."
        return measured, estimated, unavailable

    handedness = Counter(h["handedness"] for h in detected).most_common(1)[0][0]
    measured["handedness"] = handedness
    measured["hand_detection_confidence"] = confidence
    measured["hand_scale_px_median"] = round(float(np.median([h["scale"] for h in detected])), 1)

    # Fingertips inside the frame, per finger. A fingertip outside the frame cannot be measured.
    for name, tip in FINGERTIPS.items():
        measured[f"{name}_tip_visible_share"] = round(float(np.mean([h["inside"][tip] for h in detected])), 3)

    geometry = [_frame_geometry(h) for h in hands if h is not None]
    flexion = {name: [g["flexion"][name] for g in geometry] for name in list(FINGERS) + ["thumb"]}
    for name in FINGERS:
        estimated[f"{name}_flexion_deg_median"] = round(float(np.median(flexion[name])), 1)
    estimated["thumb_flexion_deg_median"] = round(float(np.median(flexion["thumb"])), 1)
    estimated["thumb_index_distance_share_of_hand"] = round(float(np.median([g["thumb_index"] for g in geometry])), 3)
    orientation = [g["orientation"] for g in geometry]
    estimated["hand_orientation_deg_median"] = round(float(np.median(orientation)), 1)
    estimated["hand_orientation_spread_deg"] = round(float(np.subtract(*np.percentile(orientation, [75, 25]))), 1)

    # Wrist movement and landmark stability use consecutive detected frames only.
    wrist_positions = [g["wrist"] for g in geometry]
    if len(wrist_positions) > 1:
        path = np.asarray(wrist_positions)
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        measured["wrist_path_length_hand_scales"] = round(float(steps.sum()), 2)
        measured["wrist_net_displacement_hand_scales"] = round(float(np.linalg.norm(path[-1] - path[0])), 2)
    landmark_motion = []
    previous = None
    for h in hands:
        if h is None:
            previous = None
            continue
        if previous is not None:
            landmark_motion.append(float(np.mean(np.linalg.norm(h["pts"] - previous["pts"], axis=1))) / h["scale"])
        previous = h
    if landmark_motion:
        estimated["landmark_motion_median_per_frame"] = round(float(np.median(landmark_motion)), 4)
        estimated["landmark_motion_spikes"] = int(np.sum(np.asarray(landmark_motion) > 3 * np.median(landmark_motion)))

    # The grip uses only frames where the essential landmarks are inside the frame.
    usable = [_frame_geometry(h) for h in detected if all(h["inside"][i] for i in GRIP_LANDMARKS)]
    estimated["grip_configuration"] = _grip_estimate(usable, frame_count, mean_score)

    unavailable["finger_occlusion"] = ("Not measured per finger: the hand model gives landmark positions but no "
                                       "per-landmark visibility. Cut-off fingertips are shown as visible shares.")
    return measured, estimated, unavailable


def draw_hand(frame_bgr, hand) -> None:
    """Landmarks and their connections in green (fingertips larger), then a convex outline of the hand in red."""
    import cv2
    if hand is None:
        return
    pts = hand["pts"].astype(np.int32)
    for a, b in CONNECTIONS:
        cv2.line(frame_bgr, tuple(pts[a]), tuple(pts[b]), (60, 200, 60), 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(pts):
        radius = 3 if index in FINGERTIPS.values() else 2
        cv2.circle(frame_bgr, (int(x), int(y)), radius, (60, 200, 60), -1, cv2.LINE_AA)
    hull = cv2.convexHull(pts)
    cv2.polylines(frame_bgr, [hull], True, (40, 40, 230), 2, cv2.LINE_AA)


def hand_mask(frame_shape, hand, margin_share: float = 0.25):
    """Filled convex hull of the hand, dilated, for excluding hand pixels from ink."""
    import cv2
    if hand is None:
        return None
    mask = np.zeros(frame_shape[:2], np.uint8)
    hull = cv2.convexHull(hand["pts"].astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    margin = max(3, int(margin_share * hand["scale"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.dilate(mask, kernel) > 0
