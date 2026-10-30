"""Image-based observations from a spiral-writing video (visual analysis only, not a diagnostic model).

Two independent detectors run on every frame:

BLUE - ink and strokes. Pen ink is found from its colour and local contrast: a pixel is ink when it is
  darker than its local background (black-hat), has a blue-violet hue and is dark. Skin-coloured pixels
  from the hand detector are excluded. Only components above a minimum size are kept. Measurements come
  from the ink on the final frame: coverage, stroke width and its variation, continuity, and ink growth.

RED - hand. The hand is found as the largest skin-coloured region (YCrCb thresholds), with its own
  centroid, bounding box and contour. It is not derived from the ink detector. Fingertip count is a rough
  convexity-defect estimate. Finger landmarks and grip posture are not measured.

The annotated video outlines the blue ink and red hand regions, shows a legend, and keeps the original
video separate. Nothing is drawn when a detector is not confident, and the reports say when a measurement
is unavailable.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from app_backend.errors import APIError

INK_HUE_RANGE = (95, 135)      # OpenCV hue units: blue to violet
INK_BLACKHAT_MIN = 25          # gray levels darker than the local background
INK_VALUE_MAX = 160            # dark pixels only
INK_MIN_COMPONENT = 12         # pixels
HAND_MIN_AREA_FRACTION = 0.03  # hand region must cover at least 3% of the frame
HAND_MIN_SOLIDITY = 0.45       # below this the skin region is too fragmented to report
RADIAL_BINS = 72

UNAVAILABLE = {
    "finger_landmarks": "Not measured: no hand-landmark model is installed in this server, so individual finger positions are not reported.",
    "grip_posture": "Not measured: pen-holding posture needs finger landmarks, which are not available here.",
    "pen_tip_tracking": "Not measured: the pen tip cannot be separated from the hand reliably in this pipeline.",
    "pressure": "Not measured: pressure needs a digitizer. Ink darkness in an image is not physical pressure.",
    "grip_force": "Not measured: grip force needs instrumented equipment.",
    "grip_angle": "Not measured: a calibrated grip angle needs a digitizer or instrumented pen.",
    "fitted_spiral_deviation": "Not measured: fitting an ideal spiral needs a calibrated, perspective-corrected view.",
}

# Clients use this to detect a server that still runs older analysis code.
ANALYSIS_VERSION = "ink-hand-v2"

LIMITATIONS = [
    "Pixel measurements depend on camera distance, angle, focus and lighting. They are not calibrated physical units.",
    "Blue ink detection assumes blue or violet pen ink. Black or grey ink is not detected by this colour rule.",
    "Skin detection depends on skin tone and lighting. Objects with similar colour can be included in the red region.",
    "These observations are research descriptors, not a Parkinson's measurement or a diagnosis.",
]


def _open(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise APIError(400, "invalid_spiral_video", "The uploaded video could not be decoded.")
    return cap


def _ink_mask(frame_bgr, skin_mask):
    import cv2
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    hue, value = hsv[..., 0], hsv[..., 2]
    mask = ((blackhat > INK_BLACKHAT_MIN) & (hue >= INK_HUE_RANGE[0]) & (hue <= INK_HUE_RANGE[1]) & (value <= INK_VALUE_MAX))
    mask = mask.astype(np.uint8) * 255
    if skin_mask is not None:
        protected = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8))
        mask[protected > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= INK_MIN_COMPONENT
    return (keep[labels]).astype(np.uint8) * 255


def _hand_region(frame_bgr):
    """Largest skin-coloured region. Returns (mask, contour, area_fraction, solidity, fingertip_estimate) or None."""
    import cv2
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y > 60)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
    if area < HAND_MIN_AREA_FRACTION * frame_area:
        return None
    hull = cv2.convexHull(contour)
    solidity = area / max(float(cv2.contourArea(hull)), 1.0)
    if solidity < HAND_MIN_SOLIDITY:
        return None
    fingertips = 0
    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_idx)
        if defects is not None:
            depth_min = 0.10 * math.sqrt(area)
            fingertips = int(np.sum(defects[:, 3] / 256.0 > depth_min)) + 1
    except cv2.error:
        fingertips = 0
    mask = np.zeros(skin.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask, contour, area / frame_area, solidity, fingertips


def analyze_spiral_video(data: bytes, filename: str, config: dict) -> tuple[bytes, str, dict]:
    """Decode a validated video and return an annotated WebM plus the measurement report."""
    try:
        import cv2
    except ImportError:
        raise APIError(503, "video_decoder_unavailable", "Video analysis is unavailable on this server.") from None

    suffix = Path(filename).suffix.lower()
    temp_root = Path(config["UPLOAD_TEMP_DIR"])
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pd-video-analysis-", dir=temp_root) as work:
        source = Path(work) / f"source{suffix}"
        # VP8 in WebM: the only H.264-free encoder this OpenCV build offers that Chromium plays.
        output = Path(work) / "annotated.webm"
        source.write_bytes(data)
        cap = _open(str(source))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if (fps <= 0 or declared <= 0 or width < 32 or height < 32
                or width > config["SPIRAL_MAX_DIMENSION"] or height > config["SPIRAL_MAX_DIMENSION"]):
            raise APIError(422, "invalid_video_dimensions", "Video frame dimensions or timing metadata are unsupported.")

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise APIError(503, "video_encoder_unavailable", "The server cannot create a playable annotated video.")

        acc = _Accumulator(width, height, fps)
        ink_ema = None
        frames_written = 0
        cap = None
        try:
            cap = _open(str(source))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape[:2] != (height, width):
                    raise APIError(400, "invalid_spiral_video", "The video contains an invalid or inconsistent frame.")
                hand = _hand_region(frame)
                hand_mask = hand[0] if hand is not None else None
                ink = _ink_mask(frame, hand_mask)
                # Light temporal smoothing reduces flicker in the drawn outline. Measurements use the raw mask.
                ink_f = (ink > 0).astype(np.float32)
                ink_ema = ink_f if ink_ema is None else 0.6 * ink_ema + 0.4 * ink_f
                acc.update(ink, hand, frame_index=frames_written)
                writer.write(_annotate(frame, ink_ema > 0.5, hand, frames_written / fps))
                frames_written += 1
        except APIError:
            raise
        except Exception:
            raise APIError(422, "video_processing_failed", "The video could not be processed into an annotated output.") from None
        finally:
            if cap is not None:
                cap.release()
            writer.release()
        if frames_written != declared or not output.exists() or output.stat().st_size == 0:
            raise APIError(422, "video_processing_failed", "The complete video could not be decoded and annotated.")
        check = cv2.VideoCapture(str(output))
        readable, first = check.read()
        out_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        if not readable or first is None or out_count <= 0:
            raise APIError(500, "annotated_video_invalid", "The server did not produce a playable annotated video.")

        result = {
            "analysis_type": "opencv_ink_and_hand_image_measurements_not_a_diagnosis",
            "analysis_version": ANALYSIS_VERSION,
            "video": {"duration_seconds": round(declared / fps, 3), "fps": round(fps, 3), "frame_count": declared,
                      "width": width, "height": height},
            "measurements": acc.report(),
            # Kept for older clients. Pen-tip tracking is not performed in this version.
            "trajectory": {"tracked_points": 0, "tracking_status": "not_performed_pen_tip_tracking_unavailable"},
            "candidate_movement_events": [],
            "visible_appearance_events": [],
            "limitations": list(LIMITATIONS),
            "annotated_video_format": "webm",
            "annotation_legend": {"blue": "Pen ink and strokes (colour and contrast analysis)",
                                  "red": "Hand region (skin-colour analysis; finger positions not measured)"},
        }
        return output.read_bytes(), "webm", result


def _annotate(frame, ink_on, hand, seconds):
    import cv2
    out = frame.copy()
    if ink_on.any():
        ink_u8 = ink_on.astype(np.uint8) * 255
        contours, _ = cv2.findContours(ink_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= INK_MIN_COMPONENT][:200]
        cv2.drawContours(out, contours, -1, (255, 120, 20), 1, cv2.LINE_AA)
    if hand is not None:
        _, contour, _, _, _ = hand
        cv2.drawContours(out, [contour], -1, (40, 40, 230), 2, cv2.LINE_AA)
    # Legend: small swatches with labels, top-left.
    cv2.rectangle(out, (8, 8), (262, 62), (250, 250, 246), -1)
    cv2.rectangle(out, (14, 16), (30, 30), (255, 120, 20), -1)
    cv2.putText(out, "Pen ink / strokes", (38, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 40, 30), 1, cv2.LINE_AA)
    cv2.rectangle(out, (14, 40), (30, 54), (40, 40, 230), -1)
    cv2.putText(out, "Hand region", (38, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 40, 30), 1, cv2.LINE_AA)
    cv2.putText(out, f"t={seconds:.2f}s  image analysis only", (12, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 60, 30), 1, cv2.LINE_AA)
    return out


class _Accumulator:
    """Collects per-frame results for both detectors without sharing their masks or coordinates."""

    def __init__(self, width: int, height: int, fps: float):
        self.width, self.height, self.fps = width, height, fps
        self.ink_counts: list[int] = []
        self.last_ink: np.ndarray | None = None
        self.hand_centroids: list[tuple[float, float]] = []
        self.hand_areas: list[float] = []
        self.hand_solidity: list[float] = []
        self.hand_fingertips: list[int] = []
        self.frames = 0

    def update(self, ink: np.ndarray, hand, frame_index: int) -> None:
        import cv2
        self.frames += 1
        self.ink_counts.append(int(np.count_nonzero(ink)))
        self.last_ink = ink
        if hand is not None:
            mask, contour, area_fraction, solidity, fingertips = hand
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                self.hand_centroids.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
            self.hand_areas.append(area_fraction)
            self.hand_solidity.append(solidity)
            self.hand_fingertips.append(fingertips)

    def report(self) -> dict:
        import cv2
        measured: dict = {}
        estimated: dict = {}
        ink_total = self.last_ink is not None and int(np.count_nonzero(self.last_ink)) >= 30
        unavailable = dict(UNAVAILABLE)

        # BLUE: ink measurements on the final frame.
        if ink_total:
            ink = self.last_ink > 0
            frame_area = float(self.width * self.height)
            measured["ink_pixels_final_frame"] = int(ink.sum())
            measured["ink_coverage_of_frame"] = round(float(ink.sum()) / frame_area, 5)
            distance = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
            widths = distance[ink] * 2.0
            measured["stroke_width_px_median"] = round(float(np.median(widths)), 2)
            estimated["stroke_width_variation"] = round(float(widths.std() / widths.mean()), 3) if widths.mean() > 0 else None
            count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            measured["ink_components"] = int(areas.size)
            estimated["continuity_largest_piece_share"] = round(float(areas.max() / areas.sum()), 3) if areas.size else None
            counts = np.asarray(self.ink_counts, dtype=float)
            if counts.size and counts.max() > 0:
                growth = counts / counts.max()
                estimated["ink_growth_frame_at_half_of_final"] = int(np.argmax(growth >= 0.5)) if np.any(growth >= 0.5) else None
        else:
            measured["ink_pixels_final_frame"] = int(np.count_nonzero(self.last_ink)) if self.last_ink is not None else 0
            estimated["ink_status"] = "insufficient_blue_ink_detected"

        # RED: hand measurements, computed only from the hand detector.
        hand_fraction = len(self.hand_centroids) / max(self.frames, 1)
        measured["frames_with_hand_region"] = round(hand_fraction, 3)
        if len(self.hand_centroids) >= max(10, int(0.5 * self.frames)):
            centres = np.asarray(self.hand_centroids, dtype=float)
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1) if len(centres) > 1 else np.array([0.0])
            diagonal = math.hypot(self.width, self.height)
            estimated["hand_stability_median_step"] = round(float(np.median(steps) / diagonal), 5)
            estimated["hand_region_area_median"] = round(float(np.median(self.hand_areas)), 4)
            estimated["hand_region_solidity_median"] = round(float(np.median(self.hand_solidity)), 3)
            estimated["fingertips_visible_estimate_median"] = int(np.median(self.hand_fingertips))
        else:
            estimated["hand_status"] = "insufficient_hand_detection"
        if len(self.hand_centroids) == 0:
            unavailable.setdefault("hand_region", "No hand region passed the confidence checks in this video.")

        return {"directly_measured_image_features": measured,
                "estimated_visual_proxies": estimated,
                "unavailable_measurements": unavailable}
