"""Explainable video-only spiral trace observations (not a diagnostic model)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from app_backend.errors import APIError


def analyze_spiral_video(data: bytes, filename: str, config: dict) -> tuple[bytes, str, dict]:
    """Decode a validated video, annotate visible-change candidates and return encoded output."""
    try:
        import cv2
    except ImportError:
        raise APIError(503, "video_decoder_unavailable", "Video analysis is unavailable on this server.") from None

    suffix = Path(filename).suffix.lower()
    temp_root = Path(config["UPLOAD_TEMP_DIR"])
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pd-video-analysis-", dir=temp_root) as work:
        source = Path(work) / f"source{suffix}"
        output = Path(work) / "annotated.mp4"
        source.write_bytes(data)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            cap.release()
            raise APIError(400, "invalid_spiral_video", "The uploaded video could not be decoded.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (fps <= 0 or declared <= 0 or width < 32 or height < 32
                or width > config["SPIRAL_MAX_DIMENSION"] or height > config["SPIRAL_MAX_DIMENSION"]):
            cap.release()
            raise APIError(422, "invalid_video_dimensions", "Video frame dimensions or timing metadata are unsupported.")

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise APIError(503, "video_encoder_unavailable", "The server cannot create a playable annotated video.")

        previous_gray = None
        previous_descriptors = None
        previous_keypoints = None
        orb = cv2.ORB_create(nfeatures=600)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        previous_point = None
        points: list[tuple[int, float, float]] = []
        blob_areas: list[float] = []
        blob_darkness: list[float] = []
        red_events: list[dict] = []
        blue_events: list[dict] = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape[:2] != (height, width):
                    raise APIError(400, "invalid_spiral_video", "The video contains an invalid or inconsistent frame.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Local contrast normalization reduces sensitivity to gradual lighting changes.
                normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                keypoints, descriptors = orb.detectAndCompute(normalized, None)
                box = None
                if previous_gray is not None:
                    # Estimate camera motion from RANSAC-consistent scene features. Phase correlation
                    # alone can mistake the moving pen tip for camera movement in sparse scenes.
                    matrix = np.float32([[1, 0, 0], [0, 1, 0]])
                    if descriptors is not None and previous_descriptors is not None:
                        pairs = matcher.knnMatch(descriptors, previous_descriptors, k=2)
                        matches = [pair[0] for pair in pairs if len(pair) >= 2 and pair[0].distance < 0.72 * pair[1].distance]
                        if len(matches) >= 8:
                            current_points = np.float32([keypoints[m.queryIdx].pt for m in matches])
                            previous_points = np.float32([previous_keypoints[m.trainIdx].pt for m in matches])
                            estimated, inliers = cv2.estimateAffinePartial2D(
                                current_points, previous_points, method=cv2.RANSAC,
                                ransacReprojThreshold=3.0, maxIters=1000, confidence=0.98,
                            )
                            if estimated is not None and inliers is not None and int(inliers.sum()) >= 6 and float(inliers.mean()) >= 0.55:
                                matrix = np.float32(estimated)
                    stabilized = cv2.warpAffine(normalized, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                    delta = cv2.absdiff(previous_gray, stabilized)
                    # Candidate is newly visible dark local structure, not the static guide.
                    changed = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)[1]
                    dark = cv2.threshold(stabilized, 165, 255, cv2.THRESH_BINARY_INV)[1]
                    mask = cv2.bitwise_and(changed, dark)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    candidates = [c for c in contours if 3 <= cv2.contourArea(c) <= width * height * 0.005]
                    if candidates:
                        def candidate_score(contour):
                            x, y, w, h = cv2.boundingRect(contour)
                            center = (x + w / 2, y + h / 2)
                            distance = math.dist(center, previous_point[1:]) if previous_point else 0
                            return cv2.contourArea(contour) / (1 + distance / max(width, height) * 30)
                        contour = max(candidates, key=candidate_score)
                        x, y, w, h = cv2.boundingRect(contour)
                        moments = cv2.moments(contour)
                        if moments["m00"]:
                            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                            point = (frame_index, cx, cy)
                            points.append(point)
                            previous_point = point
                            area = float(cv2.contourArea(contour))
                            roi = stabilized[y:y + h, x:x + w]
                            darkness = float(np.mean(roi)) if roi.size else 255.0
                            blob_areas.append(area)
                            blob_darkness.append(darkness)
                            box = (x, y, w, h, area, darkness)
                    # Keep the reference frame in its original coordinates so the next
                    # current-to-previous affine transform matches the stored keypoints.
                    previous_gray = normalized
                else:
                    previous_gray = normalized
                previous_keypoints = keypoints
                previous_descriptors = descriptors

                timestamp = frame_index / fps
                if box:
                    x, y, w, h, area, darkness = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
                    if len(blob_areas) >= 5:
                        baseline_area = float(np.median(blob_areas[:-1]))
                        baseline_dark = float(np.median(blob_darkness[:-1]))
                        changed_appearance = (baseline_area > 0 and abs(area - baseline_area) / baseline_area > 0.6) or abs(darkness - baseline_dark) > 35
                        if changed_appearance:
                            blue_events.append({"timestamp_seconds": round(timestamp, 3), "observation": "visible stroke appearance changed"})
                            cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 0, 0), 2)

                # Candidate high-frequency residual is an image-space observation, not tremor frequency diagnosis.
                if len(points) >= max(8, int(fps * 0.35)):
                    window = points[-max(8, int(fps * 0.35)):]
                    xy = np.asarray([[p[1], p[2]] for p in window], dtype=float)
                    if len(xy) >= 8:
                        smooth = np.column_stack([np.convolve(xy[:, i], np.ones(5) / 5, mode="same") for i in range(2)])
                        residual = np.linalg.norm(xy[2:-2] - smooth[2:-2], axis=1)
                        diffs = np.diff(xy, axis=0)
                        reversals = int(np.sum(np.sum(diffs[1:] * diffs[:-1], axis=1) < 0))
                        if float(np.sqrt(np.mean(residual ** 2))) > 2.0 and reversals >= 3:
                            if box:
                                x, y, w, h, *_ = box
                                cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)
                            elif previous_point:
                                px, py = int(previous_point[1]), int(previous_point[2])
                                cv2.rectangle(frame, (px - 12, py - 12), (px + 12, py + 12), (0, 0, 255), 2)
                            event = {"timestamp_seconds": round(timestamp, 3), "observation": "candidate high-frequency image-space movement", "residual_rms_px": round(float(np.sqrt(np.mean(residual ** 2))), 2)}
                            if not red_events or timestamp - red_events[-1]["timestamp_seconds"] >= 0.4:
                                red_events.append(event)
                cv2.putText(frame, f"t={timestamp:.2f}s | visual analysis only", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 60, 30), 2, cv2.LINE_AA)
                writer.write(frame)
                frame_index += 1
        except APIError:
            raise
        except Exception:
            raise APIError(422, "video_processing_failed", "The video could not be processed into an annotated output.") from None
        finally:
            cap.release()
            writer.release()
        if frame_index == 0 or frame_index != declared or not output.exists() or output.stat().st_size == 0:
            raise APIError(422, "video_processing_failed", "The complete video could not be decoded and annotated.")
        # Confirm the output is actually decodable before returning it for persistent storage.
        check = cv2.VideoCapture(str(output))
        readable, first = check.read()
        out_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        if not readable or first is None or out_count <= 0:
            raise APIError(500, "annotated_video_invalid", "The server did not produce a playable annotated video.")

        if len(points) >= 2:
            xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
            deltas = np.diff(xy, axis=0)
            segment = np.linalg.norm(deltas, axis=1)
            path_length = float(segment.sum())
            times = np.diff([p[0] for p in points]) / fps
            speeds = segment / np.maximum(times, 1 / fps)
            turns = []
            for a, b in zip(deltas[:-1], deltas[1:]):
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom:
                    turns.append(math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1, 1)))))
            speed_cv = float(np.std(speeds) / np.mean(speeds)) if np.mean(speeds) > 0 else None
            trajectory = {"tracked_points": len(points), "path_length_px": round(path_length, 2),
                          "mean_speed_px_per_second": round(float(np.mean(speeds)), 2),
                          "speed_coefficient_of_variation": round(speed_cv, 4) if speed_cv is not None else None,
                          "mean_direction_change_degrees": round(float(np.mean(turns)), 2) if turns else None}
        else:
            trajectory = {"tracked_points": len(points), "tracking_status": "insufficient_visible_motion_for_trajectory"}
        result = {
            "analysis_type": "opencv_visual_baseline_not_a_diagnosis",
            "video": {"duration_seconds": round(frame_index / fps, 3), "fps": round(fps, 3), "frame_count": frame_index,
                      "width": width, "height": height},
            "trajectory": trajectory,
            "candidate_movement_events": red_events[:100],
            "visible_appearance_events": blue_events[:100],
            "limitations": ["Pixel measurements depend on camera distance, angle, frame rate, focus, lighting, and camera stability.",
                            "Candidate movement is not a Parkinson's tremor measurement or diagnosis.",
                            "Blue annotations describe visible appearance changes only; they do not measure pen pressure.",
                            "Ordinary video cannot recover invisible pen-up movement, calibrated pressure, or grip angle."],
            "annotated_video_format": "mp4",
        }
        return output.read_bytes(), "mp4", result
