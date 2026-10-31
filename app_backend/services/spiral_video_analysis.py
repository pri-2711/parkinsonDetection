"""Image-based observations from a spiral-writing video (visual analysis only, not a diagnostic model).

Three analyses run on the same video. Each one is a separate detector and is drawn separately:

INK (spiral_ink.py). The first frame is the reference. Every later frame is aligned to it with paper features
  (ORB and a RANSAC homography), so a moving camera does not look like new ink. Ink is new only when it is stable
  across frames and clearly darker than the printed mark at the same place on the reference paper. Printed
  text, boxes and ruling are therefore not counted. The new ink is accumulated in reference coordinates and
  drawn in blue on each frame. Frames where the paper cannot be aligned add no ink and are reported.

HAND (hand_landmarks.py). MediaPipe Hands gives 21 landmarks per frame, run in a separate environment
  (HAND_LANDMARK_PYTHON, or the repository's .venv-mediapipe). The hand outline is drawn in red and the
  landmarks in green. Finger, grip and movement values use only frames where the landmarks are visible.

SPIRAL (spiral_ink.py). The accumulated new ink is separated from other handwriting and fitted to an
  Archimedean spiral: turn spacing, deviation, radius variation and angular coverage. Values are in reference
  frame pixels, without perspective correction.

The annotated WebM keeps the original duration and frames, and shows the three overlays with a legend. Anything
not detected reliably is left out and listed as unavailable with a reason.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from app_backend.errors import APIError
from app_backend.services import hand_landmarks as hands_mod
from app_backend.services import spiral_ink as ink_mod

# Clients use this to detect a server that still runs older analysis code.
ANALYSIS_VERSION = "ink-hand-v4"

LIMITATIONS = [
    "Pixel measurements depend on camera distance, angle, focus and lighting. They are not calibrated physical units.",
    "Ink darkness and stroke width are image measurements. They are not pen pressure.",
    "New ink is found against the first frame. Frames where the paper cannot be aligned add no ink, so the blue overlay can skip them.",
    "The pen body and tip are not tracked. Drawing timing uses the centroid of newly drawn ink, which lags the pen by a few frames.",
    "Hand landmarks come from a general hand model. Fingers cut off by the frame edge are not measured.",
    "The pen-holding estimate is a heuristic from one camera view and is not a validated grip classification.",
    "The spiral fit assumes a flat view of the paper. No perspective correction is applied.",
    "These observations are research descriptors, not a Parkinson's measurement or a diagnosis.",
]

SECTION_TITLES = {
    "ink_and_strokes": "Ink & Stroke Analysis",
    "hand_and_grip": "Hand & Grip Analysis",
    "spiral_geometry": "Spiral Geometry Analysis",
}

INK_COLOUR_NAMES = {
    "blue_violet": "blue / violet",
    "red": "red",
    "black_or_grey": "black / grey",
    "other_colour": "other colour",
}

# Overlay colours, BGR. Each overlay is drawn by its own detector.
BLUE = (255, 120, 20)
GREEN = (60, 200, 60)
RED = (40, 40, 230)
# Share of frames the paper must align on before the ink values are considered complete.
MIN_ALIGNED_SHARE = 0.8


def _open(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise APIError(400, "invalid_spiral_video", "The uploaded video could not be decoded.")
    return cap


def _skin_mask(frame_bgr):
    """Skin-coloured pixels, excluded from ink so a hand is never read as writing. Not drawn."""
    import cv2
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y > 60)).astype(np.uint8) * 255
    return cv2.dilate(skin, np.ones((5, 5), np.uint8)) > 0


def _item(label, value, unit, explanation, kind, reliability="moderate"):
    if reliability not in ("high", "moderate", "low"):
        reliability = "low"
    return {"label": label, "value": value, "unit": unit, "explanation": explanation,
            "kind": kind, "reliability": reliability}


def _ink_items(ever, labels, darkness, width, height, fps, ink_counts):
    """Ink section from the accumulated spiral ink (reference coordinates). Returns (items, unavailable)."""
    frame_area = float(width * height)
    measured, estimated, reason = ink_mod.ink_measurements(ever, labels, darkness, min(width, height), frame_area)
    items: list = []
    unavailable: list = []
    if reason:
        return items, [{"label": "Ink measurements", "reason": reason}]
    items.append(_item("Ink coverage", round(measured["ink_coverage_of_frame"] * 100, 2), "% of frame",
                       "Share of the frame covered by the newly drawn spiral ink.", "measured"))
    shares = measured["ink_colour_share"]
    colour_text = ", ".join(f"{INK_COLOUR_NAMES[k]} {round(v * 100)}%" for k, v in shares.items() if v >= 0.02)
    items.append(_item("Ink colour", colour_text or "not classified", "",
                       "Share of ink pixels by colour family. Black and grey ink is detected by darkness.", "measured"))
    items.append(_item("Ink darkness (median)", measured["ink_darkness_median"], "levels (0-255)",
                       "How much darker the ink is than the paper. This is not pen pressure.", "measured"))
    if "stroke_width_px_median" in measured:
        items.append(_item("Stroke width (median)", measured["stroke_width_px_median"], "px",
                           "Typical width of the drawn line, in reference-frame pixels.", "measured"))
        items.append(_item("Stroke width (spread)", measured["stroke_width_px_std"], "px",
                           "Standard deviation of the stroke width.", "measured"))
    if estimated.get("stroke_width_variation") is not None:
        items.append(_item("Stroke-width consistency", estimated["stroke_width_variation"], "variation (CV)",
                           "Lower values mean a more even line width. Estimated from the mask, not pressure.",
                           "estimated", "low"))
    items.append(_item("Stroke path length", round(measured["stroke_path_length_px"] / max(width, height), 2),
                       "frame sides", "Approximate length of the drawn line, relative to the frame size.", "estimated", "low"))
    items.append(_item("Separate ink pieces", measured["ink_components"], "pieces",
                       "Strokes split into pieces by gaps. More pieces can mean more breaks.", "measured"))
    items.append(_item("Largest piece share", estimated["continuity_largest_piece_share"], "share",
                       "Share of ink in the largest connected piece. Closer to 1 means more continuous.", "estimated", "low"))
    items.append(_item("Stroke endpoints", estimated["stroke_endpoints"], "ends",
                       "Ends of the thinned stroke. Each break adds two ends.", "estimated", "low"))
    half = next((i for i, count in enumerate(ink_counts) if count >= 0.5 * max(ink_counts)), None) if ink_counts else None
    if half is not None and fps > 0:
        items.append(_item("Half the ink drawn by", round(half / fps, 2), "seconds",
                           "Time at which half of the final ink had appeared.", "estimated", "low"))
    items.append(_item("Ink density in bounding box", measured["ink_density_in_bounding_box"], "share",
                       "Ink pixels divided by the area of the ink's bounding box.", "measured"))
    return items, unavailable


def _timing_items(events, fps, shape):
    estimated, reason = ink_mod.drawing_timeline(events, fps, shape)
    if reason:
        return [], [{"label": "Drawing timing", "reason": reason}]
    rel = "low"
    items = [
        _item("Drawing duration", estimated["drawing_duration_seconds"], "seconds",
              "Time from the first to the last newly drawn ink. Estimated from the ink, not the pen tip.", "estimated", rel),
        _item("Pauses while drawing", estimated["pauses"], "pauses",
              "Gaps of at least half a second with no new ink. A hand that covers the drawing also shows as a pause.",
              "estimated", rel),
        _item("Ink-front path length", estimated["front_path_length_frame_sides"], "frame sides",
              "Path followed by the centroid of new ink. A proxy for the drawing path, not a tracked pen tip.",
              "estimated", rel),
    ]
    if "front_speed_median_frame_sides_per_s" in estimated:
        items.append(_item("Ink-front speed (median)", estimated["front_speed_median_frame_sides_per_s"],
                           "frame sides per second", "Median speed of the ink-front path between drawn frames.",
                           "estimated", rel))
    unavailable = [{"label": "Pen tip trajectory", "reason": hands_mod.NOT_MEASURED_PEN_TIP},
                   {"label": "Smoothness of the pen path",
                    "reason": "Not measured: smoothness of a pen path needs a tracked pen tip."}]
    return items, unavailable


def _hand_items(hand_measured, hand_estimated, hand_unavailable, fps):
    items: list = []
    unavailable = [{"label": _HAND_LABELS.get(k, k), "reason": v} for k, v in hand_unavailable.items()]
    if "frames_with_hand_detected" in hand_measured:
        items.append(_item("Frames with a detected hand", round(hand_measured["frames_with_hand_detected"] * 100, 1),
                           "% of frames", "Share of frames where the hand model found a hand.", "measured",
                           hand_measured.get("hand_detection_confidence", "low")))
    if "handedness" in hand_measured:
        items.append(_item("Hand used", hand_measured["handedness"], "", "Most frequent hand label.", "measured",
                           hand_measured.get("hand_detection_confidence", "low")))
    rel = hand_measured.get("hand_detection_confidence", "low")
    for key, label in (("thumb", "Thumb"), ("index", "Index finger"), ("middle", "Middle finger"),
                       ("ring", "Ring finger"), ("little", "Little finger")):
        key_name = f"{key}_tip_visible_share"
        if key_name in hand_measured:
            items.append(_item(f"{label} tip visible", round(hand_measured[key_name] * 100, 1), "% of frames",
                               "Share of frames where this fingertip is inside the video frame.", "measured", rel))
    for key, label in (
        ("index_flexion_deg_median", "Index finger bend"),
        ("middle_flexion_deg_median", "Middle finger bend"),
        ("thumb_flexion_deg_median", "Thumb bend"),
    ):
        if key in hand_estimated:
            items.append(_item(label, hand_estimated[key], "degrees",
                               "Median bend at the joint, from the landmark angle in the image. 0 is straight.",
                               "estimated", rel))
    if "thumb_index_distance_share_of_hand" in hand_estimated:
        items.append(_item("Thumb-index distance", hand_estimated["thumb_index_distance_share_of_hand"],
                           "share of hand length", "Distance between thumb and index fingertips. Smaller means a closer pinch.",
                           "estimated", rel))
    if "hand_orientation_deg_median" in hand_estimated:
        items.append(_item("Hand orientation", hand_estimated["hand_orientation_deg_median"], "degrees",
                           "Direction of the hand in the image. 0 means fingers point up.", "estimated", rel))
    if "wrist_path_length_hand_scales" in hand_measured and fps > 0:
        items.append(_item("Wrist travel", hand_measured["wrist_path_length_hand_scales"], "hand lengths",
                           "Total wrist movement during the video, relative to hand size.", "measured", rel))
        items.append(_item("Wrist net displacement", hand_measured["wrist_net_displacement_hand_scales"],
                           "hand lengths", "Distance from the first to the last wrist position.", "measured", rel))
    if "landmark_motion_median_per_frame" in hand_estimated:
        items.append(_item("Landmark stability", hand_estimated["landmark_motion_median_per_frame"],
                           "hand lengths per frame", "Lower means steadier landmarks between frames.", "estimated", rel))
    grip = hand_estimated.get("grip_configuration")
    if grip:
        explanation = (f"{grip['note']} Usable frames: {grip['usable_frames']} "
                       f"({round(grip['usable_share_of_frames'] * 100)}% of frames).")
        items.append(_item("Pen-holding estimate", grip["label"], "", explanation, "estimated",
                           grip["estimate_confidence"]))
    return items, unavailable


_HAND_LABELS = {
    "hand_landmarks": "Hand landmarks",
    "grip_configuration": "Pen-holding estimate",
    "finger_occlusion": "Per-finger occlusion",
    "pen_tip_tracking": "Pen tip tracking",
    "finger_pressure": "Pressure and grip force",
}


def _spiral_items(fit):
    if isinstance(fit, str):
        return [], [{"label": "Spiral geometry", "reason": fit}]
    items = [
        _item("Turns detected", fit["turns_detected"], "turns", "How many turns the ink spans around its centre.", "estimated", "moderate"),
        _item("Turn spacing", fit["turn_spacing_px"], "px", "Distance between neighbouring turns of the fitted spiral.", "estimated", "moderate"),
        _item("Deviation from fitted spiral", fit["fit_rms_deviation_px"], "px (RMS)",
              "How far the ink strays from the best fitted spiral. Not perspective-corrected.", "estimated", "moderate"),
        _item("Radius variation", round(fit["radius_variation_share_of_spacing"] * 100, 1), "% of turn spacing",
              "Spread of ink around the fitted spiral, relative to turn spacing.", "estimated", "moderate"),
        _item("Angular coverage", round(fit["angular_coverage"] * 100, 1), "% of directions",
              "Share of directions around the centre that contain ink.", "estimated", "moderate"),
        _item("Centre", f"{fit['centre_px'][0]:.0f}, {fit['centre_px'][1]:.0f}", "px (x, y)",
              "Centre of the spiral in reference-frame pixels (the first frame of the video).", "estimated", "moderate"),
    ]
    return items, []


def analyze_spiral_video(data: bytes, filename: str, config: dict) -> tuple[bytes, str, dict]:
    """Decode a validated video and return an annotated WebM plus the analysis report."""
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

        # Hand landmarks first, in a separate process. Frames are aligned by index.
        landmark_frames, _, hand_reason = hands_mod.run_worker(str(source), config)
        if landmark_frames is None:
            hands_list = [None] * declared
        else:
            hands_list = hands_mod.hand_frames(landmark_frames, width, height)
            hands_list = (hands_list + [None] * declared)[:declared]

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise APIError(503, "video_encoder_unavailable", "The server cannot create a playable annotated video.")

        # Pass 2: new ink in reference coordinates, hand overlay and annotated frames.
        tracker = ink_mod.InkTracker((height, width))
        accumulator = ink_mod.InkAccumulator((height, width))
        ink_label = np.zeros((height, width), np.uint8)
        ink_dark = np.zeros((height, width), np.float32)
        ink_counts: list[int] = []
        events: list[tuple[int, float, float]] = []
        aligner = None
        reference_marks = reference_paper = None
        aligned_frames = 0
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
                hand = hands_list[frames_written]
                excluded = _skin_mask(frame)
                hand_area = hands_mod.hand_mask(frame.shape, hand)
                if hand_area is not None:
                    excluded |= hand_area
                dark, paper = ink_mod.frame_maps(frame)
                # Ruling is removed by the reference subtraction below, not by line detection here.
                candidate = ink_mod.candidate_ink(dark, paper, excluded, short_side=min(width, height), remove_lines=False)
                labels = ink_mod.colour_labels(frame, candidate)

                # The first frame is the reference: its printed marks are the baseline that new ink must exceed.
                if aligner is None:
                    aligner = ink_mod.PaperAligner(frame, paper & ~excluded)
                    reference_marks = ink_mod.reference_marks(dark)
                    reference_paper = paper
                    homography = np.eye(3)
                else:
                    homography, _inliers = aligner.homography(frame, paper, excluded)

                ink_on = None
                if homography is not None:
                    aligned_frames += 1
                    # Persistence is judged in reference coordinates, so a moving camera does not erase drawn ink.
                    cand_ref, labels_ref, dark_ref = ink_mod.to_reference(
                        candidate, labels, dark, homography, (height, width))
                    stable_ref = tracker.update(cand_ref)
                    fresh = ink_mod.new_ink(stable_ref, dark_ref, reference_marks, reference_paper)
                    ink_label[fresh] = labels_ref[fresh]
                    ink_dark = np.where(fresh, np.maximum(ink_dark, dark_ref), ink_dark)
                    accepted = accumulator.update(fresh)
                    count = int(np.count_nonzero(accepted))
                    if count >= ink_mod.MIN_FRONT_PIXELS:
                        ys, xs = np.nonzero(accepted)
                        events.append((frames_written, float(xs.mean()), float(ys.mean())))

                    ever = accumulator.ever
                    ink_on = ink_mod.project_mask(ever, homography, frame.shape) if ever.any() else None
                ink_counts.append(int(np.count_nonzero(accumulator.ever)))
                writer.write(_annotate(frame, ink_on, hand, frames_written / fps,
                                       aligned=homography is not None))
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

        # The spiral is separated from any other new handwriting. Stroke and spiral values use only the spiral.
        ever = accumulator.ever
        spiral_mask, fit, ignored_pixels = ink_mod.isolate_spiral(ever) if ever.any() else (
            None, "No new ink was drawn on the paper, so no spiral was fitted.", 0)
        if spiral_mask is None:
            ink_items, ink_unavailable = [], [{"label": "Ink and stroke values",
                                               "reason": f"No spiral was isolated to measure. {fit}"}]
        else:
            ink_items, ink_unavailable = _ink_items(spiral_mask, ink_label, ink_dark, width, height, fps, ink_counts)
            if ignored_pixels > 0:
                ink_items.append(_item("Other new ink ignored", ignored_pixels, "ink pixels",
                                       "New ink that is not part of the spiral, such as other writing. Not measured.",
                                       "measured", "moderate"))
        aligned_share = aligned_frames / declared
        ink_items.insert(0, _item("Frames aligned to the first frame", round(aligned_share * 100, 1), "% of frames",
                                  "Share of frames where the paper matched the first frame. Frames that did not match add no ink.",
                                  "measured", "high" if aligned_share >= MIN_ALIGNED_SHARE else "low"))
        timing_items, timing_gaps = _timing_items(events, fps, (height, width))
        ink_items.extend(timing_items)
        ink_unavailable.extend(timing_gaps)

        if landmark_frames is None:
            hand_measured, hand_estimated, hand_unavailable = hands_mod.not_run_metrics(hand_reason)
        else:
            hand_measured, hand_estimated, hand_unavailable = hands_mod.hand_metrics(hands_list, declared)
        hand_items, hand_gaps = _hand_items(hand_measured, hand_estimated, hand_unavailable, fps)

        spiral_items, spiral_gaps = _spiral_items(fit)

        sections = {
            "ink_and_strokes": {"title": SECTION_TITLES["ink_and_strokes"], "items": ink_items, "unavailable": ink_unavailable},
            "hand_and_grip": {"title": SECTION_TITLES["hand_and_grip"], "items": hand_items, "unavailable": hand_gaps},
            "spiral_geometry": {"title": SECTION_TITLES["spiral_geometry"], "items": spiral_items, "unavailable": spiral_gaps},
        }
        result = {
            "analysis_type": "opencv_ink_mediapipe_hand_and_spiral_measurements_not_a_diagnosis",
            "analysis_version": ANALYSIS_VERSION,
            "video": {"duration_seconds": round(declared / fps, 3), "fps": round(fps, 3), "frame_count": declared,
                      "width": width, "height": height},
            "hand_landmarks": {
                "available": landmark_frames is not None,
                "model": "MediaPipe Hands (hand_landmark_full.tflite bundled in the MediaPipe 0.10.21 wheel)",
                "frames_with_hand": sum(1 for h in hands_list if h is not None),
                "frames_total": declared,
                "reason": hand_reason,
            },
            "sections": sections,
            "limitations": list(LIMITATIONS),
            "annotated_video_format": "webm",
            "annotation_legend": {
                "blue": "Newly drawn spiral ink (not printed text, ruling or marks already on the first frame)",
                "green": "Hand landmarks (MediaPipe, drawn when the hand is detected)",
                "red": "Hand outline (convex hull of the landmarks)",
            },
        }
        return output.read_bytes(), "webm", result


def _annotate(frame, ink_on, hand, seconds, aligned=True):
    """One frame with three independent overlays: blue ink, green landmarks, red hand outline."""
    import cv2
    out = frame.copy()
    if ink_on is not None and ink_on.any():
        ink_u8 = ink_on.astype(np.uint8) * 255
        contours, _ = cv2.findContours(ink_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Thin strokes enclose almost no area, so keep outlines by length, not area.
        contours = [c for c in contours if cv2.arcLength(c, False) >= 8][:300]
        cv2.drawContours(out, contours, -1, BLUE, 1, cv2.LINE_AA)
    hands_mod.draw_hand(out, hand)
    # Legend: swatches with labels, top-left.
    cv2.rectangle(out, (8, 8), (292, 84), (250, 250, 246), -1)
    cv2.rectangle(out, (14, 16), (30, 30), BLUE, -1)
    cv2.putText(out, "New spiral ink", (38, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 40, 30), 1, cv2.LINE_AA)
    cv2.rectangle(out, (14, 40), (30, 54), GREEN, -1)
    cv2.putText(out, "Hand landmarks", (38, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 40, 30), 1, cv2.LINE_AA)
    cv2.rectangle(out, (14, 64), (30, 78), RED, -1)
    cv2.putText(out, "Hand outline", (38, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 40, 30), 1, cv2.LINE_AA)
    status = f"t={seconds:.2f}s" + ("" if aligned else "  paper not aligned: no ink drawn")
    cv2.putText(out, status, (12, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 60, 30), 1, cv2.LINE_AA)
    return out
