from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app_backend.errors import APIError
from app_backend.services import hand_landmarks as hands
from app_backend.services import spiral_ink as ink
from app_backend.services import spiral_video_analysis as video

PAPER = (235, 235, 230)  # BGR


def _paper(height=240, width=320, level=PAPER):
    return np.full((height, width, 3), level, np.uint8)


def _candidates(frame, excluded=None):
    dark, paper = ink.frame_maps(frame)
    return ink.candidate_ink(dark, paper, excluded, short_side=min(frame.shape[:2]))


def _spiral_points(centre, r0, growth, turns, steps=600):
    points = []
    for i in range(steps):
        theta = turns * 2 * math.pi * i / (steps - 1)
        r = r0 + growth * theta
        points.append((int(round(centre[0] + r * math.cos(theta))), int(round(centre[1] + r * math.sin(theta)))))
    return np.array(points, np.int32)


class InkColourTests(unittest.TestCase):
    """Each ink colour is found by contrast with the paper, and labelled by family."""

    def _stroke_colour(self, bgr, thickness=3):
        frame = _paper()
        cv2.ellipse(frame, (160, 200), (150, 120), 0, 200, 340, bgr, thickness, cv2.LINE_AA)  # a curved stroke
        candidate = _candidates(frame)
        labels = ink.colour_labels(frame, candidate)
        return candidate, labels

    def test_blue_ink_is_found_and_labelled_blue(self):
        candidate, labels = self._stroke_colour((150, 60, 30))
        self.assertGreater(int(candidate.sum()), 150)
        self.assertGreater(int(np.sum(labels == ink.COLOUR_BLUE)), int(0.8 * candidate.sum()))

    def test_black_ink_is_found_and_labelled_neutral(self):
        candidate, labels = self._stroke_colour((25, 25, 25))
        self.assertGreater(int(candidate.sum()), 150)
        self.assertGreater(int(np.sum(labels == ink.COLOUR_NEUTRAL)), int(0.8 * candidate.sum()))

    def test_red_ink_is_found_and_labelled_red(self):
        candidate, labels = self._stroke_colour((40, 40, 200))
        self.assertGreater(int(candidate.sum()), 150)
        self.assertGreater(int(np.sum(labels == ink.COLOUR_RED)), int(0.8 * candidate.sum()))

    def test_other_coloured_ink_is_found_and_labelled_other(self):
        candidate, labels = self._stroke_colour((60, 170, 60))
        self.assertGreater(int(candidate.sum()), 150)
        self.assertGreater(int(np.sum(labels == ink.COLOUR_OTHER)), int(0.8 * candidate.sum()))

    def test_plain_paper_and_smooth_shading_give_no_ink(self):
        self.assertEqual(int(_candidates(_paper()).sum()), 0)
        ramp = np.tile(np.linspace(170, 250, 320, dtype=np.uint8)[None, :, None], (240, 1, 3))
        self.assertLess(int(_candidates(ramp).sum()), 20)


class RulingAndObjectRejectionTests(unittest.TestCase):
    def test_printed_ruling_is_rejected_and_the_spiral_is_kept(self):
        # Proportions follow a real recording, where the spiral is much larger than the ruling spacing.
        frame = _paper(400, 400)
        for y in range(20, 400, 22):
            cv2.line(frame, (0, y), (399, y), (150, 150, 150), 1)  # printed ruling, darker than the faint shading
        spiral = _spiral_points((200, 200), 12, 8.0, 3.5)
        cv2.polylines(frame, [spiral], False, (150, 60, 30), 2, cv2.LINE_AA)
        candidate = _candidates(frame)
        ruling_only = np.zeros_like(candidate)
        for y in range(20, 400, 22):
            ruling_only[y - 1:y + 2, :] = True
        spiral_mask = np.zeros_like(candidate)
        cv2.polylines(spiral_mask.view(np.uint8), [spiral], False, 1, 3)
        ruling_hits = int((candidate & ruling_only).sum())
        self.assertLess(ruling_hits, 0.1 * int(ruling_only.sum()))
        self.assertGreater(int((candidate & spiral_mask.astype(bool)).sum()), 0.5 * int(spiral_mask.sum()))

    def test_thick_pen_body_is_not_ink(self):
        frame = _paper()
        cv2.rectangle(frame, (60, 60), (260, 90), (30, 30, 30), -1)  # a 30 px pen body
        candidate = _candidates(frame)
        self.assertLess(int(candidate[70:80, 100:220].sum()), 20)

    def test_skin_and_hand_are_excluded_from_ink(self):
        frame = _paper()
        cv2.ellipse(frame, (80, 200), (70, 70), 0, 200, 340, (150, 60, 30), 3, cv2.LINE_AA)
        cv2.ellipse(frame, (230, 150), (45, 60), 0, 0, 360, (120, 150, 210), -1)
        excluded = video._skin_mask(frame)
        candidate = _candidates(frame, excluded)
        self.assertEqual(int((candidate & excluded).sum()), 0)
        self.assertGreater(int(candidate[:, :160].sum()), 60)

    def test_stroke_is_not_ink_until_it_is_seen_in_several_frames(self):
        frame = _paper()
        cv2.ellipse(frame, (160, 200), (150, 120), 0, 200, 340, (150, 60, 30), 3, cv2.LINE_AA)
        candidate = _candidates(frame)
        tracker = ink.InkTracker(candidate.shape)
        blank = np.zeros_like(candidate)
        stable_after_one = tracker.update(candidate)
        self.assertEqual(int(stable_after_one.sum()), 0)
        for _ in range(ink.PERSIST_NEEDED - 1):
            stable = tracker.update(candidate)
        self.assertGreater(int(stable.sum()), 100)
        # A mark that appears only briefly, like a passing pen, is not kept.
        tracker = ink.InkTracker(candidate.shape)
        for _ in range(3):
            tracker.update(candidate)
        for _ in range(ink.PERSIST_WINDOW + 2):
            result = tracker.update(blank)
        self.assertEqual(int(result.sum()), 0)


def _textured_paper(height=400, width=400, seed=7):
    """Paper with faint ruling, printed blue text, a printed box and speckle that gives paper features."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), PAPER, np.uint8)
    for y in range(30, height, 24):
        cv2.line(page, (0, y), (width - 1, y), (140, 140, 140), 1)
    cv2.putText(page, "AI Project", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (150, 60, 30), 2, cv2.LINE_AA)
    cv2.rectangle(page, (250, 20), (380, 90), (150, 60, 30), 2)
    for _ in range(300):
        x, y = rng.integers(0, width), rng.integers(0, height)
        cv2.circle(page, (int(x), int(y)), 1, (170, 170, 170), -1)
    return page


def _camera_matrix(t):
    """A small, steady camera drift: a slight rotation and a few pixels of shift per frame."""
    matrix = cv2.getRotationMatrix2D((200.0, 200.0), 0.15 * t, 1.0)
    matrix[0, 2] += 0.8 * t
    matrix[1, 2] += 0.5 * t
    return matrix


class MovingPageInkTests(unittest.TestCase):
    """New ink is found against the first frame, so printed text and ruling are not counted as drawing."""

    FRAMES = 40
    SIZE = 400

    def test_spiral_is_accumulated_and_printed_content_is_not(self):
        base = _textured_paper(self.SIZE, self.SIZE)
        centre, steps = (200, 250), 600
        spiral = _spiral_points(centre, 6, 5.0, 3.5, steps=steps)
        printed = np.zeros((self.SIZE, self.SIZE), np.uint8)
        cv2.putText(printed, "AI Project", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 1, 2, cv2.LINE_AA)
        cv2.rectangle(printed, (250, 20), (380, 90), 1, 2)
        ruling = np.zeros_like(printed)
        for y in range(30, self.SIZE, 24):
            ruling[y, :] = 1

        frames, layers = [], []
        for t in range(self.FRAMES):
            drawn = base.copy()
            progress = int(steps * t / (self.FRAMES - 1))  # frame 0 has no spiral yet
            cv2.polylines(drawn, [spiral[:progress]], False, (150, 60, 30), 3, cv2.LINE_AA)
            frames.append(cv2.warpAffine(drawn, _camera_matrix(t), (self.SIZE, self.SIZE),
                                         borderValue=tuple(int(v) for v in PAPER)))
        # Reference coordinates are frame 0's coordinates, so the spiral truth is moved the same way.
        truth = np.zeros((self.SIZE, self.SIZE), np.uint8)
        cv2.polylines(truth, [spiral], False, 1, 3)
        truth_ref = cv2.warpAffine(truth, _camera_matrix(0), (self.SIZE, self.SIZE)) > 0
        printed_ref = cv2.warpAffine(printed, _camera_matrix(0), (self.SIZE, self.SIZE)) > 0
        ruling_ref = cv2.warpAffine(ruling, _camera_matrix(0), (self.SIZE, self.SIZE)) > 0

        accumulated, aligned = self._accumulate(frames)
        self.assertGreaterEqual(aligned, self.FRAMES - 2, "the moving paper should align on almost every frame")
        spiral_hits = int((accumulated & truth_ref).sum())
        self.assertGreater(spiral_hits, 0.5 * int(truth_ref.sum()), "most of the drawn spiral should be found")
        # Where the spiral crosses a ruling line, the ink belongs to the spiral. Only printed marks away from it count.
        near_spiral = cv2.dilate(truth_ref.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        printed_hits = int((accumulated & (printed_ref | ruling_ref) & ~near_spiral).sum())
        self.assertLess(printed_hits, 0.05 * max(int(accumulated.sum()), 1), "printed text and ruling are not drawing")

    def _accumulate(self, frames):
        """The same steps as the video analysis: align to frame 0, persist in reference coordinates, subtract print."""
        aligner = None
        tracker = ink.InkTracker((self.SIZE, self.SIZE))
        accumulator = ink.InkAccumulator((self.SIZE, self.SIZE))
        aligned = 0
        for index, frame in enumerate(frames):
            dark, paper = ink.frame_maps(frame)
            excluded = np.zeros(paper.shape, bool)
            candidate = ink.candidate_ink(dark, paper, excluded, short_side=min(paper.shape), remove_lines=False)
            labels = ink.colour_labels(frame, candidate)
            if index == 0:
                aligner = ink.PaperAligner(frame, paper & ~excluded)
                marks = ink.reference_marks(dark)
                reference_paper = paper
                homography = np.eye(3)
            else:
                homography, _ = aligner.homography(frame, paper, excluded)
            if homography is None:
                continue
            aligned += 1
            cand_ref, _, dark_ref = ink.to_reference(candidate, labels, dark, homography, paper.shape)
            stable_ref = tracker.update(cand_ref)
            accumulator.update(ink.new_ink(stable_ref, dark_ref, marks, reference_paper))
        return accumulator.ever, aligned


class SpiralGeometryTests(unittest.TestCase):
    def test_drawn_spiral_is_fitted_with_plausible_geometry(self):
        canvas = np.zeros((260, 260), bool)
        spiral = _spiral_points((130, 130), 8, 6.0, 5.0)
        cv2.polylines(canvas.view(np.uint8), [spiral], False, 1, 2)
        fit = ink.spiral_fit(canvas)
        self.assertIsInstance(fit, dict, fit)
        self.assertAlmostEqual(fit["turn_spacing_px"], 2 * math.pi * 6.0, delta=6.0)
        self.assertGreater(fit["turns_detected"], 2.5)
        self.assertLess(fit["fit_deviation_share_of_spacing"], 0.2)
        self.assertLess(abs(fit["centre_px"][0] - 130), 8)

    def test_random_scribble_is_not_reported_as_a_spiral(self):
        rng = np.random.default_rng(3)
        canvas = np.zeros((260, 260), bool)
        points = rng.integers(40, 220, size=(60, 2)).astype(np.int32)
        cv2.polylines(canvas.view(np.uint8), [points], False, 1, 2)
        self.assertIsInstance(ink.spiral_fit(canvas), str)

    def test_handwriting_is_separated_from_the_spiral(self):
        canvas = np.zeros((300, 500), bool)
        spiral = _spiral_points((130, 150), 8, 6.0, 4.5)
        cv2.polylines(canvas.view(np.uint8), [spiral], False, 1, 2)
        text = np.zeros_like(canvas)
        cv2.putText(text.view(np.uint8), "Done", (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1, 2)
        mask, fit, ignored = ink.isolate_spiral(canvas | text)
        self.assertIsNotNone(mask)
        self.assertIsInstance(fit, dict)
        self.assertGreater(ignored, 50)
        self.assertEqual(int((mask & text).sum()), 0)


class DrawingTimelineTests(unittest.TestCase):
    def test_pauses_are_gaps_of_at_least_half_a_second(self):
        events = [(i, 10.0 + i, 20.0) for i in range(0, 10)] + [(i, 20.0 + i, 25.0) for i in range(30, 40)]
        estimated, reason = ink.drawing_timeline(events, 10.0, (100, 100))
        self.assertIsNone(reason)
        self.assertEqual(estimated["pauses"], 1)
        self.assertAlmostEqual(estimated["drawing_duration_seconds"], 3.9, places=2)

    def test_too_little_new_ink_gives_a_reason_not_values(self):
        estimated, reason = ink.drawing_timeline([(0, 1.0, 1.0)], 10.0, (100, 100))
        self.assertEqual(estimated, {})
        self.assertIsNotNone(reason)


class HandLandmarkTests(unittest.TestCase):
    def _open_hand(self, pinch=False):
        """Normalised landmarks for a straight, upright hand. With pinch=True the thumb tip sits on the index tip."""
        pts = np.zeros((21, 2))
        pts[0] = (0.50, 0.90)
        for k, chain in enumerate(hands.FINGERS.values()):
            x = 0.36 + 0.07 * k
            for j, idx in enumerate(chain):
                pts[idx] = (x, 0.70 - 0.12 * j)
        for j, idx in enumerate(hands.THUMB):
            pts[idx] = (0.30 - 0.03 * j, 0.80 - 0.08 * j)
        if pinch:
            pts[4] = pts[8] + (0.01, 0.01)
        return [[float(x), float(y), 0.0] for x, y in pts]

    def _entry(self, score=0.9, pinch=False):
        return {"handedness": "Right", "score": score, "landmarks": self._open_hand(pinch=pinch)}

    def test_low_score_hands_are_dropped(self):
        frames = hands.hand_frames([self._entry(score=0.2), None], 320, 240)
        self.assertEqual(frames, [None, None])

    def test_hand_metrics_report_detection_and_no_fabricated_pen_tip(self):
        frames = hands.hand_frames([self._entry() for _ in range(40)] + [None] * 10, 320, 240)
        measured, estimated, unavailable = hands.hand_metrics(frames, 50)
        self.assertAlmostEqual(measured["frames_with_hand_detected"], 0.8, places=2)
        self.assertIn("pen_tip_tracking", unavailable)
        self.assertIn("index_flexion_deg_median", estimated)
        self.assertIn("thumb_index_distance_share_of_hand", estimated)

    def test_grip_is_insufficient_with_too_few_frames(self):
        frames = hands.hand_frames([self._entry() for _ in range(5)] + [None] * 45, 320, 240)
        _, estimated, _ = hands.hand_metrics(frames, 50)
        grip = estimated["grip_configuration"]
        self.assertEqual(grip["label"], "Insufficient visibility")
        self.assertEqual(grip["usable_frames"], 5)

    def test_pinched_thumb_and_index_give_a_capped_pinch_estimate(self):
        frames = hands.hand_frames([self._entry(pinch=True) for _ in range(40)], 320, 240)
        _, estimated, _ = hands.hand_metrics(frames, 40)
        grip = estimated["grip_configuration"]
        self.assertEqual(grip["label"], "Pinch grip without visible support")
        self.assertEqual(grip["usable_frames"], 40)
        # The grip rule is a heuristic, so its confidence never exceeds moderate.
        self.assertIn(grip["estimate_confidence"], {"low", "moderate"})
        self.assertNotEqual(grip["estimate_confidence"], "high")

    def test_grip_is_not_classified_when_essential_fingertips_are_cut_off(self):
        cut_off = self._entry(pinch=True)
        cut_off["landmarks"][8] = [1.2, 0.5, 0.0]  # index fingertip beyond the right edge
        frames = hands.hand_frames([cut_off for _ in range(40)], 320, 240)
        measured, estimated, _ = hands.hand_metrics(frames, 40)
        self.assertEqual(estimated["grip_configuration"]["label"], "Insufficient visibility")
        self.assertEqual(estimated["grip_configuration"]["usable_frames"], 0)
        self.assertLess(measured["index_tip_visible_share"], 1.0)

    def test_hand_that_did_not_run_is_not_reported_as_zero_percent(self):
        measured, estimated, unavailable = hands.not_run_metrics("Hand landmarks did not run: the worker failed.")
        self.assertNotIn("frames_with_hand_detected", measured)
        self.assertEqual(estimated, {})
        self.assertIn("did not run", unavailable["hand_landmarks"])

    def test_worker_error_text_is_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as work:
            missing = Path(work) / "missing.mp4"
            frames, info, reason = hands.run_worker(str(missing), {"HAND_LANDMARK_PYTHON": sys.executable})
        self.assertIsNone(frames)
        self.assertTrue(reason.startswith("Hand landmarks did not run"), reason)
        # The last line of the worker's error output is included, so the cause is visible.
        self.assertGreater(len(reason), len("Hand landmarks did not run: the worker failed."))

    def test_no_hand_detected_is_reported_as_unavailable(self):
        measured, estimated, unavailable = hands.hand_metrics([None] * 20, 20)
        self.assertEqual(measured["frames_with_hand_detected"], 0.0)
        self.assertIn("hand_landmarks", unavailable)
        self.assertEqual(estimated, {})

    def test_missing_worker_python_reports_unavailable(self):
        frames, info, reason = hands.run_worker("unused.mp4", {"HAND_LANDMARK_PYTHON": ""})
        self.assertIsNone(frames)
        self.assertIn("HAND_LANDMARK_PYTHON", reason)
        self.assertFalse(hands.landmark_status({"HAND_LANDMARK_PYTHON": ""})["available"])


def _write_video(path: Path, frames, fps=12.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()


class VideoAnalysisTests(unittest.TestCase):
    config = None

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = {"UPLOAD_TEMP_DIR": cls.temp.name, "SPIRAL_MAX_DIMENSION": 4096, "HAND_LANDMARK_PYTHON": ""}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _spiral_video_frames(self, count=40, width=320, height=240):
        frames = []
        spiral = _spiral_points((width // 2, height // 2), 8, 5.0, 4.0, steps=count * 12)
        # Printed texture gives the paper features that frame alignment needs, as a real page does.
        base = _textured_paper(height, width)
        for i in range(count):
            frame = base.copy()
            cv2.polylines(frame, [spiral[: i * 12]], False, (150, 60, 30), 2, cv2.LINE_AA)
            frames.append(frame)
        return frames

    def test_annotated_webm_has_same_frames_and_reports_ink_sections(self):
        frames = self._spiral_video_frames()
        path = Path(self.temp.name) / "spiral.mp4"
        _write_video(path, frames)
        blob, extension, result = video.analyze_spiral_video(path.read_bytes(), "spiral.mp4", self.config)
        self.assertEqual(extension, "webm")
        self.assertEqual(result["video"]["frame_count"], len(frames))
        self.assertEqual(result["annotated_video_format"], "webm")
        check = Path(self.temp.name) / "check.webm"
        check.write_bytes(blob)
        cap = cv2.VideoCapture(str(check))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames))
        ok, _ = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual(set(result["sections"]), {"ink_and_strokes", "hand_and_grip", "spiral_geometry"})
        ink_labels = [item["label"] for item in result["sections"]["ink_and_strokes"]["items"]]
        self.assertIn("Ink coverage", ink_labels)
        self.assertFalse(result["hand_landmarks"]["available"])

    def test_invalid_video_is_rejected(self):
        with self.assertRaises(APIError):
            video.analyze_spiral_video(b"not a video", "bad.mp4", self.config)



@unittest.skipUnless(os.environ.get("HAND_LANDMARK_PYTHON"), "set HAND_LANDMARK_PYTHON to run the MediaPipe worker test")
class MediaPipeWorkerTests(unittest.TestCase):
    def test_worker_returns_one_entry_per_frame_for_a_real_video(self):
        frames = [np.full((240, 320, 3), 200, np.uint8) for _ in range(6)]
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "blank.mp4"
            _write_video(path, frames)
            config = {"HAND_LANDMARK_PYTHON": os.environ["HAND_LANDMARK_PYTHON"]}
            landmark_frames, info, reason = hands.run_worker(str(path), config)
        self.assertIsNone(reason)
        self.assertEqual(len(landmark_frames), 6)
        self.assertEqual(info, {"width": 320, "height": 240})
        self.assertTrue(all(entry is None for entry in landmark_frames), "a blank frame has no hand")

if __name__ == "__main__":
    unittest.main()
