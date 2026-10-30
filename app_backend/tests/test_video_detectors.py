from __future__ import annotations

import unittest

import cv2
import numpy as np

from app_backend.services import spiral_video_analysis as video


def _frame():
    """Light paper, a blue ink stroke on the left and a skin-coloured hand region on the right."""
    frame = np.full((240, 320, 3), 235, np.uint8)
    cv2.line(frame, (30, 120), (130, 60), (150, 60, 30), 3)      # blue ink (BGR)
    cv2.ellipse(frame, (230, 150), (45, 60), 0, 0, 360, (120, 150, 210), -1)  # skin tone (BGR)
    return frame


class BlueInkDetectorTests(unittest.TestCase):
    def test_blue_stroke_is_ink_and_skin_is_not(self):
        frame = _frame()
        hand = video._hand_region(frame)
        self.assertIsNotNone(hand, "skin region should be detected")
        ink = video._ink_mask(frame, hand[0])
        ink_pixels = ink > 0
        self.assertGreater(int(ink_pixels.sum()), 40)
        # Ink lies on the stroke, not inside the skin region.
        self.assertEqual(int((ink_pixels & (hand[0] > 0)).sum()), 0)
        ys, xs = np.nonzero(ink_pixels)
        self.assertTrue(xs.max() < 160, "blue ink should stay on the left stroke")

    def test_plain_paper_gives_no_ink(self):
        frame = np.full((240, 320, 3), 235, np.uint8)
        ink = video._ink_mask(frame, None)
        self.assertEqual(int(np.count_nonzero(ink)), 0)


class HandDetectorTests(unittest.TestCase):
    def test_hand_region_is_independent_of_the_ink_stroke(self):
        frame = _frame()
        hand = video._hand_region(frame)
        self.assertIsNotNone(hand)
        _, contour, area_fraction, solidity, fingertips = hand
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        self.assertGreater(cx, 180)   # the skin blob centre is near x=230, the ink is at x<130
        self.assertGreater(area_fraction, 0.03)
        self.assertGreaterEqual(fingertips, 1)

    def test_no_hand_on_plain_paper(self):
        self.assertIsNone(video._hand_region(np.full((240, 320, 3), 235, np.uint8)))


if __name__ == "__main__":
    unittest.main()
