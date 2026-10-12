import unittest

import numpy as np

from src.fusion import fuse_probabilities


class FusionProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.weights = {"speech": 0.2, "handwriting": 0.3, "tremor": 0.5}

    def test_all_three_modalities(self):
        result = fuse_probabilities(0.2, 0.6, 0.8, self.weights)
        self.assertAlmostEqual(result["probability"], 0.62)
        self.assertEqual(result["available_modalities"], ["speech", "handwriting", "tremor"])

    def test_each_two_modality_combination(self):
        combinations = [
            (0.2, 0.6, None, ["speech", "handwriting"]),
            (0.2, None, 0.8, ["speech", "tremor"]),
            (None, 0.6, 0.8, ["handwriting", "tremor"]),
        ]
        for speech, handwriting, tremor, expected in combinations:
            with self.subTest(modalities=expected):
                result = fuse_probabilities(speech, handwriting, tremor, self.weights)
                self.assertEqual(result["available_modalities"], expected)
                self.assertAlmostEqual(sum(result["effective_weights"].values()), 1.0)

    def test_each_single_modality(self):
        cases = [
            (0.25, None, None, "speech"),
            (None, 0.55, None, "handwriting"),
            (None, None, 0.85, "tremor"),
        ]
        for speech, handwriting, tremor, expected in cases:
            with self.subTest(modality=expected):
                result = fuse_probabilities(speech, handwriting, tremor, self.weights)
                self.assertEqual(result["probability"], {"speech": speech, "handwriting": handwriting, "tremor": tremor}[expected])
                self.assertEqual(result["available_modalities"], [expected])
                self.assertEqual(result["effective_weights"], {expected: 1.0})

    def test_no_modality_raises(self):
        with self.assertRaisesRegex(ValueError, "At least one modality"):
            fuse_probabilities()

    def test_invalid_probability_raises(self):
        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(probability=invalid):
                with self.assertRaisesRegex(ValueError, "Probability for speech"):
                    fuse_probabilities(speech_probability=invalid)

    def test_weights_renormalize_when_modalities_are_missing(self):
        result = fuse_probabilities(speech_probability=0.2, handwriting_probability=0.8, weights=self.weights)
        self.assertAlmostEqual(result["effective_weights"]["speech"], 0.4)
        self.assertAlmostEqual(result["effective_weights"]["handwriting"], 0.6)
        self.assertAlmostEqual(result["effective_weights"]["speech"] + result["effective_weights"]["handwriting"], 1.0)

    def test_final_weights_sum_to_one(self):
        result = fuse_probabilities(0.2, 0.6, 0.8, self.weights)
        self.assertTrue(np.isclose(sum(result["effective_weights"].values()), 1.0))

    def test_threshold_controls_prediction(self):
        below = fuse_probabilities(speech_probability=0.49, threshold=0.5)
        at_threshold = fuse_probabilities(speech_probability=0.5, threshold=0.5)
        self.assertEqual(below["prediction"], 0)
        self.assertEqual(at_threshold["prediction"], 1)

    def test_arrays_and_probability_shape_validation(self):
        result = fuse_probabilities(np.array([0.1, 0.9]), tremor_probability=np.array([0.3, 0.7]))
        self.assertTrue(np.allclose(result["probability"], [0.2, 0.8]))
        with self.assertRaisesRegex(ValueError, "same shape"):
            fuse_probabilities(np.array([0.1, 0.9]), handwriting_probability=np.array([0.5]))

    def test_negative_or_zero_available_weights_raise(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fuse_probabilities(speech_probability=0.4, weights={"speech": -1})
        with self.assertRaisesRegex(ValueError, "positive total weight"):
            fuse_probabilities(speech_probability=0.4, weights={"speech": 0})


if __name__ == "__main__":
    unittest.main()
