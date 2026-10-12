from __future__ import annotations

import io
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from app_backend import create_app
from app_backend.errors import APIError
from app_backend.services.sessions import AssessmentSessionStore
from app_backend.services.spiral_preprocessing import validate_spiral_video
from app_backend.services.trajectory_features import extract_tablet_trajectory_features


ROOT = Path(__file__).resolve().parents[2]


def make_wav(seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return stream.getvalue()


def make_png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(stream, format="PNG")
    return stream.getvalue()


class BackendAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pd-api-tests-")
        cls.app = create_app({
            "TESTING": True,
            "UPLOAD_TEMP_DIR": cls.temp.name,
            "AUDIO_MAX_BYTES": 32 * 1024,
            "SPIRAL_MAX_BYTES": 2 * 1024 * 1024,
            "MAX_CONTENT_LENGTH": 3 * 1024 * 1024,
        })
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_health_reports_loaded_models_and_blocked_live_prediction(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["models"]["speech"]["model_loaded"])
        self.assertTrue(body["models"]["handwriting"]["model_loaded"])
        self.assertFalse(body["capabilities"]["speech_prediction"])
        self.assertFalse(body["capabilities"]["spiral_prediction"])
        self.assertNotIn("artifact_path", str(body))

    def test_saved_speech_model_loads_and_predicts_without_retraining(self):
        adapter = self.app.extensions["model_registry"].models["speech"]
        speech = pd.read_csv(ROOT / "cleaned_dataset" / "speech_clean.csv")
        self.assertEqual([c for c in speech.columns if c not in {"id", "recording", "status"}], adapter.features)
        frame = speech.loc[:, adapter.features].iloc[:1]
        result = adapter.predict(frame)
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)

    def test_saved_handwriting_model_loads_and_predicts_on_training_schema(self):
        adapter = self.app.extensions["model_registry"].models["handwriting"]
        frame = pd.read_csv(ROOT / "featureEngineering" / "handwriting_features.csv").loc[:, adapter.features].iloc[:1]
        result = adapter.predict(frame)
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)

    def test_speech_missing_unsupported_malformed_oversized_and_valid_uploads(self):
        missing = self.client.post("/api/speech/predict")
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.get_json()["error"]["code"], "missing_audio")

        unsupported = self.client.post("/api/speech/predict", data={
            "file": (io.BytesIO(b"audio"), "voice.mp3", "audio/mpeg")
        })
        self.assertEqual(unsupported.status_code, 415)

        malformed = self.client.post("/api/speech/predict", data={
            "file": (io.BytesIO(b"not wav"), "voice.wav", "audio/wav")
        })
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()["error"]["code"], "invalid_audio")

        oversized = self.client.post("/api/speech/predict", data={
            "file": (io.BytesIO(b"x" * (32 * 1024 + 1)), "voice.wav", "audio/wav")
        })
        self.assertEqual(oversized.status_code, 413)

        valid = self.client.post("/api/speech/predict", data={
            "file": (io.BytesIO(make_wav()), "voice.wav", "audio/wav")
        })
        self.assertEqual(valid.status_code, 501)
        self.assertEqual(valid.get_json()["error"]["code"], "raw_audio_feature_extractor_unavailable")
        self.assertNotIn("probability", valid.get_json())

        short_audio = self.client.post("/api/speech/predict", data={
            "file": (io.BytesIO(make_wav(seconds=0.5)), "short.wav", "audio/wav")
        })
        self.assertEqual(short_audio.status_code, 422)

    def test_spiral_valid_image_is_not_misclassified_and_malformed_image_is_rejected(self):
        status = self.client.get("/api/spiral/status")
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.get_json()["prediction_available"])

        malformed = self.client.post("/api/spiral/predict", data={
            "file": (io.BytesIO(b"not an image"), "spiral.png", "image/png")
        })
        self.assertEqual(malformed.status_code, 400)

        unsupported = self.client.post("/api/spiral/predict", data={
            "file": (io.BytesIO(b"GIF89a"), "spiral.gif", "image/gif")
        })
        self.assertEqual(unsupported.status_code, 415)

        valid = self.client.post("/api/spiral/predict", data={
            "file": (io.BytesIO(make_png()), "spiral.png", "image/png")
        })
        self.assertEqual(valid.status_code, 501)
        self.assertEqual(valid.get_json()["error"]["code"], "camera_to_tablet_feature_mapping_unvalidated")
        self.assertNotIn("prediction", valid.get_json())

        missing = self.client.post("/api/spiral/predict")
        self.assertEqual(missing.status_code, 400)
        oversized = self.client.post("/api/spiral/predict", data={
            "file": (io.BytesIO(b"x" * (2 * 1024 * 1024 + 1)), "oversize.png", "image/png")
        })
        self.assertEqual(oversized.status_code, 413)

    def test_video_temp_file_cleanup_after_success_and_failure(self):
        cfg = dict(self.app.config)
        cfg["UPLOAD_TEMP_DIR"] = self.temp.name
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        class Capture:
            def __init__(self, opened=True):
                self.opened = opened
                self.count = 0
                self.released = False

            def isOpened(self):
                return self.opened

            def get(self, prop):
                return 10.0 if prop == 5 else 2.0

            def read(self):
                if self.count < 2:
                    self.count += 1
                    return True, frame
                return False, None

            def release(self):
                self.released = True

        cv2 = types.SimpleNamespace(
            VideoCapture=lambda _path: Capture(), CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7
        )
        with patch.dict(sys.modules, {"cv2": cv2}):
            result = validate_spiral_video(b"test video bytes", "test.mp4", "video/mp4", cfg)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

        broken_cv2 = types.SimpleNamespace(
            VideoCapture=lambda _path: Capture(opened=False), CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7
        )
        with patch.dict(sys.modules, {"cv2": broken_cv2}):
            with self.assertRaises(APIError):
                validate_spiral_video(b"broken", "broken.mp4", "video/mp4", cfg)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_tremor_placeholder_accepts_valid_fields_and_rejects_invalid_fields(self):
        response = self.client.post("/api/tremor/submit", json={"location": "left_hand", "intensity": 4})
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertTrue(body["prototype"])
        self.assertIsNone(body["classification"])
        self.assertIsNone(body["updrs_estimate"])

        for values in (
            {"location": "brain", "intensity": 3},
            {"location": "left_hand", "intensity": 11},
            {"location": "left_hand", "intensity": True},
        ):
            with self.subTest(values=values):
                invalid = self.client.post("/api/tremor/submit", json=values)
                self.assertEqual(invalid.status_code, 422)

    def test_assessment_order_retry_state_and_results_gate(self):
        created = self.client.post("/api/assessment/sessions")
        self.assertEqual(created.status_code, 201)
        session_id = created.get_json()["session_id"]
        self.assertEqual(created.get_json()["next_step"], "speech")

        out_of_order = self.client.post("/api/tremor/submit", json={
            "location": "left_hand", "intensity": 3, "session_id": session_id
        })
        self.assertEqual(out_of_order.status_code, 409)

        blocked_speech = self.client.post("/api/speech/predict", data={
            "session_id": session_id,
            "file": (io.BytesIO(make_wav()), "voice.wav", "audio/wav"),
        })
        self.assertEqual(blocked_speech.status_code, 501)
        state = self.client.get(f"/api/assessment/sessions/{session_id}").get_json()
        self.assertEqual(state["steps"]["speech"]["status"], "failed")
        self.assertEqual(state["next_step"], "speech")
        self.assertEqual(self.client.post(f"/api/assessment/sessions/{session_id}/complete").status_code, 409)
        self.assertEqual(self.client.get(f"/api/assessment/sessions/{session_id}/results").status_code, 409)

    def test_session_store_requires_all_steps_in_order(self):
        store = AssessmentSessionStore()
        session = store.create()
        sid = session["session_id"]
        with self.assertRaises(APIError):
            store.record_success(sid, "spiral", {"status": "validated"})
        store.record_success(sid, "speech", {"prediction": 1, "probability": 0.8})
        store.record_success(sid, "spiral", {"prediction": 0, "probability": 0.2})
        store.record_success(sid, "tremor", {"prototype": True, "classification": None})
        completed = store.complete(sid)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(store.results(sid)["status"], "completed")


class TrainingFeatureReuseTests(unittest.TestCase):
    def test_tablet_feature_adapter_matches_training_extractor_and_model_order(self):
        trajectories = pd.read_csv(ROOT / "cleaned_dataset" / "handwriting_clean.csv")
        training_features = pd.read_csv(ROOT / "featureEngineering" / "handwriting_features.csv")
        from joblib import load

        model = load(ROOT / "experiments" / "regularized_svm_rf" / "results" / "artifacts" / "handwriting_svm_tuned.joblib")
        first = training_features.iloc[0]
        key_mask = (
            (trajectories["participant_id"] == first["participant_id"])
            & (trajectories["test_id"] == first["test_id"])
        )
        trajectory = trajectories.loc[key_mask].copy()
        extracted = extract_tablet_trajectory_features(trajectory, model["features"])
        expected = training_features.loc[
            (training_features["participant_id"] == first["participant_id"])
            & (training_features["test_id"] == first["test_id"]),
            model["features"],
        ].reset_index(drop=True)
        np.testing.assert_allclose(extracted.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
