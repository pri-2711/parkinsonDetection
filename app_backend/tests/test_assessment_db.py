from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from app_backend import create_app
from app_backend.database import Database


class DatabaseAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pd-db-tests-")
        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": str(Path(self.temp.name) / "test.db"),
            "UPLOAD_TEMP_DIR": self.temp.name,
            "AUDIO_MAX_BYTES": 512 * 1024,
            "SPIRAL_MAX_BYTES": 2 * 1024 * 1024,
            "MAX_CONTENT_LENGTH": 3 * 1024 * 1024,
            "SPIRAL_MEDIA_DIR": str(Path(self.temp.name) / "private-media"),
            "APP_SECRET_KEY": "test_secret_key_12345",
            "SECRET_KEY": "test_secret_key_12345",
        })
        self.client = self.app.test_client()
        from app_backend.database import Database
        self.db = Database(str(Path(self.temp.name) / "test.db"))

    def tearDown(self):
        import shutil
        import time
        try:
            conn = self.db._get_conn()
            conn.close()
        except:
            pass
        time.sleep(0.1)
        try:
            shutil.rmtree(self.temp.name, ignore_errors=True)
        except:
            pass

    def test_protected_route_requires_auth(self):
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)

    def test_register_login_invalid_credentials_and_authenticated_request(self):
        self.app.extensions["db"] = self.db
        registration = self.client.post("/api/auth/register", json={
            "username": "authtest", "email": "authtest@example.com", "password": "valid-password-123",
        })
        self.assertEqual(registration.status_code, 201, registration.get_json())

        login = self.client.post("/api/auth/login", json={
            "username": "authtest", "password": "valid-password-123",
        })
        self.assertEqual(login.status_code, 200, login.get_json())
        token = login.get_json()["token"]
        profile = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.get_json()["username"], "authtest")

        invalid = self.client.post("/api/auth/login", json={
            "username": "authtest", "password": "incorrect-password",
        })
        self.assertEqual(invalid.status_code, 401)

    def test_auth_routes_fail_clearly_without_secret_and_do_not_register_user(self):
        from app_backend import create_app

        app = create_app({"TESTING": True, "SECRET_KEY": None, "APP_SECRET_KEY": None})
        app.extensions["db"] = self.db
        client = app.test_client()
        before = self.db._get_conn().execute("SELECT COUNT(*) FROM users").fetchone()[0]

        registration = client.post("/api/auth/register", json={
            "username": "nosecret", "email": "nosecret@example.com", "password": "valid-password-123",
        })
        self.assertEqual(registration.status_code, 500)
        self.assertEqual(registration.get_json()["error"]["code"], "server_misconfigured")
        login = client.post("/api/auth/login", json={
            "username": "missing", "password": "incorrect-password",
        })
        self.assertEqual(login.status_code, 500)
        self.assertEqual(login.get_json()["error"]["code"], "server_misconfigured")
        after = self.db._get_conn().execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertEqual(after, before)

    def test_create_assessment_requires_auth(self):
        response = self.client.post("/api/assessments")
        self.assertEqual(response.status_code, 401)

    def test_database_persistence(self):
        # Create user directly in database
        from app_backend.services.auth import hash_password
        user = self.db.create_user("testuser", "test@example.com", hash_password("password123"))
        self.assertEqual(user["username"], "testuser")

        # Create assessment
        assessment = self.db.create_assessment(user["id"])
        self.assertEqual(assessment["user_id"], user["id"])
        self.assertEqual(assessment["status"], "in_progress")

        # Update step
        updated = self.db.update_assessment_step(assessment["id"], user["id"], "tremor", {
            "status": "completed",
            "outcome": {"data": {"location": "left_hand", "intensity": 5}},
            "error": None,
            "updated_at": assessment["updated_at"]
        })
        self.assertEqual(updated["steps"]["tremor"]["status"], "completed")

        # Complete assessment
        results = {"modalities": updated["steps"]}
        completed = self.db.complete_assessment(assessment["id"], user["id"], results)
        self.assertEqual(completed["status"], "completed")

        # List assessments
        history = self.db.list_assessments(user["id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], assessment["id"])

    def test_cross_user_isolation(self):
        from app_backend.services.auth import hash_password
        # Create two users
        user1 = self.db.create_user("user1", "user1@example.com", hash_password("password123"))
        user2 = self.db.create_user("user2", "user2@example.com", hash_password("password456"))

        # Create assessment for user1
        assessment1 = self.db.create_assessment(user1["id"])

        # User2 cannot access user1's assessment
        assessment = self.db.get_assessment(assessment1["id"], user2["id"])
        self.assertIsNone(assessment)

        # User1 can access their own assessment
        assessment = self.db.get_assessment(assessment1["id"], user1["id"])
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment["id"], assessment1["id"])

    def test_tremor_model_prediction_uses_exact_updrs_schema_and_persists_probability(self):
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("tremoruser", "tremor@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        adapter = self.app.extensions["model_registry"].models["tremor"]
        payload = {feature: 0 for feature in adapter.features}
        response = self.client.post(
            f"/api/assessments/{assessment['id']}/tremor/predict",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        outcome = response.get_json()["outcome"]
        self.assertGreaterEqual(outcome["probability"], 0)
        self.assertLessEqual(outcome["probability"], 1)
        saved = self.db.get_assessment(assessment["id"], user["id"])
        self.assertEqual(saved["steps"]["tremor"]["outcome"]["probability"], outcome["probability"])

        payload[adapter.features[0]] = 5
        invalid = self.client.post(
            f"/api/assessments/{assessment['id']}/tremor/predict",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(invalid.status_code, 422)

        forged = self.client.post(
            f"/api/assessments/{assessment['id']}/tremor",
            json={"outcome": {"probability": 0.99, "prediction": 1}},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(forged.status_code, 422)

        completed = self.client.post(
            f"/api/assessments/{assessment['id']}/complete",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(completed.status_code, 200, completed.get_json())
        results = self.client.get(
            f"/api/assessments/{assessment['id']}/results",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(results.status_code, 200)
        self.assertEqual(results.get_json()["fusion"]["available_modalities"], ["tremor"])
        self.assertEqual(results.get_json()["fusion"]["effective_weights"], {"tremor": 1.0})
        self.assertEqual(results.get_json()["fusion"]["patient_level_validation"], "not_measured_unmatched_modalities")
        history = self.client.get("/api/history/completed", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.get_json()["assessments"][0]["id"], assessment["id"])

        # Test PDF Report Generation
        pdf_res = self.client.get(
            f"/api/assessments/{assessment['id']}/report.pdf",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.content_type, "application/pdf")
        self.assertTrue(pdf_res.data.startswith(b"%PDF"))

    def test_handwriting_trajectory_prediction_and_fusion(self):
        from app_backend.services.auth import create_token, hash_password
        import pandas as pd

        self.app.extensions["db"] = self.db
        user = self.db.create_user("spiraluser", "spiral@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")

        # Use an existing fully-measured source tablet trajectory, not invented
        # browser-canvas pressure or grip-angle values.
        source = pd.read_csv(Path(__file__).resolve().parents[2] / "cleaned_dataset" / "handwriting_clean.csv")
        for _, group in source.groupby(["participant_id", "test_id"], sort=True):
            if len(group) >= 15 and group[["x", "y", "z", "pressure", "grip_angle", "timestamp"]].notna().all().all():
                points = group[["point_index", "x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id"]].to_dict("records")
                break
        else:
            self.fail("No complete training-compatible tablet trace was found")

        response = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/predict",
            json={"trajectory": points, "input_mode": "measured_tablet_telemetry"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        outcome = response.get_json()["outcome"]
        self.assertTrue(outcome["model_verified"])
        self.assertIn("probability", outcome)
        self.assertGreaterEqual(outcome["probability"], 0.0)
        self.assertLessEqual(outcome["probability"], 1.0)

        # Complete assessment with handwriting
        completed = self.client.post(
            f"/api/assessments/{assessment['id']}/complete",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(completed.status_code, 200)

        results = self.client.get(
            f"/api/assessments/{assessment['id']}/results",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(results.status_code, 200)
        self.assertEqual(results.get_json()["fusion"]["available_modalities"], ["handwriting"])

        # Test PDF Report for handwriting-only assessment
        pdf_res = self.client.get(
            f"/api/assessments/{assessment['id']}/report.pdf",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(pdf_res.status_code, 200)
        self.assertTrue(pdf_res.data.startswith(b"%PDF"))

    def test_spiral_rejects_canvas_points_without_measured_tablet_features(self):
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("canvasuser", "canvas@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        points = [{"point_index": i, "x": float(i), "y": float(i * 2), "timestamp": i * 20} for i in range(20)]

        response = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/predict",
            json={"trajectory": points, "input_mode": "canvas_coordinates_only"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "tablet_telemetry_required")
        self.assertIn("cannot provide", response.get_json()["error"]["message"])

    def test_canvas_observations_are_saved_without_a_classifier_prediction(self):
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("canvasobserve", "canvasobserve@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        samples = [
            {"x": 10, "y": 10, "timestampMs": 0, "phase": "down", "strokeId": 0,
             "pressure": None, "pressureAvailable": False, "tiltX": None, "tiltY": None, "tiltAvailable": False},
            {"x": 20, "y": 20, "timestampMs": 100, "phase": "move", "strokeId": 0,
             "pressure": None, "pressureAvailable": False, "tiltX": None, "tiltY": None, "tiltAvailable": False},
            {"x": 40, "y": 20, "timestampMs": 200, "phase": "up", "strokeId": 0,
             "pressure": None, "pressureAvailable": False, "tiltX": None, "tiltY": None, "tiltAvailable": False},
        ]
        response = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/canvas/observe",
            json={"samples": samples, "canvas_width": 500, "canvas_height": 500},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        observation = response.get_json()["observation"]
        self.assertEqual(observation["sample_count"], 3)
        self.assertAlmostEqual(observation["path_length_px"], 34.142, places=2)
        self.assertFalse(observation["model_compatible"])
        saved = self.db.get_assessment(assessment["id"], user["id"])
        self.assertEqual(saved["steps"]["spiral"]["status"], "unavailable")
        self.assertEqual(saved["steps"]["spiral"]["outcome"]["canvas_telemetry"], observation)
        self.assertNotIn("probability", saved["steps"]["spiral"]["outcome"])
        history = self.client.get(
            "/api/history/assessments?limit=50",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.get_json()["assessments"][0]["id"], assessment["id"])

        invalid = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/canvas/observe",
            json={"samples": [{**samples[0], "x": 9000}, samples[1]], "canvas_width": 500, "canvas_height": 500},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(invalid.status_code, 422)

    def test_spiral_video_analysis_stores_associated_playable_private_output(self):
        import cv2
        import numpy as np
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("videouser", "video@example.com", hash_password("password123"))
        other = self.db.create_user("otheruser", "other@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        other_token = create_token(other["id"], "test_secret_key_12345")
        temp_video = Path(self.temp.name) / "synthetic.mp4"
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (160, 120))
        self.assertTrue(writer.isOpened(), "test environment must encode synthetic source video")
        for i in range(36):
            frame = np.full((120, 160, 3), 245, dtype=np.uint8)
            x = 80 + int(10 * np.sin(i * 2.0))
            cv2.circle(frame, (x, 60 + int(3 * np.sin(i * 2.0))), 5, (20, 20, 20), -1)
            writer.write(frame)
        writer.release()
        video_bytes = temp_video.read_bytes()

        unauthenticated = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/video/analyze",
            data={"file": (io.BytesIO(video_bytes), "trace.mp4", "video/mp4")},
        )
        self.assertEqual(unauthenticated.status_code, 401)
        response = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/video/analyze",
            data={"file": (io.BytesIO(video_bytes), "trace.mp4", "video/mp4")},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        analysis = response.get_json()["analysis"]
        self.assertEqual(analysis["analysis_type"], "opencv_visual_baseline_not_a_diagnosis")
        self.assertEqual(analysis["video"]["frame_count"], 36)
        self.assertGreater(analysis["trajectory"]["tracked_points"], 10)
        self.assertIsNotNone(analysis["trajectory"].get("path_length_px"))
        self.assertGreater(len(analysis["candidate_movement_events"]), 0)
        saved = self.db.get_assessment(assessment["id"], user["id"])
        sidecar = saved["steps"]["spiral"]["outcome"]["video_analysis"]
        self.assertEqual(sidecar["video_id"], analysis["video_id"])
        self.assertEqual(saved["steps"]["spiral"]["status"], "unavailable")
        self.assertEqual(saved["steps"]["spiral"]["outcome"]["prediction_status"], "unavailable")

        annotated = self.client.get(sidecar["annotated_video_url"], headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(annotated.status_code, 200)
        self.assertEqual(annotated.content_type, "video/mp4")
        decoded_path = Path(self.temp.name) / "returned.mp4"
        decoded_path.write_bytes(annotated.data)
        cap = cv2.VideoCapture(str(decoded_path))
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertGreater(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        cap.release()

        original = self.client.get(sidecar["original_video_url"], headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(original.data, video_bytes)
        forbidden = self.client.get(sidecar["annotated_video_url"], headers={"Authorization": f"Bearer {other_token}"})
        self.assertEqual(forbidden.status_code, 404)
        report = self.client.get(
            f"/api/assessments/{assessment['id']}/report.pdf", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(report.status_code, 200)
        self.assertTrue(report.data.startswith(b"%PDF"))

    def test_spiral_video_rejects_corrupt_oversized_and_bad_extension(self):
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("videovalid", "videovalid@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        headers = {"Authorization": f"Bearer {token}"}
        def submit(data: bytes, name: str, mime: str):
            return self.client.post(f"/api/assessments/{assessment['id']}/spiral/video/analyze",
                                    data={"file": (io.BytesIO(data), name, mime)}, headers=headers)
        corrupt = submit(b"not a video", "broken.mp4", "video/mp4")
        self.assertEqual(corrupt.status_code, 400)
        oversized = submit(b"x" * (2 * 1024 * 1024 + 1), "large.mp4", "video/mp4")
        self.assertEqual(oversized.status_code, 413)
        extension = submit(b"whatever", "not-video.txt", "text/plain")
        self.assertEqual(extension.status_code, 415)

    def test_spiral_video_processing_failure_returns_clear_error(self):
        import cv2
        import numpy as np
        from unittest.mock import patch
        from app_backend.errors import APIError
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("videofail", "videofail@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        path = Path(self.temp.name) / "failure.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 80))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.full((80, 80, 3), 255, dtype=np.uint8))
        writer.release()
        with patch("app_backend.routes.assessment.analyze_spiral_video", side_effect=APIError(422, "video_processing_failed", "Synthetic processing failure.")):
            response = self.client.post(
                f"/api/assessments/{assessment['id']}/spiral/video/analyze",
                data={"file": (io.BytesIO(path.read_bytes()), "failure.mp4", "video/mp4")},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "video_processing_failed")

    def test_speech_assessment_extracts_audio_then_reports_missing_v2_model(self):
        from app_backend.services.auth import create_token, hash_password

        self.app.extensions["db"] = self.db
        user = self.db.create_user("speechuser", "speech@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")
        from app_backend.tests.test_backend import make_voiced_wav
        response = self.client.post(
            f"/api/assessments/{assessment['id']}/speech/predict",
            data={"mode": "sustained_vowel", "gender_code": "0",
                  "file": (io.BytesIO(make_voiced_wav()), "sample.wav", "audio/wav")},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["status"], "unavailable")
        outcome = self.db.get_assessment(assessment["id"], user["id"])["steps"]["speech"]["outcome"]
        self.assertTrue(outcome["audio_received"])
        self.assertFalse(outcome["audio_stored"])
        self.assertEqual(outcome["feature_extraction"], "completed")
        self.assertEqual(outcome["feature_schema"], "speech_audio_v2")
        self.assertEqual(outcome["feature_count"], 45)
        self.assertEqual(len(outcome["feature_values"]), 45)
        self.assertEqual(self.db.get_assessment(assessment["id"], user["id"])["steps"]["speech"]["status"], "unavailable")
        self.assertIsNone(outcome.get("probability"))

    def test_root_and_api_discovery_routes(self):
        for path in ("/", "/api"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn("service", data)
            self.assertIn("modalities", data)
            self.assertIn("disclaimer", data)


if __name__ == "__main__":
    unittest.main()

