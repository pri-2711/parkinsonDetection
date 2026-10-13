from __future__ import annotations

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
            "AUDIO_MAX_BYTES": 32 * 1024,
            "SPIRAL_MAX_BYTES": 2 * 1024 * 1024,
            "MAX_CONTENT_LENGTH": 3 * 1024 * 1024,
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
        import numpy as np

        self.app.extensions["db"] = self.db
        user = self.db.create_user("spiraluser", "spiral@example.com", hash_password("password123"))
        assessment = self.db.create_assessment(user["id"])
        token = create_token(user["id"], "test_secret_key_12345")

        # Create valid synthetic spiral trajectory points
        t = np.linspace(0, 8, 50)
        points = [
            {
                "point_index": i,
                "x": float(200 + 10 * ti * np.cos(ti)),
                "y": float(200 + 10 * ti * np.sin(ti)),
                "z": 0.0,
                "pressure": 0.6,
                "grip_angle": 45.0,
                "timestamp": float(i * 20),
                "test_id": 1,
            }
            for i, ti in enumerate(t)
        ]

        response = self.client.post(
            f"/api/assessments/{assessment['id']}/spiral/predict",
            json={"trajectory": points, "input_mode": "canvas_drawing"},
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

