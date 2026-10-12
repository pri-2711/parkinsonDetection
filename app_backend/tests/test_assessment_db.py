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


if __name__ == "__main__":
    unittest.main()
