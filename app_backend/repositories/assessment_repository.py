from __future__ import annotations

from app_backend.database import Database


class AssessmentRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_assessment(self, user_id: int) -> dict:
        return self.db.create_assessment(user_id)

    def get_assessment(self, assessment_id: int, user_id: int) -> dict | None:
        return self.db.get_assessment(assessment_id, user_id)

    def update_assessment_step(
        self, assessment_id: int, user_id: int, step: str, step_data: dict
    ) -> dict | None:
        return self.db.update_assessment_step(assessment_id, user_id, step, step_data)

    def complete_assessment(self, assessment_id: int, user_id: int, results: dict) -> dict | None:
        return self.db.complete_assessment(assessment_id, user_id, results)

    def list_assessments(self, user_id: int, limit: int = 50) -> list[dict]:
        return self.db.list_assessments(user_id, limit)

    def get_completed_assessments(self, user_id: int, limit: int = 100) -> list[dict]:
        return self.db.get_completed_assessments(user_id, limit)
