from __future__ import annotations

from app_backend.repositories.assessment_repository import AssessmentRepository


class HistoryService:
    def __init__(self, assessment_repo: AssessmentRepository):
        self.assessment_repo = assessment_repo

    def get_user_assessments(self, user_id: int, limit: int = 50) -> list[dict]:
        return self.assessment_repo.list_assessments(user_id, limit)

    def get_completed_assessments(self, user_id: int, limit: int = 100) -> list[dict]:
        return self.assessment_repo.get_completed_assessments(user_id, limit)

    def get_assessment(self, assessment_id: int, user_id: int) -> dict | None:
        return self.assessment_repo.get_assessment(assessment_id, user_id)
