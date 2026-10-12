from __future__ import annotations

import copy
import secrets
import threading
import time
from datetime import datetime, timezone

from app_backend.errors import APIError

REQUIRED_STEPS = ("speech", "spiral", "tremor")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssessmentSessionStore:
    """Process-local session storage. Data is discarded on expiry or restart."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = int(ttl_seconds)
        self._items: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _purge_expired(self) -> None:
        expired = [sid for sid, row in self._items.items() if time.time() - row["_touched"] > self.ttl_seconds]
        for sid in expired:
            del self._items[sid]

    @staticmethod
    def _public(row: dict) -> dict:
        result = copy.deepcopy(row)
        result.pop("_touched", None)
        return result

    def create(self) -> dict:
        with self._lock:
            self._purge_expired()
            session_id = secrets.token_urlsafe(24)
            row = {
                "session_id": session_id,
                "status": "in_progress",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "steps": {
                    step: {"status": "pending", "outcome": None, "error": None}
                    for step in REQUIRED_STEPS
                },
                "next_step": "speech",
                "results": None,
                "_touched": time.time(),
            }
            self._items[session_id] = row
            return self._public(row)

    def _get_live(self, session_id: str) -> dict:
        self._purge_expired()
        row = self._items.get(session_id)
        if row is None:
            raise APIError(404, "session_not_found", "The assessment session does not exist or has expired.")
        row["_touched"] = time.time()
        return row

    def get(self, session_id: str) -> dict:
        with self._lock:
            return self._public(self._get_live(session_id))

    def assert_current_step(self, session_id: str, step: str) -> dict:
        if step not in REQUIRED_STEPS:
            raise ValueError("unknown assessment step")
        with self._lock:
            row = self._get_live(session_id)
            if row["status"] != "in_progress" or row["next_step"] != step:
                raise APIError(409, "assessment_step_out_of_order", "Complete the required earlier assessment steps first.")
            return self._public(row)

    def begin_step(self, session_id: str, step: str) -> None:
        with self._lock:
            row = self._get_live(session_id)
            if row["status"] != "in_progress" or row["next_step"] != step:
                raise APIError(409, "assessment_step_out_of_order", "Complete the required earlier assessment steps first.")
            row["steps"][step] = {"status": "processing", "outcome": None, "error": None}
            row["updated_at"] = _now_iso()

    def record_failure(self, session_id: str, step: str, code: str, message: str) -> None:
        with self._lock:
            row = self._get_live(session_id)
            if row["status"] != "in_progress" or row["next_step"] != step:
                return
            row["steps"][step] = {
                "status": "failed",
                "outcome": None,
                "error": {"code": code, "message": message},
            }
            row["updated_at"] = _now_iso()

    def record_success(self, session_id: str, step: str, outcome: dict) -> dict:
        with self._lock:
            row = self._get_live(session_id)
            if row["status"] != "in_progress" or row["next_step"] != step:
                raise APIError(409, "assessment_step_out_of_order", "Complete the required earlier assessment steps first.")
            row["steps"][step] = {"status": "completed", "outcome": copy.deepcopy(outcome), "error": None}
            completed_index = REQUIRED_STEPS.index(step) + 1
            row["next_step"] = REQUIRED_STEPS[completed_index] if completed_index < len(REQUIRED_STEPS) else None
            row["updated_at"] = _now_iso()
            return self._public(row)

    def complete(self, session_id: str) -> dict:
        with self._lock:
            row = self._get_live(session_id)
            missing = [step for step in REQUIRED_STEPS if row["steps"][step]["status"] != "completed"]
            if missing:
                raise APIError(409, "assessment_incomplete", "Submit all required assessment steps before completing the assessment.")
            row["status"] = "completed"
            row["next_step"] = None
            row["results"] = {
                step: copy.deepcopy(row["steps"][step]["outcome"])
                for step in REQUIRED_STEPS
            }
            row["completed_at"] = _now_iso()
            row["updated_at"] = row["completed_at"]
            return self._public(row)

    def results(self, session_id: str) -> dict:
        with self._lock:
            row = self._get_live(session_id)
            if row["status"] != "completed":
                raise APIError(409, "assessment_incomplete", "Results are available only after all assessment steps are complete.")
            return {"session_id": session_id, "status": row["status"], "results": copy.deepcopy(row["results"])}
