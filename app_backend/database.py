"""SQLite persistence layer for user accounts and assessment history.

Uses Python's built-in sqlite3 module — no external database server required.
The database file path is set via the DATABASE_PATH environment variable
(default: ``data/parkinson_app.db`` relative to the repository root).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPOSITORY_ROOT / "data" / "parkinson_app.db"


def _db_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    pw_hash     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    status          TEXT    NOT NULL DEFAULT 'in_progress',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    completed_at    TEXT,
    steps           TEXT    NOT NULL DEFAULT '{}',
    results         TEXT
);

CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments(user_id);
"""


class Database:
    """Thread-safe SQLite wrapper with auto-schema initialization."""

    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else _db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    @contextmanager
    def transaction(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._get_conn().execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        row = self._get_conn().execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._get_conn().execute(sql, params).fetchall()]

    # ── User operations ─────────────────────────────────────────────

    def create_user(self, username: str, email: str, pw_hash: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, pw_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, pw_hash, now, now),
            )
            return {"id": cursor.lastrowid, "username": username, "email": email, "created_at": now}

    def get_user_by_username(self, username: str) -> dict | None:
        return self.fetchone("SELECT * FROM users WHERE username = ?", (username,))

    def get_user_by_email(self, email: str) -> dict | None:
        return self.fetchone("SELECT * FROM users WHERE email = ?", (email,))

    def get_user_by_id(self, user_id: int) -> dict | None:
        return self.fetchone("SELECT id, username, email, created_at, updated_at FROM users WHERE id = ?", (user_id,))

    # ── Assessment operations ────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_assessment(self, user_id: int) -> dict:
        now = self._now()
        steps = {
            "speech": {"status": "pending", "outcome": None, "error": None, "updated_at": None},
            "spiral": {"status": "pending", "outcome": None, "error": None, "updated_at": None},
            "tremor": {"status": "pending", "outcome": None, "error": None, "updated_at": None},
        }
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO assessments (user_id, status, created_at, updated_at, steps) VALUES (?, ?, ?, ?, ?)",
                (user_id, "in_progress", now, now, json.dumps(steps)),
            )
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "status": "in_progress",
                "created_at": now,
                "updated_at": now,
                "steps": steps,
                "results": None,
            }

    def get_assessment(self, assessment_id: int, user_id: int) -> dict | None:
        row = self.fetchone(
            "SELECT * FROM assessments WHERE id = ? AND user_id = ?",
            (assessment_id, user_id),
        )
        if row:
            row["steps"] = json.loads(row["steps"])
            row["results"] = json.loads(row["results"]) if row["results"] else None
        return row

    def update_assessment_step(
        self, assessment_id: int, user_id: int, step: str, step_data: dict
    ) -> dict | None:
        row = self.get_assessment(assessment_id, user_id)
        if not row:
            return None
        steps = row["steps"]
        steps[step] = step_data
        now = self._now()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE assessments SET steps = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (json.dumps(steps), now, assessment_id, user_id),
            )
        row["steps"] = steps
        row["updated_at"] = now
        return row

    def complete_assessment(self, assessment_id: int, user_id: int, results: dict) -> dict | None:
        now = self._now()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE assessments SET status = 'completed', completed_at = ?, updated_at = ?, results = ? WHERE id = ? AND user_id = ?",
                (now, now, json.dumps(results), assessment_id, user_id),
            )
        return self.get_assessment(assessment_id, user_id)

    def list_assessments(self, user_id: int, limit: int = 50) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        for row in rows:
            row["steps"] = json.loads(row["steps"])
            row["results"] = json.loads(row["results"]) if row["results"] else None
        return rows

    def get_completed_assessments(self, user_id: int, limit: int = 100) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM assessments WHERE user_id = ? AND status = 'completed' ORDER BY completed_at DESC LIMIT ?",
            (user_id, limit),
        )
        for row in rows:
            row["steps"] = json.loads(row["steps"])
            row["results"] = json.loads(row["results"]) if row["results"] else None
        return rows
