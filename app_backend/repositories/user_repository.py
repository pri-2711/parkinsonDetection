from __future__ import annotations

from app_backend.database import Database


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_user(self, username: str, email: str, pw_hash: str) -> dict:
        return self.db.create_user(username, email, pw_hash)

    def get_user_by_username(self, username: str) -> dict | None:
        return self.db.get_user_by_username(username)

    def get_user_by_email(self, email: str) -> dict | None:
        return self.db.get_user_by_email(email)

    def get_user_by_id(self, user_id: int) -> dict | None:
        return self.db.get_user_by_id(user_id)
