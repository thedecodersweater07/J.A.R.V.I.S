import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import bcrypt
from db.database_manager import DATABASE_PATHS, DatabaseManager
from db.sql.models.database import Database
from ..models.user import User
from ..config.security_config import SecurityConfig
from .auth_types import LoginResponse
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class AuthService:
    """Handles authentication and authorization"""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.db = Database.get_instance()
        self.db_manager = DatabaseManager()
        self._init_auth_db()
        logger.info("AuthService initialized")

    def _init_auth_db(self):
        """Initialize auth tables"""
        try:
            conn = self.db._get_sqlite_connection("auth")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to setup auth database: {e}")

    def authenticate(self, username: str, password: str, ip_address: str = "") -> LoginResponse:
        """Authenticate user and return login response"""
        if self._is_account_locked(username):
            return LoginResponse(success=False, error="Account is locked")

        user = self._verify_credentials(username, password)
        if not user:
            self._handle_failed_attempt(username, ip_address)
            return LoginResponse(success=False, error="Invalid credentials")

        token = self._generate_token(user)
        self._log_attempt(username, True, ip_address)
        
        return LoginResponse(
            success=True,
            token=token,
            user_info=user
        )

    def _is_account_locked(self, username: str) -> bool:
        conn = self.db._get_sqlite_connection("auth")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM auth_attempts WHERE username = ? AND timestamp > ? AND success = 0",
            (username, datetime.now() - timedelta(minutes=self.config.lockout_duration_minutes))
        )
        return cursor.fetchone()[0] >= self.config.max_login_attempts

    def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        conn = self.db._get_sqlite_connection("users")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode(), user[2]):  # index 2 is password_hash
            return {"id": user[0], "username": user[1], "role": user[3]}
        return None

    def _generate_token(self, user: Dict[str, Any]) -> str:
        payload = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours)
        }
        return jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")

    def _log_attempt(self, username: str, success: bool, ip_address: str):
        conn = self.db._get_sqlite_connection("auth")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO auth_attempts (username, timestamp, success, ip_address) VALUES (?, ?, ?, ?)",
            (username, datetime.now(), success, ip_address)
        )
        conn.commit()

    def _handle_failed_attempt(self, username: str, ip_address: str):
        self._log_attempt(username, False, ip_address)
        logger.warning(f"Failed login attempt for user {username} from IP {ip_address}")
