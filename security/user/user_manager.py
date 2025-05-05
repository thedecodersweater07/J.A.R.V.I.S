import sqlite3
import bcrypt
from uuid import uuid4
import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, db_path: str = "data/db/users.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize users database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            last_login TIMESTAMP,
            created_at TIMESTAMP,
            status TEXT
        )""")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_permissions (
            user_id TEXT,
            permission TEXT,
            granted_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        conn.commit()
        conn.close()

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        try:
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), username, password_hash, role, None, datetime.now(), "active")
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
