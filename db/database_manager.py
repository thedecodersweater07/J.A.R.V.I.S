from typing import Dict, Any, Optional
import sqlite3
import logging
from pathlib import Path
from core.file_manager import FileManager

logger = logging.getLogger(__name__)

# Database paths configuration
DATABASE_PATHS = {
    "auth": "data/db/auth/auth.db",
    "knowledge": "data/db/knowledge/knowledge.db",
    "memory": "data/db/memory/memory.db",
    "feedback": "data/db/feedback/feedback.db",
    "cache": "data/db/cache/cache.db"
}

class DatabaseManager:
    def __init__(self):
        self.db_root = Path(__file__).parent.parent / "data" / "db"
        self.db_root.mkdir(parents=True, exist_ok=True)
        self.connections = {}
        self.initialize_databases()
        
    def initialize_databases(self):
        """Initialize all required databases"""
        databases = {
            "auth": self._create_auth_db,
            "knowledge": self._create_knowledge_db,
            "memory": self._create_memory_db,
            "learning": self._create_learning_db,
            "system": self._create_system_db
        }
        
        for db_name, create_func in databases.items():
            db_dir = self.db_root / db_name
            db_dir.mkdir(parents=True, exist_ok=True)
            db_file = db_dir / f"{db_name}.db"
            create_func(db_file)

    def _create_auth_db(self, path: Path):
        """Create auth database schema"""
        try:
            conn = sqlite3.connect(str(path))
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
            self.connections["auth"] = conn
        except Exception as e:
            logger.error(f"Failed to create auth database: {e}")
            raise

    def _create_knowledge_db(self, path: Path):
        """Create knowledge database schema"""
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT,
                content TEXT,
                source TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        self.connections["knowledge"] = conn

    def _create_memory_db(self, path: Path):
        """Create memory database schema"""
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        self.connections["memory"] = conn

    def _create_learning_db(self, path: Path):
        """Create learning database schema"""
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning (
                id TEXT PRIMARY KEY,
                model TEXT,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        self.connections["learning"] = conn

    def _create_system_db(self, path: Path):
        """Create system database schema"""
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system (
                id TEXT PRIMARY KEY,
                setting TEXT,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        self.connections["system"] = conn

    def get_connection(self, db_name: str) -> Optional[sqlite3.Connection]:
        """Get a database connection"""
        return self.connections.get(db_name)

    def close_connection(self, db_name: str):
        """Close a database connection"""
        conn = self.connections.pop(db_name, None)
        if conn:
            conn.close()
