from typing import Dict, Any, Optional
import sqlite3
import logging
from pathlib import Path
from core.file_manager import FileManager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.file_manager = FileManager()
        self.connections = {}
        self.initialize_databases()
        
    def initialize_databases(self):
        """Initialize all required databases"""
        db_path = self.file_manager.get_path("db")
        
        databases = {
            "knowledge": self._create_knowledge_db,
            "memory": self._create_memory_db,
            "learning": self._create_learning_db,
            "system": self._create_system_db
        }
        
        for db_name, create_func in databases.items():
            db_file = db_path / f"{db_name}.db"
            create_func(db_file)
            
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
