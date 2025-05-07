from typing import Optional, Union, Any, Dict
import sqlite3
import logging
from pathlib import Path
from .sql.models.database import Database
from .models import Base, Conversation, SystemLog

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Unified database management interface"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = Database.get_instance()
        self._ensure_directories()
        
        # Apply configuration
        if self.config.get('path'):
            self.db.db_root = Path(self.config['path'])
            self.db.db_root.mkdir(parents=True, exist_ok=True)
            
    def _ensure_directories(self):
        """Ensure required database directories exist"""
        db_root = Path(__file__).parent / "data"
        for dir_name in ["cache", "main", "logs"]:
            (db_root / dir_name).mkdir(parents=True, exist_ok=True)
            
    def get_connection(self, db_name: str = "main") -> Union[Any, sqlite3.Connection]:
        """Get database connection with automatic initialization"""
        conn = self.db.get_client()
        if isinstance(conn, sqlite3.Connection):
            self._init_tables(conn)
        return conn
            
    def _init_tables(self, conn: sqlite3.Connection):
        """Initialize database tables"""
        cursor = conn.cursor()
        
        # Core tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence REAL,
                context_used TEXT,
                processing_time REAL,
                model_name VARCHAR(50)
            );
            
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level VARCHAR(10),
                message TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);
        """)
        conn.commit()
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute database query with error handling"""
        try:
            return self.db.execute_query(query, params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
            
    def save_conversation(self, 
                         user_input: str, 
                         response: str,
                         confidence: float = None,
                         context: str = None,
                         processing_time: float = None,
                         model_name: str = None) -> bool:
        """Save conversation to database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (user_input, response, confidence, context_used, processing_time, model_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_input, response, confidence, context, processing_time, model_name))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
            
    def get_cached_query(self, query: str) -> Any:
        """Get cached query result"""
        return self.db.get_cached_query(query)
