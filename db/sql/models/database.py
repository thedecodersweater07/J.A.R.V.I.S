"""Database connection utility"""
import os
import sqlite3
from typing import Union, Any
from pathlib import Path
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("MongoDB support not available. Using SQLite as fallback.")

class Database:
    _instance = None
    
    def __init__(self):
        if Database._instance is not None:
            raise RuntimeError("Use get_instance() instead")
        self.client = None
        self.mongo_client = None
        self.sqlite_connections = {}
        self.db_path = str(Path(__file__).parent.parent.parent.parent / "data" / "db")
        self._ensure_directories()
        self._init_connection_pool()
        self._connect()

    def _ensure_directories(self):
        """Ensure database directories exist"""
        base_path = Path(self.db_path)
        base_path.mkdir(parents=True, exist_ok=True)
        for subdir in ["auth", "knowledge", "memory", "feedback", "cache"]:
            (base_path / subdir).mkdir(exist_ok=True)

    def _init_connection_pool(self):
        """Initialize connection pool for better performance"""
        if MONGO_AVAILABLE:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(
                maxPoolSize=50,
                connectTimeoutMS=2000,
                retryWrites=True
            )
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_client(self) -> Union[Any, sqlite3.Connection]:
        """Get database client"""
        if MONGO_AVAILABLE and self.mongo_client:
            return self.mongo_client
        
        if not self.client:
            self._connect()
        return self.client
        
    def _connect(self):
        """Initialize database connection"""
        try:
            db_path = Path(self.db_path) / "main.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.client = sqlite3.connect(str(db_path))
            self.client.row_factory = sqlite3.Row
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
        
    def _get_sqlite_connection(self, name: str) -> sqlite3.Connection:
        """Get or create SQLite connection"""
        if name not in self.sqlite_connections:
            db_path = Path(self.db_path) / f"{name}" / f"{name}.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.sqlite_connections[name] = sqlite3.connect(str(db_path))
            self.sqlite_connections[name].row_factory = sqlite3.Row
            self._init_sqlite_schema(self.sqlite_connections[name])
        return self.sqlite_connections[name]
        
    def _init_sqlite_schema(self, conn: sqlite3.Connection):
        """Initialize SQLite schema"""
        cursor = conn.cursor()
        
        # Create necessary tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding BLOB,
            created_at TIMESTAMP,
            last_accessed TIMESTAMP
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT,
            response TEXT,
            timestamp TIMESTAMP,
            analyzed BOOLEAN
        )''')
        
        conn.commit()
    
    @lru_cache(maxsize=1000)
    def get_cached_query(self, query: str) -> Any:
        """Cache frequently used query results"""
        return self.execute_query(query)
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute database query with connection pooling"""
        conn = self.get_client()
        try:
            cursor = conn.cursor()
            if params:
                result = cursor.execute(query, params)
            else:
                result = cursor.execute(query)
            return result.fetchall()
        except Exception as e:
            print(f"Query error: {e}")
            return None
