"""Database connection utility"""
import os
import sqlite3
from typing import Optional, Union, Any
from pathlib import Path

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("MongoDB support not available. Using SQLite as fallback.")

class Database:
    _instance = None
    
    def __init__(self):
        self.mongo_client: Optional[Any] = None
        self.sqlite_connections = {}
        self.db_root = Path(__file__).parent.parent / "data" / "db"
        self.db_root.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Database()
        return cls._instance
    
    def connect(self, connection_string: str = "mongodb://localhost:27017/"):
        """Connect to database (MongoDB if available, otherwise SQLite)"""
        if MONGO_AVAILABLE:
            try:
                self.mongo_client = MongoClient(connection_string)
                return self.mongo_client
            except Exception as e:
                print(f"MongoDB connection failed: {e}. Using SQLite as fallback.")
                
        # SQLite fallback
        return self._get_sqlite_connection("main")
        
    def get_client(self) -> Union[Any, sqlite3.Connection]:
        """Get database client"""
        if MONGO_AVAILABLE and self.mongo_client:
            return self.mongo_client
        
        if not self.sqlite_connections.get("main"):
            return self._get_sqlite_connection("main")
            
        return self.sqlite_connections["main"]
        
    def _get_sqlite_connection(self, name: str) -> sqlite3.Connection:
        """Get or create SQLite connection"""
        if name not in self.sqlite_connections:
            db_path = self.db_root / f"{name}.db"
            self.sqlite_connections[name] = sqlite3.connect(str(db_path))
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
