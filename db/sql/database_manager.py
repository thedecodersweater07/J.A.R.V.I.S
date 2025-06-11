from typing import Optional, Union, Any, Dict
import sqlite3
from pathlib import Path
from models.database import Database

class DatabaseManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = Database.get_instance()
        if self.config.get('path'):
            self.db.db_root = Path(self.config['path'])
        
    def get_connection(self) -> Union[Any, sqlite3.Connection]:
        return self.db.get_client()
        
    def execute_query(self, query: str, params: tuple = None) -> Any:
        return self.db.execute_query(query, params)
        
    def get_cached_query(self, query: str) -> Any:
        return self.db.get_cached_query(query)
