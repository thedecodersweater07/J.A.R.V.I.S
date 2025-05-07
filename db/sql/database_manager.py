from typing import Optional, Union, Any
import sqlite3
from .database import Database

class DatabaseManager:
    def __init__(self):
        self.db = Database.get_instance()
        
    def get_connection(self) -> Union[Any, sqlite3.Connection]:
        return self.db.get_client()
        
    def execute_query(self, query: str, params: tuple = None) -> Any:
        return self.db.execute_query(query, params)
        
    def get_cached_query(self, query: str) -> Any:
        return self.db.get_cached_query(query)
