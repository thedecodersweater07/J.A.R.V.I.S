"""Database connection utility"""
from pymongo import MongoClient
from typing import Optional

class Database:
    _instance = None
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Database()
        return cls._instance
    
    def connect(self, connection_string: str = "mongodb://localhost:27017/"):
        """Connect to MongoDB"""
        self.client = MongoClient(connection_string)
        return self.client
        
    def get_client(self) -> MongoClient:
        """Get MongoDB client"""
        if not self.client:
            self.connect()
        return self.client
