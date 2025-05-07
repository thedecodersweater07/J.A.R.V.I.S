import sqlite3
import json
import yaml
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="jarvis.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        with open('schema.sql', 'r') as f:
            self.conn.executescript(f.read())
            
    def load_metrics(self):
        with open('metrics.json', 'r') as f:
            return json.load(f)
            
    def save_conversation(self, user_id, message, response):
        sql = """INSERT INTO conversations (user_id, timestamp, message, response)
                VALUES (?, datetime('now'), ?, ?)"""
        self.conn.execute(sql, (user_id, message, response))
        self.conn.commit()
