import logging
from typing import Dict, List, Optional, Any
import sqlite3
import json

logger = logging.getLogger(__name__)

class KnowledgeBaseConnector:
    """Handles connections to various knowledge bases."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.connections = {}
        self.cache = {}
        
    def connect(self, db_type: str, connection_string: str) -> bool:
        """Establish connection to a knowledge base."""
        try:
            if db_type == "sqlite":
                self.connections[db_type] = sqlite3.connect(connection_string)
            elif db_type == "json":
                with open(connection_string, 'r') as f:
                    self.connections[db_type] = json.load(f)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {db_type} database: {e}")
            return False

    def query(self, query: str, db_type: str, cache: bool = True) -> Optional[List[Dict]]:
        """Query the knowledge base."""
        if cache and query in self.cache:
            return self.cache[query]
            
        try:
            if db_type == "sqlite":
                cursor = self.connections[db_type].cursor()
                results = cursor.execute(query).fetchall()
            elif db_type == "json":
                results = self._query_json(query)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            if cache:
                self.cache[query] = results
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def _query_json(self, query: str) -> List[Dict]:
        """Execute query on JSON data."""
        # Simple implementation - could be enhanced with JMESPath or similar
        data = self.connections["json"]
        if isinstance(data, list):
            return [item for item in data if query.lower() in str(item).lower()]
        return []

    def update_knowledge(self, data: Dict, db_type: str) -> bool:
        """Update or insert new knowledge."""
        try:
            if db_type == "sqlite":
                cursor = self.connections[db_type].cursor()
                table = data.get("table")
                values = data.get("values")
                query = f"INSERT OR REPLACE INTO {table} VALUES ({','.join(['?']*len(values))})"
                cursor.execute(query, values)
                self.connections[db_type].commit()
            elif db_type == "json":
                self.connections["json"].append(data)
            return True
        except Exception as e:
            logger.error(f"Failed to update knowledge: {e}")
            return False
