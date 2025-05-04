import logging
from typing import Dict, List, Optional, Any
import sqlite3
import json
from config.database import DATABASE_PATHS

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    def __init__(self):
        self.entities = {}
        self.relationships = []
        
    def add_entity(self, entity_id: str, data: Dict[str, Any]):
        self.entities[entity_id] = data
        
    def add_relationship(self, source: str, target: str, relationship_type: str):
        self.relationships.append({
            'source': source,
            'target': target,
            'type': relationship_type
        })

class KnowledgeBaseConnector:
    """Handles connections to various knowledge bases."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.connections = {}
        self.cache = {}
        self.db_path = DATABASE_PATHS["knowledge"]
        self.graph_manager = KnowledgeGraphManager()
        self.query_patterns = self._load_query_patterns()
        
    def _ensure_tables(self):
        """Ensure all required tables exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            query TEXT PRIMARY KEY,
            result TEXT,
            timestamp REAL,
            db_type TEXT
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stored_data (
            key TEXT PRIMARY KEY,
            value TEXT,
            db_type TEXT,
            timestamp REAL
        )''')
        
        conn.commit()
        conn.close()
    
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
            self._ensure_tables()
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

    def _load_query_patterns(self) -> Dict[str, str]:
        """Load predefined query patterns"""
        return {
            "entity_search": """
                SELECT * FROM entities 
                WHERE content MATCH ? 
                ORDER BY rank
            """,
            "relationship_query": """
                SELECT * FROM relationships 
                WHERE source = ? OR target = ?
            """,
            "fact_verification": """
                SELECT * FROM facts 
                WHERE statement MATCH ? 
                AND confidence > 0.8
            """
        }

    def advanced_query(self, query_type: str, params: List[Any]) -> Optional[List[Dict]]:
        """Execute advanced queries using predefined patterns"""
        try:
            if query_type not in self.query_patterns:
                raise ValueError(f"Unknown query type: {query_type}")

            query = self.query_patterns[query_type]
            conn = sqlite3.connect(self.db_path)
            conn.create_function("MATCH", 2, self._fuzzy_match)
            
            cursor = conn.cursor()
            results = cursor.execute(query, params).fetchall()
            
            # Convert to knowledge graph entities
            for result in results:
                self._add_to_graph(result)
                
            return results
        except Exception as e:
            logger.error(f"Advanced query failed: {e}")
            return None

    def _fuzzy_match(self, text: str, pattern: str) -> bool:
        """Custom fuzzy matching function for SQLite"""
        # Implement fuzzy matching logic here
        return pattern.lower() in text.lower()

    def _add_to_graph(self, data: Dict[str, Any]):
        """Add query results to knowledge graph"""
        if 'entity_id' in data:
            self.graph_manager.add_entity(data['entity_id'], data)
        if 'source' in data and 'target' in data:
            self.graph_manager.add_relationship(
                data['source'], 
                data['target'],
                data.get('relationship_type', 'generic')
            )
