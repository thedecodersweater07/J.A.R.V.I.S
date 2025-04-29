"""
Memory indexer for efficient information retrieval.
Creates searchable indexes of memory content.
"""
import re
import json
import sqlite3
import time
import threading


class MemoryIndexer:
    def __init__(self, db_path="memory_index.db"):
        """
        Initialize the memory indexer with a SQLite FTS (Full-Text Search) database.
        
        Args:
            db_path (str): Path to the index database file
        """
        self.db_path = db_path
        self._initialize_index()
        self.indexing_lock = threading.RLock()
        
    def _initialize_index(self):
        """Initialize the database and create necessary tables for indexing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create a virtual table using FTS5 for full-text search capabilities
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_index USING fts5(
            key,
            content,
            metadata,
            tokenize='porter unicode61'
        )
        ''')
        
        # Create a table to track when items were last indexed
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            indexed_at REAL,
            source TEXT,
            last_modified REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def index_memory(self, key, content, metadata=None, source="generic"):
        """Index memory content for efficient searching."""
        if metadata is None:
            metadata = {}
            
        with self.indexing_lock:
            try:
                # Convert content to string if it's not already
                if not isinstance(content, str):
                    if isinstance(content, (dict, list)):
                        content = json.dumps(content)
                    else:
                        content = str(content)
                
                # Serialize metadata to JSON
                metadata_json = json.dumps(metadata)
                current_time = time.time()
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if this key already exists in the index
                cursor.execute("SELECT key FROM memory_index WHERE key = ?", (key,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing index
                    cursor.execute('''
                    UPDATE memory_index SET content = ?, metadata = ? WHERE key = ?
                    ''', (content, metadata_json, key))
                else:
                    # Insert new index entry
                    cursor.execute('''
                    INSERT INTO memory_index (key, content, metadata) VALUES (?, ?, ?)
                    ''', (key, content, metadata_json))
                
                # Update the index metadata
                cursor.execute('''
                INSERT OR REPLACE INTO index_metadata 
                (key, indexed_at, source, last_modified) 
                VALUES (?, ?, ?, ?)
                ''', (key, current_time, source, current_time))
                
                conn.commit()
                return True
                
            except Exception as e:
                print(f"Error indexing memory: {e}")
                if conn:
                    conn.rollback()
                return False
                
            finally:
                if conn:
                    conn.close()
    
    def search(self, query, limit=10):
        """Search the memory index for relevant information."""
        try:
            # Clean the query for FTS
            clean_query = self._sanitize_query(query)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute FTS query with ranking
            cursor.execute('''
            SELECT 
                key, 
                content, 
                metadata, 
                rank
            FROM memory_index
            WHERE memory_index MATCH ?
            ORDER BY rank
            LIMIT ?
            ''', (clean_query, limit))
            
            results = []
            for key, content, metadata_json, rank in cursor.fetchall():
                try:
                    metadata = json.loads(metadata_json)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                # Get the source and last modified time
                cursor.execute('''
                SELECT source, last_modified FROM index_metadata WHERE key = ?
                ''', (key,))
                
                source_info = cursor.fetchone()
                source = source_info[0] if source_info else "unknown"
                last_modified = source_info[1] if source_info else None
                
                results.append({
                    'key': key,
                    'content': content,
                    'metadata': metadata,
                    'source': source,
                    'last_modified': last_modified,
                    'relevance': rank
                })
                
            return results
            
        except Exception as e:
            print(f"Error searching memory index: {e}")
            return []
            
        finally:
            if conn:
                conn.close()
    
    def remove_from_index(self, key):
        """Remove an item from the memory index."""
        with self.indexing_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM memory_index WHERE key = ?", (key,))
                cursor.execute("DELETE FROM index_metadata WHERE key = ?", (key,))
                
                conn.commit()
                return True
                
            except Exception as e:
                print(f"Error removing from index: {e}")
                if conn:
                    conn.rollback()
                return False
                
            finally:
                if conn:
                    conn.close()
    
    def _sanitize_query(self, query):
        """Sanitize the search query for FTS5."""
        # Remove special characters that could cause problems with FTS
        query = re.sub(r'[^\w\s*"]', ' ', query)
        
        # Add wildcards for partial matching if no special FTS operators are used
        if '"' not in query and '*' not in query:
            terms = query.split()
            query = ' '.join([f"{term}*" for term in terms])
            
        return query
    
    def reindex_all(self, memory_store):
        """Reindex all items from a memory store."""
        with self.indexing_lock:
            try:
                items = memory_store.get_all()
                count = 0
                
                for key, value in items.items():
                    # Determine the appropriate content to index
                    if isinstance(value, dict) and 'value' in value:
                        content = value['value']
                        metadata = {k: v for k, v in value.items() if k != 'value'}
                    else:
                        content = value
                        metadata = {}
                    
                    if self.index_memory(key, content, metadata, source=memory_store.__class__.__name__):
                        count += 1
                        
                return count
                
            except Exception as e:
                print(f"Error reindexing memory: {e}")
                return 0