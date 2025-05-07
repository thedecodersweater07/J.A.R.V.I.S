"""
Long-term memory implementation.
Handles persistent storage, organization, and retrieval of information.
"""
import os
import json
import time
import sqlite3
from config.database import DATABASE_PATHS


class LongTermMemory:
    def __init__(self, db_path=None):
        """
        Initialize long-term memory with a SQLite database for storage.
        
        Args:
            db_path (str): Optional custom path to the SQLite database file
        """
        self.db_path = db_path or DATABASE_PATHS["memory"]
        self._initialize_db()
    
    def _initialize_db(self):
        """Create the necessary database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memory items table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER DEFAULT 0,
            importance REAL DEFAULT 1.0
        )
        ''')
        
        # Create tags table for categorizing memories
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            tag TEXT,
            FOREIGN KEY (memory_id) REFERENCES memory_items (id) ON DELETE CASCADE,
            UNIQUE (memory_id, tag)
        )
        ''')
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_items (key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag ON memory_tags (tag)')
        
        conn.commit()
        conn.close()
    
    def store(self, key, value, tags=None, importance=1.0):
        """Store information in long-term memory."""
        if tags is None:
            tags = []
            
        # Serialize complex values to JSON
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            value = json.dumps(value)
            
        current_time = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or replace the memory item
            cursor.execute('''
            INSERT OR REPLACE INTO memory_items 
            (key, value, created_at, last_accessed, access_count, importance)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (key, value, current_time, current_time, 0, min(max(importance, 1), 10)))
            
            memory_id = cursor.lastrowid if cursor.lastrowid else cursor.execute(
                "SELECT id FROM memory_items WHERE key = ?", (key,)).fetchone()[0]
            
            # Clear existing tags and add new ones
            cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            
            for tag in tags:
                cursor.execute('''
                INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)
                ''', (memory_id, tag))
                
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error storing in long-term memory: {e}")
            if conn:
                conn.rollback()
            return False
            
        finally:
            if conn:
                conn.close()
    
    def retrieve(self, key):
        """Retrieve an item from long-term memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the memory item
            cursor.execute('''
            SELECT id, value, created_at FROM memory_items WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            if not result:
                return None
                
            memory_id, value, created_at = result
            
            # Update access metrics
            current_time = time.time()
            cursor.execute('''
            UPDATE memory_items 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
            ''', (current_time, memory_id))
            
            # Get tags
            cursor.execute('SELECT tag FROM memory_tags WHERE memory_id = ?', (memory_id,))
            tags = [row[0] for row in cursor.fetchall()]
            
            conn.commit()
            
            # Try to deserialize JSON values
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, keep as is
                pass
                
            return {
                'value': value,
                'created_at': created_at,
                'tags': tags
            }
            
        except Exception as e:
            print(f"Error retrieving from long-term memory: {e}")
            return None
            
        finally:
            if conn:
                conn.close()
    
    def search_by_tags(self, tags, match_all=True):
        """Search memory items by tags."""
        if not tags:
            return {}
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if match_all:
                # All tags must match (items with all specified tags)
                query = '''
                SELECT m.key, m.value FROM memory_items m
                JOIN memory_tags t ON m.id = t.memory_id
                WHERE t.tag IN ({})
                GROUP BY m.id
                HAVING COUNT(DISTINCT t.tag) = ?
                '''.format(','.join(['?'] * len(tags)))
                cursor.execute(query, tags + [len(tags)])
            else:
                # Any tag matches (items with any of the specified tags)
                query = '''
                SELECT DISTINCT m.key, m.value FROM memory_items m
                JOIN memory_tags t ON m.id = t.memory_id
                WHERE t.tag IN ({})
                '''.format(','.join(['?'] * len(tags)))
                cursor.execute(query, tags)
                
            results = {}
            for key, value in cursor.fetchall():
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                results[key] = value
                
            return results
            
        except Exception as e:
            print(f"Error searching by tags: {e}")
            return {}
            
        finally:
            if conn:
                conn.close()
    
    def get_all_tags(self):
        """Get all unique tags in the memory system."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT tag FROM memory_tags")
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Error getting tags: {e}")
            return []
            
        finally:
            if conn:
                conn.close()
                
    def delete(self, key):
        """Delete a specific memory item."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM memory_items WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Error deleting from long-term memory: {e}")
            return False
            
        finally:
            if conn:
                conn.close()