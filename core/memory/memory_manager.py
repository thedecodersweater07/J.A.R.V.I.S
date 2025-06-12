"""
Memory Manager
=============

Handles the creation, storage, and retrieval of memories.
"""

import os
from typing import Dict, Any, List, Optional
from enum import Enum
import json
from datetime import datetime
from pathlib import Path
import logging

from ..database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories that can be stored."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"

class MemoryManager:
    """Manages all memory operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the memory manager."""
        self.db_manager = db_manager
        self.memory_path = Path(__file__).parent.parent.parent / 'data' / 'memory'
        self._ensure_directories()
        logger.info("Memory Manager initialized")
    
    def _ensure_directories(self) -> None:
        """Ensure all necessary memory directories exist."""
        required_dirs = [
            self.memory_path / memory_type.value
            for memory_type in MemoryType
        ]
        
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created memory directory: {dir_path}")
    
    def create_memory(self, 
                     content: Any, 
                     memory_type: MemoryType, 
                     metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create and store a new memory.
        
        Args:
            content: The content of the memory
            memory_type: Type of memory (short_term, long_term, etc.)
            metadata: Additional metadata about the memory
            
        Returns:
            Dictionary containing memory details
        """
        memory = {
            'id': str(uuid.uuid4()),
            'type': memory_type.value,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }
        
        # Store in appropriate location
        self._store_memory(memory)
        
        return memory
    
    def _store_memory(self, memory: Dict[str, Any]) -> None:
        """Store memory in appropriate location."""
        try:
            # Store in database
            self.db_manager.distribute_data(memory, 'memory')
            
            # Store in filesystem cache
            self._cache_memory(memory)
            
            logger.info(f"Memory stored successfully: {memory['id']}")
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    def _cache_memory(self, memory: Dict[str, Any]) -> None:
        """Cache memory in filesystem."""
        memory_dir = self.memory_path / memory['type']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'memory_{memory["id"]}_{timestamp}.json'
        
        with open(memory_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    
    def retrieve_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query.
        
        Args:
            query: Dictionary containing search parameters
            
        Returns:
            List of matching memories
        """
        # First try to get from cache
        cached_memories = self._get_cached_memories(query)
        
        if not cached_memories:
            # If not in cache, get from database
            cached_memories = self._get_db_memories(query)
        
        return cached_memories
    
    def _get_cached_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get memories from filesystem cache."""
        matching_memories = []
        
        # Search through cached memory files
        for memory_type in MemoryType:
            memory_dir = self.memory_path / memory_type.value
            if not memory_dir.exists():
                continue
                
            for file in memory_dir.glob('*.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        memory = json.load(f)
                        if self._matches_query(memory, query):
                            matching_memories.append(memory)
                except Exception as e:
                    logger.warning(f"Error reading memory file {file}: {e}")
        
        return matching_memories
    
    def _get_db_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get memories from database."""
        # TODO: Implement database query
        return []
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update
        """
        # TODO: Implement memory update
        pass
    
    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
        """
        # TODO: Implement memory deletion
        pass
