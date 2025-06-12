"""
Memory Storage Interface
======================

Defines the interface for memory storage systems.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class MemoryStorage(ABC):
    """Abstract base class for memory storage systems."""
    
    @abstractmethod
    def store(self, memory: Dict[str, Any]) -> str:
        """
        Store a memory.
        
        Args:
            memory: Dictionary containing memory data
            
        Returns:
            ID of the stored memory
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query.
        
        Args:
            query: Dictionary containing search parameters
            
        Returns:
            List of matching memories
        """
        pass
    
    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
        """
        pass

class FileSystemStorage(MemoryStorage):
    """Memory storage using filesystem."""
    
    def __init__(self, base_path: str):
        """Initialize file system storage."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def store(self, memory: Dict[str, Any]) -> str:
        """Store memory in filesystem."""
        memory_dir = self.base_path / memory['type']
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'memory_{memory["id"]}_{timestamp}.json'
        
        with open(memory_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        
        return memory['id']
    
    def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories from filesystem."""
        matching_memories = []
        
        for memory_type in MemoryType:
            memory_dir = self.base_path / memory_type.value
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
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """Update memory in filesystem."""
        # TODO: Implement memory update
        pass
    
    def delete(self, memory_id: str) -> None:
        """Delete memory from filesystem."""
        # TODO: Implement memory deletion
        pass
    
    def _matches_query(self, memory: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if memory matches query."""
        for key, value in query.items():
            if key not in memory or memory[key] != value:
                return False
        return True
