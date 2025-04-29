"""
Short Term Memory Implementation

This module handles short-term memory functionality which includes:
- Temporary storage of recent information
- Quick access and retrieval
- Limited capacity with automatic pruning
"""

import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple

class ShortTermMemory:
    def __init__(self, capacity: int = 100, retention_period: float = 3600):
        """
        Initialize short-term memory with configurable capacity and retention period
        
        Args:
            capacity: Maximum number of items to store
            retention_period: Time in seconds before memories start to decay (default: 1 hour)
        """
        self.capacity = capacity
        self.retention_period = retention_period
        self.memories = OrderedDict()  # Uses insertion order for automatic LRU behavior
        self.timestamps = {}  # Track when each memory was added
        self.importance_scores = {}  # Score from 0-1 for each memory
    
    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """
        Store a new memory item
        
        Args:
            key: Unique identifier for the memory
            value: The content to store
            importance: Score from 0-1 indicating memory importance (higher = more important)
        """
        # If memory exists, update it
        if key in self.memories:
            self.memories.pop(key)  # Remove to re-add at the end (most recent)
        
        # Add new memory
        self.memories[key] = value
        self.timestamps[key] = time.time()
        self.importance_scores[key] = importance
        
        # Prune if we exceed capacity
        self._prune_if_needed()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory item by key
        
        Args:
            key: The identifier for the memory to retrieve
            
        Returns:
            The memory value or None if not found or expired
        """
        if key in self.memories:
            # Check if memory has expired
            if self._is_memory_active(key):
                # Move to end (most recently used)
                value = self.memories.pop(key)
                self.memories[key] = value
                return value
            else:
                # Memory has expired, remove it
                self._remove_memory(key)
        
        return None
    
    def get_recent_memories(self, limit: int = 10) -> List[Tuple[str, Any]]:
        """
        Get most recently added memories
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of (key, value) tuples of recent memories
        """
        result = []
        count = 0
        
        # Start from the most recent (end of OrderedDict)
        for key, value in reversed(self.memories.items()):
            if count >= limit:
                break
                
            if self._is_memory_active(key):
                result.append((key, value))
                count += 1
            else:
                # Remove expired memories during retrieval
                self._remove_memory(key)
        
        return result
    
    def search(self, query: str) -> List[Tuple[str, Any, float]]:
        """
        Simple search function for memory content
        
        Args:
            query: Search string to look for in memory keys or values
            
        Returns:
            List of (key, value, relevance) tuples matching the query
        """
        results = []
        
        for key, value in list(self.memories.items()):
            # Skip expired memories
            if not self._is_memory_active(key):
                self._remove_memory(key)
                continue
                
            relevance = 0.0
            
            # Check if query appears in key
            if isinstance(key, str) and query.lower() in key.lower():
                relevance = 0.8
            
            # Check if query appears in value
            if isinstance(value, str) and query.lower() in value.lower():
                relevance = max(relevance, 0.9)
            
            # Add relevance boost based on importance and recency
            if relevance > 0:
                age_factor = self._get_age_factor(key)
                importance = self.importance_scores.get(key, 0.5)
                final_relevance = relevance * (0.7 * age_factor + 0.3 * importance)
                results.append((key, value, final_relevance))
        
        # Sort by relevance (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def clear(self) -> None:
        """Clear all short-term memories"""
        self.memories.clear()
        self.timestamps.clear()
        self.importance_scores.clear()
    
    def _is_memory_active(self, key: str) -> bool:
        """Check if a memory is still active based on its age and importance"""
        if key not in self.timestamps:
            return False
            
        age = time.time() - self.timestamps[key]
        importance = self.importance_scores.get(key, 0.5)
        
        # Important memories last longer
        adjusted_retention = self.retention_period * (0.5 + importance)
        
        return age < adjusted_retention
    
    def _get_age_factor(self, key: str) -> float:
        """Calculate age factor (1.0 = new, 0.0 = expired)"""
        if key not in self.timestamps:
            return 0.0
            
        age = time.time() - self.timestamps[key]
        importance = self.importance_scores.get(key, 0.5)
        adjusted_retention = self.retention_period * (0.5 + importance)
        
        # Linear decay from 1.0 to 0.0 as memory ages
        age_factor = 1.0 - min(1.0, age / adjusted_retention)
        return max(0.0, age_factor)
    
    def _remove_memory(self, key: str) -> None:
        """Remove a memory and its metadata"""
        if key in self.memories:
            self.memories.pop(key)
        if key in self.timestamps:
            self.timestamps.pop(key)
        if key in self.importance_scores:
            self.importance_scores.pop(key)
    
    def _prune_if_needed(self) -> None:
        """Prune memories if capacity is exceeded"""
        while len(self.memories) > self.capacity:
            # First remove any expired memories
            for key in list(self.memories.keys()):
                if not self._is_memory_active(key):
                    self._remove_memory(key)
                    if len(self.memories) <= self.capacity:
                        return
                        
            if len(self.memories) <= self.capacity:
                return
                
            # If still over capacity, remove LRU item with lowest importance
            candidates = list(self.memories.keys())[:int(self.capacity * 0.2) + 1]
            if candidates:
                least_important = min(candidates, key=lambda k: self.importance_scores.get(k, 0))
                self._remove_memory(least_important)
            else:
                # Fallback: remove oldest item
                oldest_key = next(iter(self.memories))
                self._remove_memory(oldest_key)