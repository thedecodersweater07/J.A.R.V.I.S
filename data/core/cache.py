import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import lru_cache
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

class CacheItem:
    """Class representing a cached item with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Initialize cache item
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds (None for no expiration)
        """
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        
    def is_expired(self) -> bool:
        """Check if item is expired"""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
        
    def access(self) -> None:
        """Update access metadata"""
        self.last_accessed = time.time()
        self.access_count += 1


class AICache:
    """
    Intelligent caching system with predictive preloading and adaptive TTL
    """
    
    def __init__(self, max_size: int = 100, default_ttl: Optional[int] = 3600):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time to live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.items: Dict[str, CacheItem] = OrderedDict()
        self.access_patterns: Dict[str, Dict[str, int]] = {}  # Track key access patterns
        self.lock = threading.RLock()
        self.preload_callbacks: Dict[str, Callable] = {}
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
    def get(self, key: str) -> Any:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            KeyError: If key not found
        """
        with self.lock:
            if key not in self.items:
                raise KeyError(f"Cache key not found: {key}")
                
            item = self.items[key]
            
            # Check expiration
            if item.is_expired():
                del self.items[key]
                raise KeyError(f"Cache key expired: {key}")
                
            # Update access metadata
            item.access()
            
            # Move to end (most recently used)
            self.items.move_to_end(key)
            
            # Update access patterns
            self._update_access_patterns(key)
            
            # Trigger preloading of related items
            self._preload_related(key)
            
            return item.value
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)
        """
        with self.lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
                
            # Create cache item
            item = CacheItem(key, value, ttl)
            
            # Add to cache
            self.items[key] = item
            
            # Move to end (most recently used)
            self.items.move_to_end(key)
            
            # Enforce max size
            self._enforce_max_size()
            
    def delete(self, key: str) -> bool:
        """
        Delete item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.items:
                del self.items[key]
                return True
            return False
            
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            self.items.clear()
            self.access_patterns.clear()
            
    def register_preload_callback(self, key_pattern: str, callback: Callable) -> None:
        """
        Register callback for preloading related items
        
        Args:
            key_pattern: Pattern to match keys
            callback: Function to call for preloading
        """
        with self.lock:
            self.preload_callbacks[key_pattern] = callback
            logger.debug(f"Registered preload callback for pattern: {key_pattern}")
            
    def _enforce_max_size(self) -> None:
        """Enforce maximum cache size by removing least recently used items"""
        while len(self.items) > self.max_size:
            # Remove first item (least recently used)
            self.items.popitem(last=False)
            
    def _update_access_patterns(self, key: str) -> None:
        """Update access pattern tracking"""
        # Get previous key accessed (if any)
        prev_keys = list(self.items.keys())
        if len(prev_keys) > 1:
            idx = prev_keys.index(key)
            if idx > 0:
                prev_key = prev_keys[idx-1]
                
                # Update access pattern count
                if prev_key not in self.access_patterns:
                    self.access_patterns[prev_key] = {}
                    
                if key not in self.access_patterns[prev_key]:
                    self.access_patterns[prev_key][key] = 0
                    
                self.access_patterns[prev_key][key] += 1
                
    def _preload_related(self, key: str) -> None:
        """Preload related items based on access patterns"""
        # Check if we have a pattern for this key
        if key in self.access_patterns:
            # Get most commonly accessed next keys
            next_keys = sorted(
                self.access_patterns[key].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Preload top related keys
            for next_key, count in next_keys[:3]:  # Preload top 3
                if count > 2 and next_key not in self.items:  # Only if accessed together more than twice
                    # Check for preload callback
                    for pattern, callback in self.preload_callbacks.items():
                        if pattern in next_key:
                            try:
                                # Execute callback in a separate thread
                                threading.Thread(
                                    target=self._execute_preload,
                                    args=(callback, next_key),
                                    daemon=True
                                ).start()
                                break
                            except Exception as e:
                                logger.error(f"Error in preload callback: {str(e)}")
                                
    def _execute_preload(self, callback: Callable, key: str) -> None:
        """Execute preload callback and cache result"""
        try:
            result = callback(key)
            if result is not None:
                self.set(key, result)
                logger.debug(f"Preloaded cache key: {key}")
        except Exception as e:
            logger.error(f"Error preloading cache key {key}: {str(e)}")
            
    def _cleanup_loop(self) -> None:
        """Background thread for cleaning up expired items"""
        while True:
            try:
                # Sleep for a while
                time.sleep(60)  # Check every minute
                
                # Clean up expired items
                with self.lock:
                    expired_keys = [
                        key for key, item in self.items.items()
                        if item.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self.items[key]
                        
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
                        
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")


class PredictiveCache:
    """
    Cache with predictive loading based on access patterns and ML
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize predictive cache
        
        Args:
            max_size: Maximum number of items in cache
        """
        self.cache = AICache(max_size=max_size)
        self.access_history: List[str] = []
        self.max_history = 1000
        self.prediction_model = None
        self.feature_vectors: Dict[str, np.ndarray] = {}
        
    def get(self, key: str) -> Any:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value
            
        Raises:
            KeyError: If key not found
        """
        # Record access
        self._record_access(key)
        
        # Get from cache
        return self.cache.get(key)
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        self.cache.set(key, value, ttl)
        
    def _record_access(self, key: str) -> None:
        """Record key access in history"""
        self.access_history.append(key)
        
        # Trim history if needed
        if len(self.access_history) > self.max_history:
            self.access_history = self.access_history[-self.max_history:]
            
        # Update prediction model periodically
        if len(self.access_history) % 100 == 0:
            self._update_prediction_model()
            
    def _update_prediction_model(self) -> None:
        """Update prediction model based on access history"""
        # This would be implemented with a simple ML model
        # For now, we'll use the built-in pattern tracking in AICache
        pass
