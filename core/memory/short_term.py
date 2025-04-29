"""
Short-term memory implementation.
Handles temporary storage of information with limited capacity and duration.
"""
import time
from collections import OrderedDict
import threading


class ShortTermMemory:
    def __init__(self, capacity=7, retention_time=300):
        """
        Initialize short-term memory with specified capacity and retention time.
        
        Args:
            capacity (int): Maximum number of items to store (Miller's law suggests ~7)
            retention_time (int): Time in seconds before items decay (default 5 minutes)
        """
        self.capacity = capacity
        self.retention_time = retention_time
        self.memory_store = OrderedDict()
        self.lock = threading.RLock()
        
        # Start the decay thread
        self._start_decay_thread()
    
    def store(self, key, value, importance=1):
        """Store a new item in short-term memory."""
        with self.lock:
            # If we're at capacity, make room by removing least important or oldest item
            if len(self.memory_store) >= self.capacity and key not in self.memory_store:
                self._make_room()
            
            # Store the new item with timestamp and importance
            self.memory_store[key] = {
                'value': value,
                'timestamp': time.time(),
                'importance': min(max(importance, 1), 10),  # Clamp between 1-10
                'access_count': 0
            }
            
            # Move to end to indicate it's the most recently used
            self.memory_store.move_to_end(key)
            return True
    
    def retrieve(self, key):
        """Retrieve an item from short-term memory."""
        with self.lock:
            if key in self.memory_store:
                item = self.memory_store[key]
                
                # Check if the item has expired
                if time.time() - item['timestamp'] > self.retention_time:
                    self.memory_store.pop(key)
                    return None
                
                # Update access metrics and move to end (most recently used)
                item['access_count'] += 1
                item['timestamp'] = time.time()  # Reset decay timer on access
                self.memory_store.move_to_end(key)
                
                return item['value']
            return None
    
    def get_all(self):
        """Return all non-expired items in short-term memory."""
        with self.lock:
            current_time = time.time()
            active_items = {}
            
            for key, item in list(self.memory_store.items()):
                if current_time - item['timestamp'] <= self.retention_time:
                    active_items[key] = item['value']
                else:
                    self.memory_store.pop(key)
                    
            return active_items
    
    def _make_room(self):
        """Remove the least important or oldest item to make room for new items."""
        if not self.memory_store:
            return
            
        # Find the item with lowest importance
        min_importance = float('inf')
        min_key = None
        
        for key, item in self.memory_store.items():
            if item['importance'] < min_importance:
                min_importance = item['importance']
                min_key = key
        
        # If all items have same importance, remove the oldest (first in the OrderedDict)
        if min_key is None:
            min_key = next(iter(self.memory_store))
            
        self.memory_store.pop(min_key)
    
    def _start_decay_thread(self):
        """Start a background thread to handle memory decay."""
        def decay_process():
            while True:
                time.sleep(10)  # Check every 10 seconds
                self._decay_old_items()
        
        thread = threading.Thread(target=decay_process, daemon=True)
        thread.start()
    
    def _decay_old_items(self):
        """Remove items that have exceeded retention time."""
        with self.lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, item in self.memory_store.items():
                if current_time - item['timestamp'] > self.retention_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.memory_store.pop(key)