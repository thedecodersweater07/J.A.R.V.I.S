# memory_manager.py
# Manages memory storage and retrieval for the Jarvis system

import logging
import json
import os
import time
from datetime import datetime

class MemoryManager:
    """Manages Jarvis's memory systems"""
    
    def __init__(self, memory_path="data/memory"):
        self.logger = logging.getLogger("MemoryManager")
        self.memory_path = memory_path
        self.running = False
        self.short_term_memory = []
        self.max_short_term_items = 100
        
        # Ensure memory directory exists
        os.makedirs(memory_path, exist_ok=True)
        os.makedirs(f"{memory_path}/interactions", exist_ok=True)
        os.makedirs(f"{memory_path}/knowledge", exist_ok=True)
        
        self.logger.info("Memory Manager initialized")
    
    def start(self):
        """Start the memory manager"""
        self.running = True
        self.logger.info("Memory Manager started")
        return True
    
    def stop(self):
        """Stop the memory manager"""
        # Flush short-term memory to disk before stopping
        self._flush_short_term_memory()
        self.running = False
        self.logger.info("Memory Manager stopped")
        return True
    
    def store_interaction(self, input_data, context, decisions, results):
        """Store an interaction in memory"""
        if not self.running:
            self.logger.warning("Cannot store interaction - memory manager not running")
            return False
            
        # Create memory record
        memory_record = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "input": input_data,
            "context": context,
            "decisions": decisions,
            "results": results
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory_record)
        
        # If short-term memory is full, flush oldest items to disk
        if len(self.short_term_memory) > self.max_short_term_items:
            self._flush_short_term_memory(items_to_flush=len(self.short_term_memory) // 2)
            
        return True
    
    def _flush_short_term_memory(self, items_to_flush=None):
        """Flush short-term memory items to disk"""
        if items_to_flush is None:
            items_to_flush = len(self.short_term_memory)
            
        if not self.short_term_memory:
            return
            
        items_to_save = self.short_term_memory[:items_to_flush]
        self.short_term_memory = self.short_term_memory[items_to_flush:]
        
        # Save items to disk
        timestamp = int(time.time())
        filename = f"{self.memory_path}/interactions/interaction_batch_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(items_to_save, f, indent=2)
            self.logger.info(f"Flushed {len(items_to_save)} items to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to flush memory: {e}")
            # Put items back in short-term memory
            self.short_term_memory = items_to_save + self.short_term_memory
    
    def retrieve_by_keyword(self, keyword, limit=10):
        """Retrieve memories by keyword"""
        if not self.running:
            self.logger.warning("Cannot retrieve memories - memory manager not running")
            return []
            
        results = []
        
        # First check short-term memory
        for item in self.short_term_memory:
            if keyword.lower() in str(item).lower():
                results.append(item)
                if len(results) >= limit:
                    return results
        
        # Then check on-disk memory
        try:
            interaction_files = os.listdir(f"{self.memory_path}/interactions")
            interaction_files.sort(reverse=True)  # Most recent first
            
            for filename in interaction_files:
                if not filename.endswith('.json'):
                    continue
                    
                filepath = f"{self.memory_path}/interactions/{filename}"
                try:
                    with open(filepath, 'r') as f:
                        batch = json.load(f)
                        
                    for item in batch:
                        if keyword.lower() in str(item).lower():
                            results.append(item)
                            if len(results) >= limit:
                                return results
                except Exception as e:
                    self.logger.error(f"Error reading memory file {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"Error accessing memory directory: {e}")
        
        return results
    
    def store_knowledge(self, category, key, data):
        """Store knowledge in long-term memory"""
        if not self.running:
            self.logger.warning("Cannot store knowledge - memory manager not running")
            return False
            
        filepath = f"{self.memory_path}/knowledge/{category}.json"
        
        try:
            # Load existing knowledge
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    knowledge = json.load(f)
            else:
                knowledge = {}
            
            # Add or update knowledge
            knowledge[key] = {
                "data": data,
                "updated": time.time()
            }
            
            # Save knowledge
            with open(filepath, 'w') as f:
                json.dump(knowledge, f, indent=2)
                
            self.logger.info(f"Stored knowledge: {category}/{key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store knowledge: {e}")
            return False
    
    def retrieve_knowledge(self, category, key=None):
        """Retrieve knowledge from long-term memory"""
        if not self.running:
            self.logger.warning("Cannot retrieve knowledge - memory manager not running")
            return None
            
        filepath = f"{self.memory_path}/knowledge/{category}.json"
        
        try:
            if not os.path.exists(filepath):
                return None
                
            with open(filepath, 'r') as f:
                knowledge = json.load(f)
                
            if key is not None:
                return knowledge.get(key, {}).get("data")
            else:
                # Return all knowledge in the category
                return {k: v["data"] for k, v in knowledge.items()}
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge: {e}")
            return None
    
    def get_status(self):
        """Return the current status"""
        return {
            "running": self.running,
            "short_term_items": len(self.short_term_memory)
        }