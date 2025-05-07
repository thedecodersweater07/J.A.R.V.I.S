from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class EnhancedMemoryManager:
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        self.priority_scores = {}
        
    def get_context(self, query: str) -> Dict:
        relevant_history = self._find_relevant_history(query)
        return {
            "relevant_history": relevant_history,
            "timestamp": datetime.now().isoformat()
        }
    
    def store_interaction(self, prompt: str, response: str, context: Dict):
        importance = self._calculate_importance(prompt, response)
        
        if importance >= self.config["priority_threshold"]:
            self._store_in_cache(prompt, response, importance, context)
            
        # Cleanup old cache entries if needed
        if len(self.cache) > self.config["cache_size"]:
            self._cleanup_cache()
    
    def _find_relevant_history(self, query: str) -> List[Dict]:
        # Use vector similarity to find relevant history
        relevant = sorted(
            self.cache.items(),
            key=lambda x: self._calculate_relevance(query, x[1]),
            reverse=True
        )
        
        return [
            {
                "text": item[1]["text"],
                "category": item[1]["category"],
                "timestamp": item[1]["timestamp"]
            }
            for item in relevant[:self.config["context_window"]]
        ]
    
    def _calculate_importance(self, prompt: str, response: str) -> float:
        # Implement importance calculation based on:
        # - Length of interaction
        # - Presence of key terms
        # - User feedback (if available)
        # Returns float between 0 and 1
        return 0.8  # Placeholder implementation
    
    def _calculate_relevance(self, query: str, entry: Dict) -> float:
        # Implement relevance calculation
        # Could use techniques like cosine similarity
        return 0.5  # Placeholder implementation
    
    def _store_in_cache(self, prompt: str, response: str, importance: float, context: Dict):
        timestamp = datetime.now().isoformat()
        key = f"{timestamp}_{hash(prompt)}"
        
        self.cache[key] = {
            "text": f"Q: {prompt}\nA: {response}",
            "category": self._categorize_interaction(prompt, response),
            "importance": importance,
            "timestamp": timestamp,
            "context": context
        }
        
    def _cleanup_cache(self):
        # Remove least important entries
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1]["importance"], x[1]["timestamp"])
        )
        
        # Keep only the most important entries
        self.cache = dict(sorted_entries[-self.config["cache_size"]:])
    
    def _categorize_interaction(self, prompt: str, response: str) -> str:
        # Implement interaction categorization
        return "general"  # Placeholder implementation
