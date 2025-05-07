"""
Recall engine that integrates different memory systems.
Handles memory retrieval, association, and context-aware recall.
"""
import time
import threading
from collections import defaultdict


class RecallEngine:
    def __init__(self, short_term_memory=None, long_term_memory=None, memory_indexer=None):
        """
        Initialize the recall engine with memory components.
        
        Args:
            short_term_memory: ShortTermMemory instance
            long_term_memory: LongTermMemory instance
            memory_indexer: MemoryIndexer instance
        """
        self.short_term = short_term_memory
        self.long_term = long_term_memory
        self.indexer = memory_indexer
        self.context = {}
        self.retrieval_stats = defaultdict(int)
        self.lock = threading.RLock()
    
    def remember(self, query, context=None):
        """Retrieve information from memory systems based on query and context."""
        if not query:
            return {'results': [], 'stats': {}}
            
        if context is None:
            context = {}
            
        with self.lock:
            self.context.update(context)
            start_time = time.time()
            results = []
            
            # Try direct key lookup in short-term memory first
            stm_result = None
            if self.short_term:
                stm_result = self.short_term.retrieve(query)
                if stm_result:
                    results.append({
                        'key': query,
                        'value': stm_result,
                        'source': 'short_term',
                        'access_time': time.time() - start_time,
                        'confidence': 1.0,
                        'direct_match': True
                    })
                    self.retrieval_stats['short_term_direct'] += 1
            
            # Try direct key lookup in long-term memory
            ltm_result = None
            if self.long_term and not stm_result:
                ltm_result = self.long_term.retrieve(query)
                if ltm_result:
                    results.append({
                        'key': query,
                        'value': ltm_result['value'],
                        'tags': ltm_result.get('tags', []),
                        'source': 'long_term',
                        'access_time': time.time() - start_time,
                        'confidence': 0.95,
                        'direct_match': True
                    })
                    self.retrieval_stats['long_term_direct'] += 1
            
            # If no direct match or we need more results, use the indexer
            if self.indexer and (not results or len(query.split()) > 1):
                index_results = self.indexer.search(query, limit=5)
                
                # Process and add indexed results
                for result in index_results:
                    # Skip results we've already found through direct lookup
                    if any(r['key'] == result['key'] for r in results):
                        continue
                        
                    # Calculate confidence based on relevance score
                    confidence = min(0.9, 0.5 + (1.0 / (1 + abs(result['relevance']))))
                    
                    results.append({
                        'key': result['key'],
                        'value': result['content'],
                        'metadata': result.get('metadata', {}),
                        'source': result['source'],
                        'access_time': time.time() - start_time,
                        'confidence': confidence,
                        'direct_match': False
                    })
                    self.retrieval_stats['index_search'] += 1
            
            # Use contextual recall for remaining gaps
            if context and (not results or len(results) < 3):
                context_results = self._contextual_recall(query, context)
                results.extend(context_results)
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Update stats
            self.retrieval_stats['total_recalls'] += 1
            
            return {
                'results': results,
                'query_time': time.time() - start_time,
                'stats': dict(self.retrieval_stats)
            }
    
    def _contextual_recall(self, query, context):
        """Use context to enhance memory recall."""
        results = []
        context_keys = []
        
        # Extract potential memory keys from context
        for key, value in context.items():
            if key == 'tags' and isinstance(value, list):
                # Search by tags if provided
                if self.long_term:
                    tag_results = self.long_term.search_by_tags(value, match_all=False)
                    for tag_key, tag_value in tag_results.items():
                        # Check if the query terms appear in the tag key or value
                        query_terms = set(query.lower().split())
                        key_matches = any(term in tag_key.lower() for term in query_terms)
                        
                        value_matches = False
                        if isinstance(tag_value, str):
                            value_matches = any(term in tag_value.lower() for term in query_terms)
                        
                        if key_matches or value_matches:
                            results.append({
                                'key': tag_key,
                                'value': tag_value,
                                'source': 'long_term',
                                'access_time': 0,
                                'confidence': 0.75 if key_matches else 0.6,
                                'direct_match': False,
                                'via_context': 'tags'
                            })
            
            elif key == 'related_keys' and isinstance(value, list):
                # Direct lookup of related keys
                context_keys.extend(value)
        
        # Look up any context keys in both memory systems
        for key in context_keys:
            if self.short_term:
                stm_result = self.short_term.retrieve(key)
                if stm_result:
                    results.append({
                        'key': key,
                        'value': stm_result,
                        'source': 'short_term',
                        'access_time': 0,
                        'confidence': 0.7,
                        'direct_match': False,
                        'via_context': 'related_key'
                    })
            
            if self.long_term:
                ltm_result = self.long_term.retrieve(key)
                if ltm_result:
                    results.append({
                        'key': key,
                        'value': ltm_result['value'],
                        'tags': ltm_result.get('tags', []),
                        'source': 'long_term',
                        'access_time': 0,
                        'confidence': 0.65,
                        'direct_match': False,
                        'via_context': 'related_key'
                    })
        
        return results
    
    def memorize(self, key, value, importance=1.0, tags=None, store_in_long_term=True):
        """Store information in both short-term and optionally long-term memory."""
        success = True
        
        # Always store in short-term memory if available
        if self.short_term:
            stm_success = self.short_term.store(key, value, importance=importance)
            success = success and stm_success
        
        # Optionally store in long-term memory
        if store_in_long_term and self.long_term:
            ltm_success = self.long_term.store(key, value, tags=tags, importance=importance)
            success = success and ltm_success
        
        # Index the memory for search
        if self.indexer:
            metadata = {'importance': importance, 'tags': tags or []}
            idx_success = self.indexer.index_memory(
                key, 
                value, 
                metadata=metadata,
                source="long_term" if store_in_long_term else "short_term"
            )
            success = success and idx_success
            
        return success
    
    def associate(self, key1, key2, association_type="related"):
        """Create an association between two memory items."""
        # For this implementation, we'll store associations as special entries in long-term memory
        if not self.long_term:
            return False
            
        assoc_key = f"association:{key1}:{key2}:{association_type}"
        assoc_value = {
            'key1': key1,
            'key2': key2,
            'type': association_type,
            'created_at': time.time()
        }
        
        return self.long_term.store(
            assoc_key, 
            assoc_value,
            tags=['association', f'assoc_type:{association_type}', 
                  f'assoc_key:{key1}', f'assoc_key:{key2}']
        )
    
    def find_associations(self, key):
        """Find all memories associated with a given key."""
        if not self.long_term:
            return []
            
        # Search for associations by tag
        assoc_tag = f'assoc_key:{key}'
        assoc_results = self.long_term.search_by_tags([assoc_tag])
        
        associations = []
        for assoc_key, assoc_value in assoc_results.items():
            if isinstance(assoc_value, dict) and 'key1' in assoc_value and 'key2' in assoc_value:
                # Get the other key in the association
                other_key = assoc_value['key2'] if assoc_value['key1'] == key else assoc_value['key1']
                
                # Get the associated memory
                memory = self.long_term.retrieve(other_key)
                if memory:
                    associations.append({
                        'key': other_key,
                        'value': memory['value'],
                        'association_type': assoc_value.get('type', 'related'),
                        'tags': memory.get('tags', [])
                    })
        
        return associations