from typing import Dict, List, Any
from .base_agent import BaseAgent
from llm.knowledge import KnowledgeBaseConnector

class SearchAgent(BaseAgent):
    """Agent that performs searches and gathers information."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.knowledge = KnowledgeBaseConnector()
        
    def search(self, query: str, sources: List[str] = None) -> List[Dict]:
        """Search for information across specified sources."""
        results = []
        
        for source in (sources or self.config.get("default_sources", [])):
            source_results = self.query_source(source, query)
            results.extend(source_results)
            
        return self.rank_results(results)
        
    def update_knowledge_base(self, new_information: Dict[str, Any]):
        """Update the knowledge base with new information."""
        self.knowledge.update_knowledge(new_information)
