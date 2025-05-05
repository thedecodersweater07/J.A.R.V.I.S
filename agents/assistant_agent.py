from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from core.memory import RecallEngine
from llm.knowledge import KnowledgeBaseConnector

class AssistantAgent(BaseAgent):
    """Agent that handles user interactions and provides assistance."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.recall = RecallEngine()
        self.knowledge = KnowledgeBaseConnector()
        
    def process_request(self, request: str) -> str:
        """Process a user request and generate appropriate response."""
        # Query knowledge base
        context = self.knowledge.query(request)
        
        # Get relevant memories
        memories = self.recall.search_related(request)
        
        # Generate response using LLM
        response = self.generate_response(request, context, memories)
        
        return response
        
    def learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn from user interaction to improve future responses."""
        self.knowledge.update_knowledge({
            "type": "interaction",
            "content": interaction,
            "timestamp": interaction.get("timestamp")
        })
