import logging
from typing import Optional, Dict, Any
from ..learning import LearningManager
from ..knowledge import KnowledgeManager
from ..inference import InferenceEngine
from db.database import Database

logger = logging.getLogger(__name__)

class LLMCore:
    """Core LLM system coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = Database.get_instance()
        
        # Initialize components
        self.knowledge_manager = KnowledgeManager(self.db)
        self.learning_manager = LearningManager(self.db)
        self.inference_engine = InferenceEngine(self.knowledge_manager)
        
        logger.info("LLM Core initialized")
        
    async def start(self):
        """Start LLM system"""
        await self.knowledge_manager.initialize()
        await self.learning_manager.start_continuous_learning()
        await self.inference_engine.warm_up()
        
    async def process(self, input_text: str) -> str:
        """Process input and generate response"""
        # Get context from knowledge base
        context = await self.knowledge_manager.get_relevant_context(input_text)
        
        # Generate response
        response = await self.inference_engine.generate(input_text, context)
        
        # Learn from interaction
        await self.learning_manager.learn_from_interaction(input_text, response)
        
        return response
