import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from db.sql.database_manager import DatabaseManager  # Use DatabaseManager only
from .pipelines.structured_learning import StructuredLearningPipeline
from .pipelines.adaptive_learning import AdaptiveLearningPipeline
from llm.knowledge import KnowledgeManager
from llm.virtual_environment import VirtualEnvironmentManager

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

logger = logging.getLogger(__name__)

class LearningManager:
    """Manages continuous learning from various sources with improved error handling"""
    
    def __init__(self, config: Dict[str, Any], knowledge_manager: Optional[KnowledgeManager] = None):
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.model = None
        self.tokenizer = None
        self.db = DatabaseManager()
        self.knowledge_manager = knowledge_manager or KnowledgeManager(self.db)
        self.virtual_env_manager = VirtualEnvironmentManager(config.get('virtual_environment', {}))
        self.experiment_tasks = []
        
        # Initialize pipelines with config and knowledge manager
        self.structured_pipeline = StructuredLearningPipeline(
            config=config.get('structured_learning', {}),
            knowledge_manager=self.knowledge_manager
        )
        self.adaptive_pipeline = AdaptiveLearningPipeline(
            config=config.get('adaptive_learning', {}),
            knowledge_manager=self.knowledge_manager
        )

    async def initialize(self):
        """Initialize the learning system"""
        self.model = await self._load_model()
        self.tokenizer = await self._load_tokenizer()
        await self._setup_optimizer()
        
    async def start_continuous_learning(self):
        """Start continuous learning processes"""
        self.session = aiohttp.ClientSession()
        
        # Start different learning tasks
        self.learning_tasks = [
            asyncio.create_task(self._learn_from_internet()),
            asyncio.create_task(self._process_interaction_logs()),
            asyncio.create_task(self._analyze_system_feedback()),
            asyncio.create_task(self._update_knowledge_base()),
            asyncio.create_task(self._perform_self_improvement())
        ]
        
    async def learn_from_interaction(self, input_text: str, response: str):
        """Learn from user interactions"""
        try:
            if MONGO_AVAILABLE:
                await self._store_mongo_interaction(input_text, response)
            else:
                await self._store_sqlite_interaction(input_text, response)
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            
    async def _store_mongo_interaction(self, input_text: str, response: str):
        """Store interaction in MongoDB"""
        await self.db.get_client()['interactions'].insert_one({
            'input': input_text,
            'response': response,
            'timestamp': datetime.utcnow(),
            'analyzed': False
        })
        
    async def _store_sqlite_interaction(self, input_text: str, response: str):
        """Store interaction in SQLite"""
        conn = self.db.get_client()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (input, response, timestamp, analyzed)
            VALUES (?, ?, ?, ?)
        ''', (input_text, response, datetime.utcnow(), False))
        conn.commit()
            
    async def _process_interaction(self, interaction_id):
        """Process a single interaction for learning"""
        try:
            interaction = await self.db.get_client()['interactions'].find_one({'_id': interaction_id})
            if not interaction:
                return
                
            # Extract learning examples
            training_data = self._prepare_training_data(interaction)
            
            # Perform learning step
            await self._learning_step(training_data)
            
            # Update interaction status
            await self.db.get_client()['interactions'].update_one(
                {'_id': interaction_id},
                {'$set': {'learned': True, 'analyzed': True}}
            )
            
        except Exception as e:
            logger.error(f"Error processing interaction {interaction_id}: {e}")
            
    async def _learn_from_internet(self):
        """Continuously learn from internet sources"""
        while True:
            try:
                sources = self.config.get('learning_sources', {
                    'wikipedia': {'enabled': True, 'frequency': 3600},
                    'arxiv': {'enabled': True, 'frequency': 7200},
                    'tech_blogs': {'enabled': True, 'frequency': 3600},
                    'scientific_papers': {'enabled': True, 'frequency': 14400}
                })
                
                for source_name, settings in sources.items():
                    if settings.get('enabled', False):
                        await self._scrape_source(source_name)
                        await self._run_virtual_experiments(source_name)
                        await asyncio.sleep(settings.get('frequency', 3600))
                        
            except Exception as e:
                logger.error(f"Internet learning error: {e}")
                await asyncio.sleep(300)
                
    async def _scrape_source(self, source: str):
        """Scrape and process data from a specific source"""
        try:
            scraper = self._get_source_scraper(source)
            data = await scraper.fetch_data()
            
            if data:
                # Process and validate data
                processed_data = await self._process_source_data(data)
                
                # Store in knowledge base
                await self.db.get_client()['knowledge'].insert_many(processed_data)
                
                # Trigger learning if enough new data
                if len(processed_data) >= self.config.get('min_batch_size', 10):
                    await self._batch_learning_step(processed_data)
                    
        except Exception as e:
            logger.error(f"Error scraping source {source}: {e}")
            
    async def _run_virtual_experiments(self, topic: str) -> None:
        """Run experiments in virtual environment based on learned topics"""
        try:
            sim_id = await self.virtual_env_manager.create_simulation(
                "knowledge_validation",
                {"topic": topic}
            )
            
            experiment_config = {
                "topic": topic,
                "validation_steps": 10,
                "learning_rate": 0.001
            }
            
            results = await self.virtual_env_manager.run_experiment(
                sim_id, 
                experiment_config
            )
            
            await self._integrate_experiment_results(results)
            
        except Exception as e:
            logger.error(f"Virtual experiment error: {e}")

    async def _integrate_experiment_results(self, results: Dict[str, Any]) -> None:
        """Integrate experimental results into knowledge base"""
        if results.get("confidence_score", 0) > 0.8:
            await self.knowledge_manager.add_validated_knowledge(
                results["topic"],
                results["learned_data"]
            )
            
    async def _perform_self_improvement(self):
        """Enhanced continuous self-improvement loop with batch processing"""
        while True:
            try:
                batch = await self._get_learning_batch()
                if batch:
                    results = await self._process_batch(batch)
                    await self._store_results(results)
                await asyncio.sleep(self.config.get('learning_interval', 60))
            except Exception as e:
                logger.error(f"Self-improvement error: {str(e)}")
                await self._handle_error(e)

    async def _get_learning_batch(self) -> List[Dict[str, Any]]:
        """Fetch a batch of items for learning"""
        return await self.db.get_learning_queue(limit=self.batch_size)

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Tuple[bool, Any]]:
        """Process a batch of learning items with retry logic"""
        results = []
        for item in batch:
            for attempt in range(self.retry_attempts):
                try:
                    result = await self._process_single_item(item)
                    results.append((True, result))
                    break
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        results.append((False, str(e)))
                    await asyncio.sleep(1 * (attempt + 1))
        return results

    async def _load_model(self):
        """Load the learning model"""
        try:
            model_name = self.config.get('model_name', 'gpt2')
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # ...existing helper methods...
