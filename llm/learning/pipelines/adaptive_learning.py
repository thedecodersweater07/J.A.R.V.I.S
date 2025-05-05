import logging
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
from llm.knowledge import KnowledgeManager
from ml.training.trainers import ModelTrainer
from ml.feature_engineering import FeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive learning"""
    learning_rate: float = 0.001
    batch_size: int = 32
    adaptation_threshold: float = 0.75
    min_confidence: float = 0.6
    max_retry_attempts: int = 3
    feedback_weight: float = 0.5
    performance_window: int = 100
    difficulty_levels: List[str] = ("basic", "intermediate", "advanced")

class AdaptiveLearningPipeline:
    """Implements adaptive learning strategies that adjust based on performance"""
    
    def __init__(self, config: Dict[str, Any], knowledge_manager: KnowledgeManager):
        self.config = AdaptiveConfig(**config.get('adaptive_config', {}))
        self.knowledge_manager = knowledge_manager
        self.feature_extractor = FeatureExtractor()
        self.trainer = ModelTrainer(config.get('training_config', {}))
        self.performance_history = []
        self.current_difficulty = "basic"
        
    async def process_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch with adaptive learning strategies"""
        try:
            # Extract features with current difficulty
            features = await self._extract_features_adaptive(data)
            
            # Adjust learning parameters based on performance
            self._adjust_learning_params()
            
            # Train with adaptive parameters
            results = await self._train_adaptive(features)
            
            # Update performance history
            self._update_performance(results)
            
            # Adapt difficulty if needed
            self._adapt_difficulty()
            
            return results
            
        except Exception as e:
            logger.error(f"Adaptive learning pipeline error: {e}")
            raise

    async def _extract_features_adaptive(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features with complexity based on current difficulty"""
        feature_config = {
            "complexity": self.current_difficulty,
            "min_confidence": self.config.min_confidence
        }
        return await self.feature_extractor.process_batch(data, feature_config)

    def _adjust_learning_params(self):
        """Adjust learning parameters based on performance history"""
        if len(self.performance_history) >= self.config.performance_window:
            avg_performance = sum(self.performance_history[-self.config.performance_window:]) / self.config.performance_window
            
            if avg_performance > self.config.adaptation_threshold:
                # Increase learning complexity
                self.config.learning_rate *= 0.95
            elif avg_performance < self.config.adaptation_threshold * 0.8:
                # Decrease learning complexity
                self.config.learning_rate *= 1.05

    async def _train_adaptive(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model with adaptive parameters"""
        training_config = {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "difficulty": self.current_difficulty
        }
        return await self.trainer.train_batch(features, training_config)

    def _update_performance(self, results: Dict[str, Any]):
        """Update performance history"""
        performance = results.get("accuracy", 0.0)
        self.performance_history.append(performance)
        
        # Keep history within window
        if len(self.performance_history) > self.config.performance_window:
            self.performance_history.pop(0)

    def _adapt_difficulty(self):
        """Adapt difficulty level based on sustained performance"""
        if len(self.performance_history) >= self.config.performance_window:
            avg_performance = sum(self.performance_history[-self.config.performance_window:]) / self.config.performance_window
            
            current_level_idx = self.config.difficulty_levels.index(self.current_difficulty)
            
            if avg_performance > self.config.adaptation_threshold and current_level_idx < len(self.config.difficulty_levels) - 1:
                self.current_difficulty = self.config.difficulty_levels[current_level_idx + 1]
                logger.info(f"Difficulty increased to: {self.current_difficulty}")
            elif avg_performance < self.config.adaptation_threshold * 0.7 and current_level_idx > 0:
                self.current_difficulty = self.config.difficulty_levels[current_level_idx - 1]
                logger.info(f"Difficulty decreased to: {self.current_difficulty}")
