"""
JARVIS AI Core Module
Provides centralized management and coordination for all AI components.
"""

from typing import Dict, Any, List, Optional
import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Export core AI components
try:
    from .coordinator import AICoordinator
    from .model_registry import ModelRegistry
    from .pipeline import PipelineManager
    from .events import EventBus
    from .resource_manager import ResourceManager
    
    # Import adapters if they exist
    try:
        from .adapters import NLPProcessorAdapter, ModelManagerAdapter
    except ImportError:
        logger.warning("AI adapters not available")
except ImportError as e:
    logger.warning(f"Error importing AI components: {e}")
