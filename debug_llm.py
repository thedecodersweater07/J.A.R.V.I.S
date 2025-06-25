import sys
import os
import logging
import torch
from pathlib import Path

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_imports():
    """Check if required imports are available"""
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        return False

def check_torch():
    """Check PyTorch installation and CUDA availability"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    return True

def test_llm_core():
    """Test LLM Core functionality"""
    try:
        # Add project root to path
        project_root = str(Path(__file__).parent.absolute())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Import LLMCore
        from llm.core.llm_core import LLMCore
        
        logger.info("LLMCore imported successfully")
        
        # Initialize with minimal config
        config = {
            "model": {
                "name": "distilgpt2",
                "auto_load": True,
                "max_length": 50,
                "device_map": "auto"
            },
            "memory_management": {
                "cache_size": 1000
            }
        }
        
        logger.info("Creating LLMCore instance...")
        llm = LLMCore(config)
        logger.info("LLMCore instance created successfully")
        
        # Test response generation
        test_prompt = "Hello, how are you?"
        logger.info(f"Testing generate_response with prompt: {test_prompt}")
        response = llm.generate_response(test_prompt)
        logger.info(f"Response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_llm_core: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("=== Starting LLM Core Debug ===\n")
    
    print("1. Checking imports...")
    if not check_imports():
        print("\nError: Required imports are missing.")
        sys.exit(1)
    
    print("\n2. Checking PyTorch...")
    check_torch()
    
    print("\n3. Testing LLM Core...")
    success = test_llm_core()
    
    print("\n=== Debug Completed ===")
    print(f"Test {'succeeded' if success else 'failed'}")
