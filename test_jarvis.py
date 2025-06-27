"""
Test script for JARVIS response generation.
"""
import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_response_generation():
    """Test response generation with different inputs."""
    from models.jarvis import JarvisModel
    
    logger.info("Initializing JarvisModel...")
    jarvis = JarvisModel()
    
    test_cases = [
        "hallo",
        "wat is de hoofdstad van nederland?",
        "hoe gaat het met je?",
        "kun je me helpen met een vraag?",
        "bedankt voor je hulp"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test_input}")
        try:
            response = jarvis.process_input(test_input)
            if isinstance(response, dict) and 'response' in response:
                logger.info(f"Response: {response['response']}")
                logger.info(f"Success: {response.get('success', False)}")
                logger.info(f"Processing time: {response.get('processing_time', 'N/A')}s")
            else:
                logger.warning(f"Unexpected response format: {response}")
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
    
    logger.info("\nTesting complete!")

if __name__ == "__main__":
    test_response_generation()
