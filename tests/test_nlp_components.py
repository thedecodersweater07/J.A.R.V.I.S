"""
Test script for NLP components.

This script tests the functionality of the custom NLP components
including tokenization, parsing, and sentiment analysis.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tokenizer():
    """Test the Dutch tokenizer."""
    from llm.processors.dutch_tokenizer import DutchTokenizer
    
    logger.info("Testing DutchTokenizer...")
    tokenizer = DutchTokenizer()
    
    # Test cases
    test_cases = [
        "Hallo, hoe gaat het met jou?",
        "Ik heb een hond en een kat.",
        "Amsterdam is de hoofdstad van Nederland."
    ]
    
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        logger.info(f"Text: {text}")
        logger.info(f"Tokens: {tokens}")
        logger.info("-" * 50)
    
    logger.info("DutchTokenizer tests completed.\n")

def test_parser():
    """Test the Dutch parser."""
    from llm.processors.dutch_parser import DutchParser
    
    logger.info("Testing DutchParser...")
    parser = DutchParser()
    
    # Test cases
    test_cases = [
        "Ik hou van Nederlandse kaas.",
        "De kat zit op de mat.",
        "Amsterdam is de hoofdstad van Nederland."
    ]
    
    for text in test_cases:
        logger.info(f"Parsing: {text}")
        result = parser.parse(text)
        
        logger.info(f"Tokens: {result.get('tokens', [])}")
        logger.info(f"Sentences: {result.get('sentences', [])}")
        logger.info(f"Noun chunks: {result.get('noun_chunks', [])}")
        logger.info("-" * 50)
    
    logger.info("DutchParser tests completed.\n")

def test_sentiment_analyzer():
    """Test the sentiment analyzer."""
    from llm.processors.sentiment_analyzer import SentimentAnalyzer
    
    logger.info("Testing SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()
    
    # Test cases with expected sentiment
    test_cases = [
        ("Ik hou van dit product!", "positive"),
        ("Dit is verschrikkelijk slecht.", "negative"),
        ("Het is een gewone dag vandaag.", "neutral")
    ]
    
    for text, expected in test_cases:
        logger.info(f"Analyzing: {text}")
        result = analyzer.analyze(text)
        
        # Get the dominant sentiment
        sentiment = max(result.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Scores: {result}")
        logger.info(f"Detected sentiment: {sentiment} (expected: {expected})")
        logger.info("-" * 50)
    
    logger.info("SentimentAnalyzer tests completed.\n")

def test_custom_nlp():
    """Test the custom NLP components if available."""
    try:
        from llm.processors.custom_nlp import (
            DutchTokenizer as CustomTokenizer,
            DutchParser as CustomParser,
            DutchSentimentAnalyzer as CustomSentimentAnalyzer
        )
        
        logger.info("Testing Custom NLP components...")
        
        # Test custom tokenizer
        logger.info("Testing Custom DutchTokenizer...")
        tokenizer = CustomTokenizer()
        tokens = tokenizer.tokenize("Dit is een test.")
        logger.info(f"Custom Tokenizer output: {tokens}")
        
        # Test custom parser
        logger.info("\nTesting Custom DutchParser...")
        parser = CustomParser()
        parse_result = parser.parse("De kat zit op de mat.")
        logger.info(f"Custom Parser output: {parse_result}")
        
        # Test custom sentiment analyzer
        logger.info("\nTesting Custom Sentiment Analyzer...")
        analyzer = CustomSentimentAnalyzer()
        sentiment = analyzer.analyze("Dit is geweldig!")
        logger.info(f"Custom Sentiment Analyzer output: {sentiment}")
        
        logger.info("Custom NLP tests completed.\n")
        
    except ImportError as e:
        logger.warning(f"Custom NLP components not available: {e}")

def main():
    """Run all tests."""
    logger.info("Starting NLP component tests...\n")
    
    # Run tests
    test_tokenizer()
    test_parser()
    test_sentiment_analyzer()
    test_custom_nlp()
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main()
