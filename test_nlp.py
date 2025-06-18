import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from llm.processors.custom_nlp import DutchProcessor, DutchSentimentAnalyzer

def test_dutch_processor():
    print("Testing Dutch Processor...")
    processor = DutchProcessor()
    text = "Dit is een eenvoudige Nederlandse zin."
    
    # Test tokenization
    tokens = processor.tokenize(text)
    print(f"Tokens: {tokens}")
    
    # Test sentence tokenization
    sentences = processor.sentence_tokenize("Dit is de eerste zin. En dit is de tweede.")
    print(f"Sentences: {sentences}")
    
    # Test POS tagging
    tagged = processor.pos_tag(tokens)
    print(f"POS tags: {tagged}")
    
    # Test noun phrase extraction
    chunks = processor.extract_noun_phrases(text)
    print(f"Noun chunks: {chunks}")

def test_sentiment_analyzer():
    print("\nTesting Sentiment Analyzer...")
    analyzer = DutchSentimentAnalyzer()
    
    # Test positive sentiment
    pos_text = "Dit is echt geweldig en mooi!"
    pos_result = analyzer.analyze(pos_text)
    print(f"Positive text: {pos_text}")
    print(f"Sentiment: {pos_result}")
    
    # Test negative sentiment
    neg_text = "Dit is heel slecht en vervelend"
    neg_result = analyzer.analyze(neg_text)
    print(f"\nNegative text: {neg_text}")
    print(f"Sentiment: {neg_result}")
    
    # Test neutral sentiment
    neu_text = "Dit is een gewone zin"
    neu_result = analyzer.analyze(neu_text)
    print(f"\nNeutral text: {neu_text}")
    print(f"Sentiment: {neu_result}")

if __name__ == "__main__":
    test_dutch_processor()
    test_sentiment_analyzer()
