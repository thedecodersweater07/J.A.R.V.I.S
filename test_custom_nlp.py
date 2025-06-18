import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import our custom NLP components
from llm.processors.custom_nlp import (
    DutchTokenizer,
    DutchParser,
    DutchSentimentAnalyzer
)

def test_tokenizer():
    print("\n=== Testing DutchTokenizer ===")
    tokenizer = DutchTokenizer()
    text = "Dit is een testzin voor de Nederlandse tokenizer."
    
    # Test basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    
    # Test tokenization with stemming
    stemmed = tokenizer.tokenize_and_stem(text)
    print(f"Stemmed tokens: {stemmed}")
    
    # Test stopword removal
    without_stopwords = tokenizer.remove_stopwords(tokens)
    print(f"Without stopwords: {without_stopwords}")

def test_parser():
    print("\n=== Testing DutchParser ===")
    parser = DutchParser()
    text = "De snelle bruine vos springt over de luie hond."
    
    # Test parsing
    result = parser.parse(text)
    print(f"Original text: {text}")
    print(f"Parse result:")
    print(f"- Tokens: {result['tokens']}")
    print(f"- Sentences: {result['sentences']}")
    print(f"- Dependencies: {result['dependencies']}")
    print(f"- Noun chunks: {result['noun_chunks']}")

def test_sentiment_analyzer():
    print("\n=== Testing DutchSentimentAnalyzer ===")
    analyzer = DutchSentimentAnalyzer()
    
    # Test positive sentiment
    positive_text = "Dit is geweldig! Ik ben er heel blij mee."
    pos_result = analyzer.analyze(positive_text)
    print(f"Positive text: {positive_text}")
    print(f"Sentiment: {pos_result}")
    
    # Test negative sentiment
    negative_text = "Dit is verschrikkelijk en ik ben erg teleurgesteld."
    neg_result = analyzer.analyze(negative_text)
    print(f"\nNegative text: {negative_text}")
    print(f"Sentiment: {neg_result}")
    
    # Test neutral sentiment
    neutral_text = "Dit is een gewone zin zonder sterke gevoelens."
    neu_result = analyzer.analyze(neutral_text)
    print(f"\nNeutral text: {neutral_text}")
    print(f"Sentiment: {neu_result}")

if __name__ == "__main__":
    print("Testing Custom Dutch NLP Components\n" + "="*50)
    
    # Run tests
    test_tokenizer()
    test_parser()
    test_sentiment_analyzer()
    
    print("\nAll tests completed!")
