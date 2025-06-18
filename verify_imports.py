import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Try to import the custom_nlp module
    from llm.processors.custom_nlp import DutchParser, DutchSentimentAnalyzer, DutchTokenizer
    
    # Test the imports
    print("Successfully imported custom_nlp module!")
    print(f"DutchParser: {DutchParser}")
    print(f"DutchSentimentAnalyzer: {DutchSentimentAnalyzer}")
    print(f"DutchTokenizer: {DutchTokenizer}")
    
    # Test the processor
    print("\nTesting DutchParser:")
    parser = DutchParser()
    result = parser.parse("Dit is een test.")
    print(f"Parse result: {result}")
    
    # Test the sentiment analyzer
    print("\nTesting DutchSentimentAnalyzer:")
    analyzer = DutchSentimentAnalyzer()
    sentiment = analyzer.analyze("Dit is geweldig!")
    print(f"Sentiment analysis: {sentiment}")
    
except ImportError as e:
    print(f"Error importing module: {e}")
    print("\nCurrent Python path:")
    for path in sys.path:
        print(f"  {path}")
    
    print("\nCurrent directory contents:")
    for item in os.listdir(project_root):
        print(f"  {item}")
