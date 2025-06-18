import re
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class DutchTokenizer:
    def __init__(self):
        self.stemmer = SnowballStemmer("dutch")
        self.stopwords = set(nltk_stopwords.words('dutch'))
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization for Dutch text"""
        return word_tokenize(text, language='dutch')
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem Dutch text"""
        tokens = self.tokenize(text)
        return [self.stemmer.stem(token) for token in tokens]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Dutch stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stopwords]

class DutchParser:
    def __init__(self):
        self.tokenizer = DutchTokenizer()
        # Basic Dutch dependency patterns (simplified)
        self.dependency_patterns = {
            'nsubj': [('is', 'wordt', 'lijkt', 'schijnt', 'blijkt')],  # Subject
            'dobj': [('heeft', 'hebt', 'had', 'zal hebben')],  # Direct object
            'iobj': [('geeft', 'gaf', 'zou geven')],  # Indirect object
            'amod': [('grote', 'kleine', 'mooie', 'lelijke')],  # Adjective
            'nmod': [('van', 'naar', 'voor', 'achter')],  # Nominal modifier
        }
    
    def extract_noun_chunks(self, tokens: List[str]) -> List[str]:
        """Extract simple noun phrases based on patterns"""
        chunks = []
        current_chunk = []
        
        for i, token in enumerate(tokens):
            # Simple pattern: determiner + adjective + noun
            if (i > 1 and tokens[i-2].lower() in ('de', 'het', 'een', 'mijn', 'jouw') 
                and tokens[i-1].endswith(('e', 'en'))):
                current_chunk = tokens[i-2:i+1]
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        return chunks
    
    def parse(self, text: str) -> Dict[str, any]:
        """Basic Dutch parsing without spaCy"""
        tokens = self.tokenizer.tokenize(text)
        sentences = sent_tokenize(text, language='dutch')
        
        # Simple dependency parsing (very basic)
        dependencies = []
        for i, token in enumerate(tokens):
            if i > 0:
                prev_token = tokens[i-1].lower()
                for dep_type, patterns in self.dependency_patterns.items():
                    for pattern in patterns:
                        if any(prev_token.endswith(p) for p in pattern):
                            dependencies.append((prev_token, dep_type, token))
        
        return {
            'tokens': tokens,
            'sentences': sentences,
            'dependencies': dependencies,
            'noun_chunks': self.extract_noun_chunks(tokens)
        }

class DutchSentimentAnalyzer:
    def __init__(self):
        # Simple sentiment lexicon (can be expanded)
        self.lexicon = {
            'positief': {'goed', 'mooi', 'leuk', 'fijn', 'blij', 'geweldig', 'perfect'},
            'negatief': {'slecht', 'lelijk', 'vervelend', 'moeilijk', 'verdrietig', 'teleurgesteld'}
        }
        self.tokenizer = DutchTokenizer()
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis for Dutch text"""
        tokens = self.tokenizer.tokenize_and_stem(text.lower())
        
        pos_count = sum(1 for token in tokens if token in self.lexicon['positief'])
        neg_count = sum(1 for token in tokens if token in self.lexicon['negatief'])
        total = len(tokens) or 1  # Avoid division by zero
        
        # Simple scoring
        score = (pos_count - neg_count) / total
        
        # Normalize to 0-1 range for consistency
        normalized_score = (score + 1) / 2
        
        return {
            'positive': max(0, normalized_score),
            'negative': max(0, 1 - normalized_score),
            'neutral': 1 - (abs(score))  # Neutral is inverse of confidence
        }

# Example usage
if __name__ == "__main__":
    # Test the parser
    parser = DutchParser()
    text = "Dit is een eenvoudige Nederlandse zin."
    print(f"Parsing: {text}")
    print(parser.parse(text))
    
    # Test sentiment analysis
    sentiment = DutchSentimentAnalyzer()
    print("\nSentiment analysis:")
    print(sentiment.analyze("Ik vind dit heel mooi en goed!"))  # Positive
    print(sentiment.analyze("Dit is niet zo goed"))  # Neutral
    print(sentiment.analyze("Dit is heel slecht en vervelend"))  # Negative
