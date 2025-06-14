"""Word tokenization implementation"""

class WordTokenizer:
    def __init__(self):
        self.vocab = set()
        
    def tokenize(self, text: str) -> list:
        """Convert text into word tokens"""
        # Simple word tokenization by splitting on spaces
        tokens = text.lower().split()
        self.vocab.update(tokens)
        return tokens
