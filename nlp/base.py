from typing import List, Dict, Any
import re

class NLPBase:
    """Base class for custom NLP functionality"""
    
    def __init__(self):
        self.initialized = False
        self.vocab = set()
        
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        return text.split()
        
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """Basic entity detection using regex patterns"""
        entities = []
        # Basic patterns for dates, numbers, etc
        patterns = {
            'DATE': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'NUMBER': r'\b\d+\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end()
                })
                
        return entities
