from typing import List, Dict, Any
from ..base import NLPBase
import re

class DutchNER(NLPBase):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.nlp = config.get('nlp') if config else None
        # Dutch-specific entity patterns
        self.patterns = {
            'PERSON': r'(?:Dhr\.|Mevr\.|Dr\.|Prof\.) [A-Z][a-z]+ [A-Z][a-z]+',
            'LOCATION': r'(?:Amsterdam|Rotterdam|Den Haag|Utrecht|Eindhoven|Groningen)',
            'ORGANIZATION': r'(?:BV|NV|Stichting|Vereniging) [A-Z][A-Za-z ]+'
        }
        
    def __call__(self, tokens: List[str]) -> List[Dict[str, Any]]:
        text = " ".join(tokens)
        entities = []
        
        # Check text against Dutch entity patterns
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })
                
        # Also get basic entities from parent class
        entities.extend(self.detect_entities(text))
        return entities
