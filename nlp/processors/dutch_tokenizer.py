from typing import List
import re

class DutchTokenizer:
    def __init__(self):
        self.word_pattern = re.compile(r'\b\w+\b')
        self.special_chars = {
            "ij": "ĳ",  # Ligatuur voor 'ij'
            "IJ": "Ĳ"   # Hoofdletter ligatuur
        }
        
    def __call__(self, text: str) -> List[str]:
        # Normaliseer speciale Nederlandse karakters
        for old, new in self.special_chars.items():
            text = text.replace(old, new)
            
        # Tokenize met behoud van Nederlandse taalregels
        tokens = []
        for match in self.word_pattern.finditer(text):
            token = match.group()
            # Handle samenstellingen (compounds)
            if "-" in token:
                parts = token.split("-")
                tokens.extend(parts)
            else:
                tokens.append(token)
                
        return tokens
