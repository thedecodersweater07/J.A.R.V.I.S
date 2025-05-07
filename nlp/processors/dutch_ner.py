from typing import List, Dict, Any
import spacy

class DutchNER:
    def __init__(self, nlp):
        self.nlp = nlp
        
    def __call__(self, tokens: List[str]) -> List[Dict[str, Any]]:
        doc = self.nlp(" ".join(tokens))
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]
