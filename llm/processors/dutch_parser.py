from typing import List, Dict, Any
import spacy

class DutchParser:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, tokens: List[str]) -> Dict[str, Any]:
        doc = self.nlp(" ".join(tokens))
        return {
            "dependencies": [(token.head.text, token.dep_, token.text) for token in doc],
            "sentences": [sent.text for sent in doc.sents],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
        }
