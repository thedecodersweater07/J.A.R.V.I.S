import torch
import spacy
from typing import List, Dict, Any, Optional
from .processors.dutch_tokenizer import DutchTokenizer
from .processors.dutch_parser import DutchParser  
from .processors.dutch_ner import DutchNER
from .processors.sentiment_analyzer import SentimentAnalyzer
from .processors.intent_classifier import IntentClassifier

class NLPPipeline:
    def __init__(self):
        # Load Dutch language models
        self.nlp = spacy.load("nl_core_news_lg")
        self.tokenizer = DutchTokenizer()
        self.parser = DutchParser(self.nlp)
        
        # Initialize NER with proper config
        ner_config = {"nlp": self.nlp}
        self.ner = DutchNER(config=ner_config)
        
        # Load specialized models
        self.sentiment = SentimentAnalyzer()
        self.intent = IntentClassifier()
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_parser(self):
        return self.parser
        
    def get_ner(self):
        return self.ner

    def process_text(self, text: str) -> Dict[str, Any]:
        """Voer volledige NLP analyse uit"""
        doc = self.nlp(text)
        
        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "pos_tags": [token.pos_ for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "sentiment": self.sentiment.analyze(text),
            "intent": self.intent.classify(text),
            "parse_tree": self._get_parse_tree(doc),
            "embeddings": self._get_embeddings(text)
        }
        
    def _get_parse_tree(self, doc) -> Dict[str, Any]:
        """Genereer syntactische parse tree"""
        return {
            "dependencies": [(token.head.text, token.dep_, token.text) 
                           for token in doc],
            "noun_chunks": list(doc.noun_chunks),
            "sentences": [sent.text for sent in doc.sents]
        }
        
    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Genereer contextuele embeddings met spaCy"""
        doc = self.nlp(text)
        return torch.tensor([token.vector for token in doc])
