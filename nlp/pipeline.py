import torch
import spacy
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
from .processors import (
    DutchTokenizer,
    DutchParser,
    DutchNER,
    SentimentAnalyzer,
    IntentClassifier
)

class NLPPipeline:
    def __init__(self):
        # Load Dutch language models
        self.nlp = spacy.load("nl_core_news_lg")
        self.tokenizer = DutchTokenizer()
        self.parser = DutchParser(self.nlp)
        self.ner = DutchNER(self.nlp)
        
        # Load specialized models
        self.sentiment = SentimentAnalyzer()
        self.intent = IntentClassifier()
        
        # Transformer model voor Nederlandse taal
        self.model_name = "bert-base-dutch-cased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
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
        """Genereer contextuele embeddings"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state
