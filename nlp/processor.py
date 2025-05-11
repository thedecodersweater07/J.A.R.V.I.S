import logging
from typing import Dict, Any, Optional, List, Union
import spacy
import numpy as np

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Natural Language Processing component for JARVIS
    Handles text analysis, sentiment detection, and other NLP tasks
    """
    
    def __init__(self, model_name: str = "nl_core_news_sm"):
        self.model_name = model_name
        self.nlp = None
        self._initialize_model()
        logger.info(f"NLP Processor initialized with model: {model_name}")
        
    def _initialize_model(self):
        """Initialize the spaCy NLP model"""
        try:
            # Try to load the model
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            # If model not found, try to download it
            try:
                logger.info(f"Model {self.model_name} not found, downloading...")
                spacy.cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Downloaded and loaded model: {self.model_name}")
            except Exception as e:
                # If download fails, use a fallback approach
                logger.error(f"Failed to download model {self.model_name}: {e}")
                logger.info("Using blank model as fallback")
                self.nlp = spacy.blank("nl")
                
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text with NLP pipeline
        Returns dictionary with analysis results
        """
        if not text.strip():
            return {"error": "Empty text provided"}
            
        if not self.nlp:
            return {"error": "NLP model not initialized"}
            
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract basic information
            result = {
                "text": text,
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ],
                "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
                "sentiment": self._analyze_sentiment(doc)
            }
            
            # Add dependency parsing if requested in context
            if context and context.get("include_dependencies", False):
                result["dependencies"] = [
                    {
                        "text": token.text,
                        "dep": token.dep_,
                        "head": token.head.text
                    }
                    for token in doc
                ]
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {"error": str(e)}
            
    def _analyze_sentiment(self, doc) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        Returns dictionary with sentiment scores
        """
        # Simple rule-based sentiment analysis as fallback
        # This is very basic and should be replaced with a proper model
        positive_words = {"goed", "geweldig", "fantastisch", "mooi", "leuk", "prima"}
        negative_words = {"slecht", "verschrikkelijk", "vreselijk", "lelijk", "stom", "boos"}
        
        tokens = [token.lemma_.lower() for token in doc]
        
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # Calculate simple sentiment score
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, positive_count / len(doc))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = -min(1.0, negative_count / len(doc))
        else:
            sentiment = "neutral"
            score = 0.0
            
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_terms": positive_count,
            "negative_terms": negative_count
        }
        
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key terms from text"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            
            # Extract nouns and proper nouns as keywords
            keywords = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}]
            
            # Count frequencies
            from collections import Counter
            keyword_freq = Counter(keywords)
            
            # Return top N keywords
            return [kw for kw, _ in keyword_freq.most_common(top_n)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
            
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        Simple text classification using word overlap
        Returns scores for each category
        """
        if not text or not categories:
            return {}
            
        try:
            # Process text
            doc = self.nlp(text.lower())
            text_tokens = set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
            
            # Calculate overlap with each category
            scores = {}
            for category in categories:
                # Process category name
                cat_doc = self.nlp(category.lower())
                cat_tokens = set(token.lemma_ for token in cat_doc if not token.is_stop and not token.is_punct)
                
                # Calculate Jaccard similarity
                if text_tokens and cat_tokens:
                    intersection = len(text_tokens.intersection(cat_tokens))
                    union = len(text_tokens.union(cat_tokens))
                    scores[category] = intersection / union if union > 0 else 0.0
                else:
                    scores[category] = 0.0
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return {category: 0.0 for category in categories}
