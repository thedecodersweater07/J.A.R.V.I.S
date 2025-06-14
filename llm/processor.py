from typing import Dict, Any, Optional, List
import logging
from .base import NLPBase
from .processors.dutch_ner import DutchNER

logger = logging.getLogger(__name__)

class NLPProcessor(NLPBase):
    """Natural Language Processing component without external dependencies"""
    
    def __init__(self, model_name: str = None):
        super().__init__()
        self.logger = logger
        self.ner = DutchNER()

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text with NLP pipeline"""
        if not text.strip():
            return {"error": "Empty text provided"}
            
        try:
            # Basic NLP operations using in-house implementations
            tokens = self.tokenize(text)
            entities = self.ner(tokens)
            
            # Extract basic information
            result = {
                "text": text,
                "tokens": tokens,
                "entities": entities,
                "noun_chunks": self._extract_noun_chunks(tokens)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {"error": str(e)}
            
    def _extract_noun_chunks(self, tokens: List[str]) -> List[str]:
        """Simple noun phrase chunking"""
        chunks = []
        current_chunk = []
        
        # Very basic chunking - can be enhanced with more sophisticated rules
        for token in tokens:
            if token[0].isupper() or token.lower() in {'de', 'het', 'een'}:
                current_chunk.append(token)
            elif current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis"""
        pos_words = {"goed", "mooi", "geweldig", "fantastisch", "prima"}
        neg_words = {"slecht", "vervelend", "vreselijk", "waardeloos"}
        
        tokens = self.tokenize(text.lower())
        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)
        
        if pos_count > neg_count:
            sentiment = "positive"
            score = min(1.0, pos_count / len(tokens))
        elif neg_count > pos_count:
            sentiment = "negative" 
            score = -min(1.0, neg_count / len(tokens))
        else:
            sentiment = "neutral"
            score = 0.0
            
        return {
            "sentiment": sentiment,
            "score": score,
        }
