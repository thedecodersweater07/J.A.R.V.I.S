from transformers import pipeline
import re
from typing import Dict, Optional

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")
        self.emotion_patterns = {
            'positive': [r'\b(happy|great|awesome|excellent|good|love|wonderful)\b'],
            'negative': [r'\b(sad|bad|awful|terrible|hate|angry|upset)\b'],
            'neutral': [r'\b(okay|fine|normal|alright)\b']
        }

    def analyze(self, text: str) -> str:
        # Quick pattern check first
        for sentiment, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    return sentiment

        # Transformer-based analysis for complex cases
        try:
            result = self.analyzer(text)[0]
            return result["label"].lower()
        except:
            return "neutral"
