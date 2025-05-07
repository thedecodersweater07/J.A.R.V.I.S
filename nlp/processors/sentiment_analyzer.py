import torch
from typing import Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "wietsedv/bert-base-dutch-cased-finetuned-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def analyze(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
        return {
            "positive": float(probs[0][2]),
            "neutral": float(probs[0][1]),
            "negative": float(probs[0][0])
        }
