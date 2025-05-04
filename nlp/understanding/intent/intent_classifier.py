import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re

class IntentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 10)  # 10 intent classes
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.intent_patterns = {
            "greeting": r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
            "question": r"\b(what|where|when|why|how|who)\b.*\?",
            "command": r"\b(do|show|tell|find|search|open|close)\b",
            "farewell": r"\b(goodbye|bye|see you|later)\b"
        }

    def classify(self, text: str) -> str:
        # Pattern matching first
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text.lower()):
                return intent
                
        # BERT classification for complex cases
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.bert(**inputs)
            logits = self.classifier(outputs.pooler_output)
            return self._get_intent_label(torch.argmax(logits).item())
        except:
            return "general"

    def _get_intent_label(self, index: int) -> str:
        labels = [
            "greeting", "question", "command", "farewell",
            "request", "confirmation", "denial", "clarification",
            "smalltalk", "general"
        ]
        return labels[index]
