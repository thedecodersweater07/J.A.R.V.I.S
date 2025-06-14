import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
from typing import Dict, Optional

class IntentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Dutch BERT model
        self.model_name = "GroNLP/bert-base-dutch-cased"
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, 12)  # 12 Dutch intent classes
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Dutch-specific intent patterns
        self.intent_patterns = {
            "begroeting": r"\b(hallo|hoi|hey|goedemorgen|goedemiddag|goedenavond)\b",
            "vraag": r"\b(wat|waar|wanneer|waarom|hoe|wie)\b.*\?",
            "opdracht": r"\b(doe|toon|vertel|zoek|open|sluit)\b",
            "afscheid": r"\b(doei|dag|tot ziens|later)\b",
            "bevestiging": r"\b(ja|jawel|zeker|graag|prima|goed)\b",
            "ontkenning": r"\b(nee|neen|niet|nooit)\b",
            "bedanking": r"\b(bedankt|dank|dankjewel|thanks)\b",
            "hulp": r"\b(help|hulp|assistent|ondersteuning)\b"
        }

    def classify(self, text: str) -> Dict[str, float]:
        # Pattern matching voor eenvoudige gevallen
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text.lower()):
                return {"intent": intent, "confidence": 1.0}
                
        # BERT classificatie voor complexere gevallen
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            outputs = self.bert(**inputs)
            logits = self.classifier(outputs.pooler_output)
            probs = torch.softmax(logits, dim=1)[0]
            
            intent_idx = torch.argmax(probs).item()
            return {
                "intent": self._get_intent_label(intent_idx),
                "confidence": float(probs[intent_idx])
            }
        except Exception as e:
            return {"intent": "algemeen", "confidence": 0.0}

    def _get_intent_label(self, index: int) -> str:
        labels = [
            "begroeting", "vraag", "opdracht", "afscheid",
            "bevestiging", "ontkenning", "bedanking", "hulp",
            "smalltalk", "informatie", "emotie", "algemeen"
        ]
        return labels[index] if 0 <= index < len(labels) else "algemeen"

