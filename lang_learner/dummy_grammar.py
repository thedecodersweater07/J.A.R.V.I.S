"""Dummy implementation of GrammarChecker for when the real one is not available."""

from typing import Any, Dict, List, Optional

class DummyGrammarChecker:
    def __init__(self, language: str = "english", level: str = "beginner"):
        self.language = language
        self.level = level
    
    def set_language(self, language: str) -> None:
        self.language = language
    
    def set_level(self, level: str) -> None:
        self.level = level
    
    def check_grammar(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "issues": [],
            "score": 1.0,
            "feedback": "No grammar issues found."
        }
    
    def get_grammar_rules(self) -> List[Dict[str, Any]]:
        return []
