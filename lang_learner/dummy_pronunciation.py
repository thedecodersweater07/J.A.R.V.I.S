"""Dummy implementation of PronunciationAnalyzer for when the real one is not available."""

from typing import Any, Dict, List, Optional

class DummyPronunciationAnalyzer:
    def __init__(self, language: str = "english"):
        self.language = language
    
    def set_language(self, language: str) -> None:
        self.language = language
    
    def analyze_pronunciation(self, audio_data: bytes) -> Dict[str, Any]:
        return {
            "text": "Sample text",
            "score": 1.0,
            "feedback": "Pronunciation analysis not available.",
            "phonemes": []
        }
    
    def analyze_pronunciation_from_file(self, file_path: str) -> Dict[str, Any]:
        return self.analyze_pronunciation(b"")
    
    def get_phonemes(self) -> List[Dict[str, Any]]:
        return []
