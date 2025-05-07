from typing import List, Dict, Any
from pathlib import Path
import torch
from security.secure_data_handler import SecureDataHandler

class LLMDataManager:
    def __init__(self):
        self.secure_handler = SecureDataHandler()
        
    def load_training_corpus(self, texts_path: Path, exclude_patterns: List[str]) -> List[str]:
        """Load and sanitize training texts"""
        texts = []
        for file in texts_path.glob("*.txt"):
            if self._is_safe_file(file):
                text = self._load_and_sanitize(file, exclude_patterns)
                texts.extend(text)
        return texts
        
    def _is_safe_file(self, filepath: Path) -> bool:
        """Check if file is safe to load"""
        return "sensitive" not in filepath.parts
        
    def _load_and_sanitize(self, filepath: Path, exclude_patterns: List[str]) -> List[str]:
        """Load and sanitize text data"""
        with open(filepath) as f:
            lines = f.readlines()
            
        # Remove lines containing sensitive patterns
        return [
            line.strip() for line in lines
            if not any(pattern in line.lower() for pattern in exclude_patterns)
        ]
