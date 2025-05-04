import logging
from typing import Dict, List, Optional
import re

class LanguageProcessor:
    def __init__(self, language: str = "nl"):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.language = language
        
        # Language-specific patterns
        self.patterns = self._get_language_patterns(language)
        
    def _get_language_patterns(self, language: str) -> Dict[str, str]:
        """Get regex patterns for specific language"""
        patterns = {
            "nl": {
                'question': r'\b(wat|waar|wie|waarom|hoe|welke)\b.*\?',
                'command': r'\b(doe|maak|zet|start|stop|open|sluit)\b.*',
                'statement': r'.*\.$'
            },
            "en": {
                'question': r'\b(what|where|who|why|how|which)\b.*\?',
                'command': r'\b(do|make|set|start|stop|open|close)\b.*',
                'statement': r'.*\.$'
            }
        }
        return patterns.get(language, patterns["nl"])  # Default to Dutch if language not found

    def initialize(self):
        """Initialize language processing components"""
        try:
            # Add more sophisticated NLP initialization here
            self.initialized = True
            self.logger.info("Language processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize language processor: {e}")
            raise

    def process(self, text: str) -> str:
        """Process and normalize input text"""
        if not self.initialized:
            self.initialize()

        # Basic text normalization
        text = text.lower().strip()
        text = self._remove_duplicates(text)
        text = self._normalize_spacing(text)
        
        return text

    def _remove_duplicates(self, text: str) -> str:
        """Remove duplicate words in sequence"""
        words = text.split()
        return ' '.join(word for i, word in enumerate(words) 
                       if i == 0 or word != words[i-1])

    def _normalize_spacing(self, text: str) -> str:
        """Normalize spacing in text"""
        return ' '.join(text.split())
