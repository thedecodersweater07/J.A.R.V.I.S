import logging
from typing import Dict, List, Optional
import re
from datetime import datetime

class LanguageProcessor:
    def __init__(self, language: str = "nl"):
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.patterns = self._get_language_patterns(language)
        self.context_memory = []
        self.max_context_length = 10

    def _get_language_patterns(self, language: str) -> Dict[str, str]:
        """Enhanced patterns for better language understanding"""
        patterns = {
            "nl": {
                'question': r'\b(wat|waar|wie|waarom|hoe|welke|wanneer|waarmee)\b.*\?',
                'command': r'\b(doe|maak|zet|start|stop|open|sluit|ga|kom|geef|toon|zoek)\b.*',
                'statement': r'.*\.$',
                'greeting': r'\b(hallo|hoi|hey|goedemorgen|goedemiddag|goedenavond)\b',
                'farewell': r'\b(doei|dag|tot ziens|tot later|bye)\b',
                'affirmative': r'\b(ja|jawel|zeker|klopt|correct|precies)\b',
                'negative': r'\b(nee|nope|niet|geen)\b'
            },
            "en": {
                'question': r'\b(what|where|who|why|how|which|when|whose)\b.*\?',
                'command': r'\b(do|make|set|start|stop|open|close|go|come|give|show|search)\b.*',
                'statement': r'.*\.$',
                'greeting': r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                'farewell': r'\b(goodbye|bye|see you|later)\b',
                'affirmative': r'\b(yes|yeah|correct|right|exactly)\b',
                'negative': r'\b(no|nope|not|none)\b'
            }
        }
        return patterns.get(language, patterns["nl"])

    def initialize(self):
        """Initialize language processing components"""
        try:
            self.initialized = True
            self.logger.info("Language processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize language processor: {e}")
            raise

    def process(self, text: str) -> str:
        """Main interface method for processing text"""
        if not self.initialized:
            self.initialize()
            
        try:
            # Normalize input
            text = text.strip()
            text = self.process_text(text)
            return text
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return text

    def process_text(self, text: str) -> str:
        """Internal processing method"""
        self.logger.debug(f"Processing text: {text}")
        
        # Basic text normalization
        text = text.lower().strip()
        text = self._remove_special_characters(text)
        text = self._normalize_spacing(text)
        
        return text

    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters but keep basic punctuation"""
        return re.sub(r'[^\w\s.!?,]', '', text)

    def _normalize_spacing(self, text: str) -> str:
        """Normalize spacing in text"""
        return ' '.join(text.split())

    def tokenize(self, text: str) -> List[str]:
        self.logger.debug("Tokenizing text")
        return text.split()

    def count_words(self, text: str) -> Dict[str, int]:
        self.logger.debug("Counting words in text")
        tokens = self.tokenize(text)
        word_count = {}
        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1
        return word_count

    def detect_language(self, text: str) -> Optional[str]:
        self.logger.debug("Detecting language of text")
        # Placeholder for language detection logic
        return None

    def get_context(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        return self.context_memory[-limit:] if self.context_memory else []

    def analyze_pattern(self, text: str) -> Dict[str, bool]:
        """Analyze text against all patterns"""
        results = {}
        for pattern_name, pattern in self.patterns[self.language].items():
            results[pattern_name] = bool(re.search(pattern, text.lower()))
        return results