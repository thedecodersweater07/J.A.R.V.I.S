# context_analyzer.py
# Een module voor het analyseren van contextgegevens in een applicatiesysteem

import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configuratie voor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ContextAnalyzer")

class ContextAnalyzer:
    """
    Klasse voor het analyseren van context in binnenkomende berichten of gegevens.
    Bevat methoden voor patroonherkenning, prioriteitsbepaling en contextverwerking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialiseert de ContextAnalyzer met optionele configuratie.
        
        Args:
            config: Een dictionary met configuratie-instellingen
        """
        self.config = config or {}
        self.patterns = self.config.get('patterns', {})
        self.context_history = []
        self.max_history_size = self.config.get('max_history_size', 100)
        logger.info("ContextAnalyzer geïnitialiseerd met %d patronen", len(self.patterns))
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyseert de gegeven content en extraheert relevante context.
        
        Args:
            content: De te analyseren tekst of data
            
        Returns:
            Een dictionary met de geanalyseerde context
        """
        timestamp = datetime.now()
        context_data = {
            'timestamp': timestamp,
            'length': len(content),
            'keywords': self._extract_keywords(content),
            'sentiment': self._analyze_sentiment(content),
            'entities': self._extract_entities(content),
            'priority_score': self._calculate_priority(content)
        }
        
        # Voeg deze context toe aan de geschiedenis
        self._update_history(context_data)
        
        return context_data
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extraheert keywords uit de content."""
        # Eenvoudige keyword extractie
        words = re.findall(r'\b\w+\b', content.lower())
        # Filter algemene woorden
        stop_words = set(['de', 'het', 'een', 'en', 'of', 'is', 'dat', 'in', 'op', 'voor'])
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))[:10]  # Top 10 unieke keywords
    
    def _analyze_sentiment(self, content: str) -> float:
        """Eenvoudige sentimentanalyse op de content."""
        positive_words = ['goed', 'geweldig', 'uitstekend', 'prima', 'top']
        negative_words = ['slecht', 'probleem', 'fout', 'moeilijk', 'niet']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extraheert entiteiten zoals data, e-mails, etc. uit de content."""
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content),
            'dates': re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', content),
            'urls': re.findall(r'https?://\S+|www\.\S+', content)
        }
        return entities
    
    def _calculate_priority(self, content: str) -> int:
        """Berekent een prioriteitsscore voor de content."""
        # Eenvoudig prioriteitssysteem
        score = 0
        urgent_terms = ['direct', 'urgent', 'meteen', 'kritiek', 'belangrijk']
        for term in urgent_terms:
            if term in content.lower():
                score += 2
                
        return min(score, 10)  # Maximaal 10