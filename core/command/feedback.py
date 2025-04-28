# feedback.py
# Een module voor het verzamelen en verwerken van feedback in een systeem

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import os

# Configuratie voor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FeedbackManager")

class FeedbackEntry:
    """Klasse die een enkele feedback vertegenwoordigt."""
    
    def __init__(self, 
                 feedback_id: str, 
                 source: str, 
                 rating: int, 
                 message: str = None, 
                 categories: List[str] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialiseert een nieuw feedback item.
        
        Args:
            feedback_id: Unieke identifier voor de feedback
            source: Bron van de feedback (bijv. 'gebruiker', 'systeem')
            rating: Numerieke waardering (bijv. 1-5)
            message: Optionele tekstuele feedback
            categories: Tags of categorieën voor de feedback
            metadata: Extra contextuele gegevens
        """
        self.feedback_id = feedback_id
        self.source = source
        self.rating = rating
        self.message = message
        self.categories = categories or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converteert feedback naar dictionary formaat."""
        return {
            'feedback_id': self.feedback_id,
            'source': self.source,
            'rating': self.rating,
            'message': self.message,
            'categories': self.categories,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Maakt een FeedbackEntry object van een dictionary."""
        timestamp = data.pop('timestamp', None)
        entry = cls(**data)
        if timestamp:
            try:
                entry.timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                entry.timestamp = datetime.now()
        return entry

class FeedbackSystem:
    """Klasse voor het beheren, opslaan en analyseren van feedback."""
    
    def __init__(self, storage_path: str = None):
        """
        Initialiseert de FeedbackManager.
        
        Args:
            storage_path: Padnaam voor het opslaan van feedbackgegevens
        """
        self.storage_path = storage_path or './feedback_data'
        self.feedback_entries = []
        self._ensure_storage_exists()
        logger.info("FeedbackManager geïnitialiseerd met opslag in %s", self.storage_path)
    
    def _ensure_storage_exists(self):
        """Zorgt ervoor dat de opslagdirectory bestaat."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            logger.info("Opslagdirectory aangemaakt: %s", self.storage_path)
    
    def add_feedback(self, 
                    feedback_id: str, 
                    source: str, 
                    rating: int, 
                    message: str = None, 
                    categories: List[str] = None,
                    metadata: Dict[str, Any] = None) -> FeedbackEntry:
        """
        Voegt nieuwe feedback toe aan het systeem.
        
        Args:
            feedback_id: Unieke identifier voor de feedback
            source: Bron van de feedback
            rating: Numerieke waardering
            message: Optionele tekstuele feedback
            categories: Tags of categorieën voor de feedback
            metadata: Extra contextuele gegevens
            
        Returns:
            De nieuwe FeedbackEntry
        """
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            source=source,
            rating=rating,
            message=message,
            categories=categories,
            metadata=metadata
        )
        self.feedback_entries.append(entry)
        self._save_entry(entry)
        logger.info("Feedback %s toegevoegd van bron %s met waardering %d", 
                    feedback_id, source, rating)
        return entry
    
    def _save_entry(self, entry: FeedbackEntry):
        """Slaat een feedback entry op in de opslag."""
        filename = os.path.join(self.storage_path, f"{entry.feedback_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """Haalt een specifiek feedback item op op basis van ID."""
        for entry in self.feedback_entries:
            if entry.feedback_id == feedback_id:
                return entry
                
        # Controleer of het in de opslag staat
        filename = os.path.join(self.storage_path, f"{feedback_id}.json")
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = FeedbackEntry.from_dict(data)
                self.feedback_entries.append(entry)
                return entry
                
        return None