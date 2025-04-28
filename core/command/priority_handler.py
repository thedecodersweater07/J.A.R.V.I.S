# priority_handler.py
# Een module voor het beheren van prioriteiten binnen een taken- of berichtensysteem

import time
import logging
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import threading

# Configuratie voor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PriorityHandler")

class PriorityItem:
    """Representeert een item met prioriteit in het systeem."""
    
    def __init__(self, item_id: str, priority: int, data: Any, expiry: datetime = None):
        """
        Initialiseert een nieuw prioriteitsitem.
        
        Args:
            item_id: Unieke identifier voor het item
            priority: Prioriteitsniveau (lager getal = hogere prioriteit)
            data: De data of payload van het item
            expiry: Optionele vervaldatum voor het item
        """
        self.item_id = item_id
        self.priority = priority
        self.data = data
        self.expiry = expiry
        self.created_at = datetime.now()
        
    def is_expired(self) -> bool:
        """Controleert of het item verlopen is."""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
    
    def __lt__(self, other):
        """Voor vergelijking in de priority queue (lager = hogere prioriteit)."""
        return self.priority < other.priority

class PriorityHandler:
    """
    Klasse voor het beheren van prioriteiten in een systeem.
    Ondersteunt prioriteitsqueues, dynamische prioriteitsaanpassing en vervaldatum.
    """
    
    def __init__(self, auto_cleanup: bool = True, cleanup_interval: int = 300):
        """
        Initialiseert de PriorityHandler.
        
        Args:
            auto_cleanup: Of automatische opschoning van verlopen items actief is
            cleanup_interval: Interval in seconden voor automatische opschoning
        """
        self.items = {}  # Dictionary van alle items op ID
        self.priority_queue = []  # Priority queue als heap
        self.lock = threading.RLock()  # Thread-safe operaties
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self.stop_cleanup = threading.Event()
        
        if auto_cleanup:
            self._start_cleanup_thread()
            
        logger.info("PriorityHandler geïnitialiseerd met auto_cleanup=%s", auto_cleanup)
    
    def _start_cleanup_thread(self):
        """Start de thread voor automatische opschoning."""
        thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        thread.start()
        logger.debug("Cleanup thread gestart")
    
    def _cleanup_loop(self):
        """Hoofdlus voor de cleanup thread."""
        while not self.stop_cleanup.is_set():
            time.sleep(self.cleanup_interval)
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.error("Fout in cleanup thread: %s", str(e))
    
    def add_item(self, item_id: str, priority: int, data: Any, 
                expires_in: Optional[timedelta] = None) -> PriorityItem:
        """
        Voegt een nieuw item toe aan de priority handler.
        
        Args:
            item_id: Unieke identifier voor het item
            priority: Prioriteitsniveau (lager = hogere prioriteit)
            data: De data of payload van het item
            expires_in: Optionele vervaltijd vanaf nu
            
        Returns:
            Het toegevoegde PriorityItem
        """
        expiry = None
        if expires_in:
            expiry = datetime.now() + expires_in
        
        item = PriorityItem(item_id, priority, data, expiry)
        
        with self.lock:
            self.items[item_id] = item
            heapq.heappush(self.priority_queue, (priority, item_id))
        
        logger.info("Item %s toegevoegd met prioriteit %d", item_id, priority)
        return item
    
    def get_next_item(self) -> Optional[PriorityItem]:
        """
        Haalt het item met de hoogste prioriteit op en verwijdert het.
        
        Returns:
            Het PriorityItem met de hoogste prioriteit, of None als er geen items zijn
        """
        with self.lock:
            while self.priority_queue:
                # Haal het item met de hoogste prioriteit
                _, item_id = heapq.heappop(self.priority_queue)
                
                # Controleer of het item nog bestaat en niet verlopen is
                if item_id in self.items:
                    item = self.items[item_id]
                    if not item.is_expired():
                        del self.items[item_id]
                        logger.info("Item %s opgehaald en verwijderd", item_id)
                        return item
                    else:
                        # Verwijder verlopen items
                        del self.items[item_id]
                        logger.debug("Verlopen item %s verwijderd", item_id)
                
            return None
    
    def peek_next_item(self) -> Optional[PriorityItem]:
        """
        Toont het item met de hoogste prioriteit zonder het te verwijderen.
        
        Returns:
            Het PriorityItem met de hoogste prioriteit, of None als er geen items zijn
        """
        with self.lock:
            # Maak een kopie van de priority queue voor het zoeken
            temp_queue = self.priority_queue.copy()
            
            while temp_queue:
                # Haal het item met de hoogste prioriteit
                _, item_id = heapq.heappop(temp_queue)
                
                # Controleer of het item nog bestaat en niet verlopen is
                if item_id in self.items:
                    item = self.items[item_id]
                    if not item.is_expired():
                        return item
            
            return None
    
    def update_priority(self, item_id: str, new_priority: int) -> bool:
        """
        Werkt de prioriteit van een bestaand item bij.
        
        Args:
            item_id: ID van het item om bij te werken
            new_priority: De nieuwe prioriteitswaarde
            
        Returns:
            True als het bijwerken gelukt is, anders False
        """
        with self.lock:
            if item_id not in self.items:
                logger.warning("Item %s niet gevonden voor prioriteitsupdate", item_id)
                return False
            
            item = self.items[item_id]
            old_priority = item.priority
            item.priority = new_priority
            
            # Voeg een nieuw item toe aan de queue met de bijgewerkte prioriteit
            # (het oude blijft, maar wordt overgeslagen bij get_next_item)
            heapq.heappush(self.priority_queue, (new_priority, item_id))
            
            logger.info("Prioriteit van item %s bijgewerkt van %d naar %d", 
                       item_id, old_priority, new_priority)
            return True