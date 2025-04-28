# executor.py
# Een module voor het uitvoeren van taken gebaseerd op inkomende opdrachten

import time
import logging
import threading
import queue
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime

# Configuratie voor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Executor")

class Task:
    """Representeert een taak die uitgevoerd moet worden."""
    
    def __init__(self, task_id: str, action: Callable, params: Dict[str, Any], priority: int = 0):
        """
        Initialiseert een nieuwe taak.
        
        Args:
            task_id: Unieke identificatie voor de taak
            action: De functie die uitgevoerd moet worden
            params: Parameters voor de functie
            priority: Prioriteit van de taak (hoger = belangrijker)
        """
        self.task_id = task_id
        self.action = action
        self.params = params
        self.priority = priority
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = "pending"
        self.result = None
        self.error = None

class Executor:
    """
    Klasse voor het beheren en uitvoeren van taken op basis van prioriteit.
    Ondersteunt zowel synchrone als asynchrone uitvoering.
    """
    
    def __init__(self, max_workers: int = 5, queue_size: int = 100):
        """
        Initialiseert de Executor met een pool van workers.
        
        Args:
            max_workers: Maximaal aantal parallelle worker threads
            queue_size: Maximale grootte van de taakwachtrij
        """
        self.max_workers = max_workers
        self.tasks = queue.PriorityQueue(maxsize=queue_size)
        self.active_tasks = {}
        self.completed_tasks = {}
        self.shutdown_flag = threading.Event()
        self.workers = []
        self._initialize_workers()
        logger.info("Executor geïnitialiseerd met %d workers", max_workers)
    
    def _initialize_workers(self):
        """Start de worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """
        De hoofdlus voor elke worker thread.
        
        Args:
            worker_id: ID van de worker voor logging
        """
        logger.debug("Worker %d gestart", worker_id)
        while not self.shutdown_flag.is_set():
            try:
                # Wacht op een taak met timeout zodat we regelmatig kunnen controleren op shutdown
                priority, task = self.tasks.get(block=True, timeout=1.0)
                
                # Update task status
                task.status = "running"
                task.started_at = datetime.now()
                self.active_tasks[task.task_id] = task
                
                # Voer de taak uit
                try:
                    logger.info("Worker %d voert taak %s uit", worker_id, task.task_id)
                    task.result = task.action(**task.params)
                    task.status = "completed"
                except Exception as e:
                    logger.error("Fout bij uitvoeren van taak %s: %s", task.task_id, str(e))
                    task.error = str(e)
                    task.status = "failed"
                
                # Werk de taakstatus bij
                task.completed_at = datetime.now()
                self.completed_tasks[task.task_id] = task
                del self.active_tasks[task.task_id]
                
                # Markeer de taak als voltooid in de queue
                self.tasks.task_done()
                
            except queue.Empty:
                # Timeout verlopen, controleer op shutdown flag
                pass
            except Exception as e:
                logger.error("Onverwachte fout in worker %d: %s", worker_id, str(e))
    
    def submit(self, task_id: str, action: Callable, params: Dict[str, Any], priority: int = 0) -> str:
        """
        Dient een nieuwe taak in voor uitvoering.
        
        Args:
            task_id: Unieke ID voor de taak
            action: De functie die uitgevoerd moet worden
            params: Parameters voor de functie
            priority: Prioriteit van de taak (lager = hogere prioriteit)
            
        Returns:
            task_id: De ID van de ingediende taak
        """
        task = Task(task_id, action, params, priority)
        # PriorityQueue sorteert op het eerste element van de tuple
        self.tasks.put((-priority, task))  # Negatief omdat lagere waarden = hogere prioriteit
        logger.info("Taak %s ingediend met prioriteit %d", task_id, priority)
        return task_id