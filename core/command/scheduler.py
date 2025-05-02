#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Opdrachtplanner voor het Commandocentrum
Beheert en plant de uitvoering van opdrachten in de tijd
"""

import os
import time
import uuid
import json
import heapq
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable

# Configuratie voor logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandScheduler:
    """Planner voor het beheren en plannen van opdrachten in het commandocentrum"""
    
    def __init__(self, scheduler_config_file: str = "scheduler_config.json"):
        """
        Initialiseer de opdrachtplanner
        
        Args:
            scheduler_config_file: Pad naar configuratiebestand voor de planner
        """
        self.priority_queue = []  # Heap voor geplande taken
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}  # Opslag voor taakgegevens
        self.recurring_tasks: Dict[str, Dict[str, Any]] = {}  # Terugkerende taken
        self.lock = threading.RLock()  # Thread-veilige toegang tot de wachtrij
        self.is_running = False
        self.scheduler_thread = None
        self._load_config(scheduler_config_file)
        logger.info("CommandScheduler geïnitialiseerd")
    
    def _load_config(self, config_file: str) -> None:
        """Laad configuratie voor de planner"""
        default_config = {
            "max_queue_size": 1000,
            "min_execution_interval": 0.1,  # seconden
            "check_interval": 1.0,  # seconden voor de scheduler loop
            "default_priority": 5  # op een schaal van 1-10
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.config = {**default_config, **loaded_config}
            else:
                self.config = default_config
                logger.warning(f"Configuratiebestand {config_file} niet gevonden, standaardconfiguratie gebruikt")
        except Exception as e:
            logger.error(f"Fout bij laden configuratie: {e}")
            self.config = default_config
    
    def schedule_task(self, 
                      command_type: str, 
                      args: List[str], 
                      execution_time: Optional[datetime] = None,
                      priority: int = None,
                      task_id: Optional[str] = None) -> str:
        """
        Plan een taak in voor uitvoering
        
        Args:
            command_type: Type van het commando
            args: Argumenten voor het commando
            execution_time: Wanneer de taak moet worden uitgevoerd (standaard: nu)
            priority: Prioriteit van de taak (1-10, 10 is hoogste)
            task_id: Optionele taak-ID (wordt automatisch gegenereerd als niet opgegeven)
            
        Returns:
            ID van de ingeplande taak
        """
        # Genereer taak-ID als deze niet is opgegeven
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Standaardwaarden
        if execution_time is None:
            execution_time = datetime.now()
        
        if priority is None:
            priority = self.config["default_priority"]
        
        # Begrens prioriteit
        priority = max(1, min(10, priority))
        
        # Stel de uitvoeringstijd in naar Unix epoch time voor de wachtrij
        execution_timestamp = execution_time.timestamp()
        
        # Maak de taak aan
        task = {
            "id": task_id,
            "command_type": command_type,
            "args": args,
            "scheduled_time": execution_timestamp,
            "priority": priority,
            "status": "scheduled"
        }
        
        with self.lock:
            # Controleer of we niet te veel taken hebben
            if len(self.scheduled_tasks) >= self.config["max_queue_size"]:
                logger.warning("Maximum aantal geplande taken bereikt")
                return ""
            
            # Voeg toe aan de wachtrij - heapq sorteert op eerste element (tijd) en dan tweede (prioriteit)
            # We gebruiken negatieve prioriteit omdat heapq een min-heap is
            heapq.heappush(self.priority_queue, (execution_timestamp, -priority, task_id))
            self.scheduled_tasks[task_id] = task
            
            logger.info(f"Taak ingepland: {task_id}, type: {command_type}, tijd: {execution_time}")
            
        return task_id
    
    def schedule_recurring_task(self,
                               command_type: str,
                               args: List[str],
                               interval: timedelta,
                               priority: int = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> str:
        """
        Plan een terugkerende taak in
        
        Args:
            command_type: Type van het commando
            args: Argumenten voor het commando
            interval: Tijdsinterval tussen uitvoeringen
            priority: Prioriteit van de taak (1-10, 10 is hoogste)
            start_time: Starttijd voor de eerste uitvoering (standaard: nu)
            end_time: Eindtijd na wanneer de taak niet meer wordt uitgevoerd (optioneel)
            
        Returns:
            ID van de terugkerende taak
        """
        recurring_id = str(uuid.uuid4())
        
        if start_time is None:
            start_time = datetime.now()
            
        if priority is None:
            priority = self.config["default_priority"]
            
        # Maak de terugkerende taakdefinitie
        recurring_task = {
            "id": recurring_id,
            "command_type": command_type,
            "args": args,
            "interval_seconds": interval.total_seconds(),
            "priority": priority,
            "start_time": start_time.timestamp(),
            "end_time": end_time.timestamp() if end_time else None,
            "last_scheduled": None
        }
        
        with self.lock:
            self.recurring_tasks[recurring_id] = recurring_task
            
            # Plan direct de eerste uitvoering
            first_task_id = self.schedule_task(
                command_type=command_type,
                args=args,
                execution_time=start_time,
                priority=priority
            )
            
            # Update de laatste geplande tijd
            recurring_task["last_scheduled"] = start_time.timestamp()
            recurring_task["last_task_id"] = first_task_id
            
            logger.info(f"Terugkerende taak ingepland: {recurring_id}, interval: {interval}")
            
        return recurring_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Annuleer een geplande taak
        
        Args:
            task_id: ID van de te annuleren taak
            
        Returns:
            Boolean die aangeeft of de annulering succesvol was
        """
        with self.lock:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task["status"] = "cancelled"
                
                # Opmerking: we verwijderen het niet uit de priority_queue omdat dat duur is
                # In plaats daarvan controleren we de status bij het uitvoeren
                logger.info(f"Taak geannuleerd: {task_id}")
                return True
            else:
                logger.warning(f"Taak niet gevonden voor annulering: {task_id}")
                return False
    
    def cancel_recurring_task(self, recurring_id: str) -> bool:
        """
        Annuleer een terugkerende taak
        
        Args:
            recurring_id: ID van de terugkerende taak
            
        Returns:
            Boolean die aangeeft of de annulering succesvol was
        """
        with self.lock:
            if recurring_id in self.recurring_tasks:
                self.recurring_tasks.pop(recurring_id)
                logger.info(f"Terugkerende taak geannuleerd: {recurring_id}")
                return True
            else:
                logger.warning(f"Terugkerende taak niet gevonden: {recurring_id}")
                return False
    
    def start(self, executor_callback: Callable[[str, List[str]], None]) -> None:
        """
        Start de scheduler
        
        Args:
            executor_callback: Functie om taken uit te voeren
        """
        if self.is_running:
            logger.warning("Scheduler draait al")
            return
            
        self.is_running = True
        self.executor_callback = executor_callback
        
        # Start een thread voor de planner
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Scheduler gestart")
    
    def stop(self) -> None:
        """Stop de scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Scheduler gestopt")
    
    def _scheduler_loop(self) -> None:
        """Hoofdlus voor de planner die taken verwerkt"""
        while self.is_running:
            now = time.time()
            tasks_to_execute = []
            
            with self.lock:
                # Zoek taken die uitgevoerd moeten worden
                while self.priority_queue and self.priority_queue[0][0] <= now:
                    # Haal de volgende taak van de wachtrij
                    scheduled_time, neg_priority, task_id = heapq.heappop(self.priority_queue)
                    
                    # Controleer of de taak niet is geannuleerd
                    if task_id in self.scheduled_tasks and self.scheduled_tasks[task_id]["status"] == "scheduled":
                        task = self.scheduled_tasks[task_id]
                        task["status"] = "executing"
                        tasks_to_execute.append((task_id, task["command_type"], task["args"]))
                
                # Plan de volgende uitvoering voor terugkerende taken
                for r_id, recurring in self.recurring_tasks.items():
                    if recurring["last_scheduled"] is None:
                        continue
                        
                    # Bereken wanneer de volgende uitvoering zou moeten zijn
                    next_execution = recurring["last_scheduled"] + recurring["interval_seconds"]
                    
                    # Als het tijd is voor de volgende uitvoering
                    if next_execution <= now:
                        # Controleer of we de eindtijd hebben bereikt
                        if recurring["end_time"] and now > recurring["end_time"]:
                            # Verwijder de terugkerende taak als we voorbij de eindtijd zijn
                            self.recurring_tasks.pop(r_id)
                            logger.info(f"Terugkerende taak {r_id} beëindigd (eindtijd bereikt)")
                            continue
                            
                        # Plan de volgende uitvoering
                        next_task_id = self.schedule_task(
                            command_type=recurring["command_type"],
                            args=recurring["args"],
                            execution_time=datetime.fromtimestamp(next_execution),
                            priority=recurring["priority"]
                        )
                        
                        # Update de laatste geplande tijd
                        recurring["last_scheduled"] = next_execution
                        recurring["last_task_id"] = next_task_id
                        logger.debug(f"Volgende uitvoering gepland voor terugkerende taak {r_id}")
            
            # Voer de taken uit buiten de lock om blokkeren te voorkomen
            for task_id, command_type, args in tasks_to_execute:
                try:
                    self.executor_callback(command_type, args)
                except Exception as e:
                    logger.error(f"Fout bij uitvoeren taak {task_id}: {e}")
                finally:
                    # Verwijder de taak uit de planning na uitvoering
                    with self.lock:
                        if task_id in self.scheduled_tasks:
                            self.scheduled_tasks.pop(task_id)
            
            # Slaap voor een korte tijd om CPU-gebruik te beperken
            time.sleep(self.config["check_interval"])
    
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        Geef een lijst van alle geplande taken terug
        
        Returns:
            Lijst met taakdetails
        """
        with self.lock:
            return [task for task in self.scheduled_tasks.values() if task["status"] == "scheduled"]
    
    def get_recurring_tasks(self) -> List[Dict[str, Any]]:
        """
        Geef een lijst van alle terugkerende taken terug
        
        Returns:
            Lijst met terugkerende taakdetails
        """
        with self.lock:
            return list(self.recurring_tasks.values())