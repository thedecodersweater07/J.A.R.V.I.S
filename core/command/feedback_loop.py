#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedbackmechanisme voor het Commandocentrum
Verzamelt en verwerkt feedback over uitgevoerde opdrachten
"""

import os
import json
import time
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Configuratie voor logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackLoop:
    """
    Feedbackmechanisme voor het verbeteren van de opdrachtenverwerking
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialisatie van het feedbackmechanisme
        
        Args:
            db_path: Pad naar de SQLite database voor feedback opslag
        """
        self.db_path = db_path
        self._init_database()
        self.feedback_thresholds = {
            "success_rate": 0.8,  # 80% succes vereist
            "response_time": 5.0,  # seconden
            "user_satisfaction": 3.5  # op een schaal van 1-5
        }
        logger.info("FeedbackLoop geïnitialiseerd")
    
    def _init_database(self) -> None:
        """Initialiseer de feedbackdatabase"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Maak tabellen aan indien ze nog niet bestaan
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS command_execution (
                execution_id TEXT PRIMARY KEY,
                command_type TEXT NOT NULL,
                arguments TEXT,
                timestamp REAL NOT NULL,
                success BOOLEAN NOT NULL,
                execution_time REAL NOT NULL,
                error_message TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                execution_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comments TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (execution_id) REFERENCES command_execution(execution_id)
            )
            ''')
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialisatiefout: {e}")
            # Maak een in-memory fallback als het bestand niet toegankelijk is
            self.db_path = ":memory:"
            logger.warning("Teruggevallen op in-memory database")
            self._init_database()
    
    def record_execution(self, 
                         execution_id: str, 
                         command_type: str, 
                         arguments: List[str],
                         success: bool, 
                         execution_time: float,
                         error_message: Optional[str] = None) -> bool:
        """
        Registreer een commando-uitvoering
        
        Args:
            execution_id: Unieke ID voor deze uitvoering
            command_type: Type van het uitgevoerde commando
            arguments: Lijst met argumenten
            success: Of de uitvoering succesvol was
            execution_time: Tijdsduur van de uitvoering in seconden
            error_message: Foutbericht indien van toepassing
        
        Returns:
            Boolean die aangeeft of de registratie succesvol was
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO command_execution VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    execution_id,
                    command_type,
                    json.dumps(arguments),
                    time.time(),
                    success,
                    execution_time,
                    error_message
                )
            )
            
            conn.commit()
            conn.close()
            logger.debug(f"Uitvoering geregistreerd: {execution_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Fout bij registreren uitvoering: {e}")
            return False
    
    def record_user_feedback(self, 
                            feedback_id: str,
                            execution_id: str, 
                            rating: int,
                            comments: Optional[str] = None) -> bool:
        """
        Registreer gebruikersfeedback over een commando-uitvoering
        
        Args:
            feedback_id: Unieke ID voor deze feedback
            execution_id: ID van de betreffende commando-uitvoering
            rating: Gebruikerswaardering (1-5)
            comments: Tekstuele feedback van de gebruiker
        
        Returns:
            Boolean die aangeeft of de registratie succesvol was
        """
        if not 1 <= rating <= 5:
            logger.warning(f"Ongeldige waardering: {rating}, moet tussen 1 en 5 liggen")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO user_feedback VALUES (?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    execution_id,
                    rating,
                    comments,
                    time.time()
                )
            )
            
            conn.commit()
            conn.close()
            logger.debug(f"Feedback geregistreerd: {feedback_id} voor uitvoering {execution_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Fout bij registreren feedback: {e}")
            return False
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Bereken prestatiemetriek over een periode
        
        Args:
            days: Aantal dagen om terug te kijken
        
        Returns:
            Dictionary met prestatiestatistieken
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Bereken tijdsgrens
            time_threshold = time.time() - (days * 24 * 60 * 60)
            
            # Haal statistieken op
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                    AVG(execution_time) as avg_execution_time
                FROM command_execution
                WHERE timestamp > ?
            """, (time_threshold,))
            
            exec_stats = cursor.fetchone()
            
            # Haal feedback statistieken op
            cursor.execute("""
                SELECT AVG(rating) as avg_rating
                FROM user_feedback
                WHERE timestamp > ?
            """, (time_threshold,))
            
            feedback_stats = cursor.fetchone()
            
            conn.close()
            
            # Bereken metriek
            total = exec_stats[0] if exec_stats[0] else 0
            success_rate = exec_stats[1] / total if total > 0 else 0
            avg_time = exec_stats[2] if exec_stats[2] else 0
            avg_rating = feedback_stats[0] if feedback_stats[0] else 0
            
            return {
                "total_executions": total,
                "success_rate": success_rate,
                "average_execution_time": avg_time,
                "average_rating": avg_rating,
                "period_days": days
            }
            
        except sqlite3.Error as e:
            logger.error(f"Fout bij ophalen prestatiemetriek: {e}")
            return {
                "error": str(e),
                "status": "Kon prestatiemetriek niet berekenen"
            }