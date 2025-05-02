#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Opdrachtenparser voor het Commandocentrum
Verantwoordelijk voor het parsen en valideren van opdrachten
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

# Configuratie voor logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandParser:
    """Parser voor het verwerken van opdrachten in het commandocentrum"""
    
    def __init__(self, command_schema_file: str = "command_schema.json"):
        """
        Initialiseer de commandoparser
        
        Args:
            command_schema_file: Pad naar het JSON-schema voor opdrachten
        """
        self.schema = self._load_schema(command_schema_file)
        self.command_patterns = self._compile_patterns()
        logger.info("CommandParser geïnitialiseerd")
    
    def _load_schema(self, schema_file: str) -> Dict[str, Any]:
        """Laad het schema voor opdrachten"""
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Fout bij laden schema: {e}")
            # Terugvallen op een basis schema als het bestand niet gevonden wordt
            return {
                "commands": {
                    "start": {"args": ["process_name"], "required": True},
                    "stop": {"args": ["process_id"], "required": True},
                    "status": {"args": [], "required": False},
                    "help": {"args": [], "required": False}
                }
            }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compileer regex patronen voor alle gedefinieerde opdrachten"""
        patterns = {}
        for cmd, config in self.schema.get("commands", {}).items():
            # Bouw een patroon op basis van de opdrachtconfiguratie
            arg_pattern = r'\s+'.join([rf'(\S+)' for _ in config.get("args", [])])
            if arg_pattern:
                patterns[cmd] = re.compile(rf'^{cmd}\s+{arg_pattern}$', re.IGNORECASE)
            else:
                patterns[cmd] = re.compile(rf'^{cmd}$', re.IGNORECASE)
        return patterns
    
    def parse(self, command_str: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse een opdrachtstring en retourneer opdrachttype en argumenten
        
        Args:
            command_str: De te parsen opdrachtregel
        
        Returns:
            Tuple met opdrachttype en lijst van argumenten, of (None, []) als niet geldig
        """
        command_str = command_str.strip()
        
        # Controleer eerst de hoofdopdracht
        cmd_parts = command_str.split(maxsplit=1)
        if not cmd_parts:
            logger.warning("Lege opdracht ontvangen")
            return None, []
            
        cmd_type = cmd_parts[0].lower()
        
        # Als de opdracht niet in het schema staat
        if cmd_type not in self.schema.get("commands", {}):
            logger.warning(f"Onbekende opdracht: {cmd_type}")
            return None, []
        
        # Match de opdracht tegen het patroon
        match = self.command_patterns[cmd_type].match(command_str)
        if not match:
            logger.warning(f"Ongeldige opdrachtsyntax: {command_str}")
            return None, []
            
        # Haal argumenten op uit de match
        args = list(match.groups())
        
        # Valideer argumenten
        if self.validate_args(cmd_type, args):
            return cmd_type, args
        
        return None, []
    
    def validate_args(self, cmd_type: str, args: List[str]) -> bool:
        """
        Valideer argumenten voor een opdracht
        
        Args:
            cmd_type: Type van de opdracht
            args: Lijst met argumenten
        
        Returns:
            Boolean die aangeeft of de argumenten geldig zijn
        """
        cmd_config = self.schema.get("commands", {}).get(cmd_type, {})
        expected_args = len(cmd_config.get("args", []))
        
        if len(args) != expected_args:
            logger.warning(f"Verkeerd aantal argumenten voor {cmd_type}. Verwacht: {expected_args}, gekregen: {len(args)}")
            return False
            
        # Hier kunnen extra validaties voor specifieke argumenten worden toegevoegd
        
        return True

# Voorbeeld gebruik
if __name__ == "__main__":
    parser = CommandParser()
    cmd, args = parser.parse("start process_name")
    print(f"Commando: {cmd}, Argumenten: {args}")