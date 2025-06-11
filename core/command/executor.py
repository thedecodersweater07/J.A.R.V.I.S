#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uitvoeringslogica voor het Commandocentrum
Verantwoordelijk voor het uitvoeren van gevalideerde opdrachten
"""

import os
import sys
import time
import uuid
import logging
import subprocess
from typing import Dict, Any, List, Tuple, Optional

# Configuratie voor logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandExecutor:
    """Executes parsed commands"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.commands = {}
        self.running_processes = {}  # Correct geplaatst
        self.max_concurrent_processes = 5
        self.logger.info("CommandExecutor initialized with max concurrent processes set to 5")

    def register_command(self, command: str, handler: callable) -> None: # type: ignore
        """Register a command with its handler function"""
        if not callable(handler):
            raise ValueError("Handler must be a callable function")
        self.commands[command] = handler
        self.logger.info(f"Command '{command}' registered successfully")

    def unregister_command(self, command: str) -> None:
        """Unregister a command"""
        if command in self.commands:
            del self.commands[command]
            self.logger.info(f"Command '{command}' unregistered successfully")
        else:
            self.logger.warning(f"Command '{command}' not found for unregistration")

    def parse_command(self, command_str: str) -> Dict[str, Any]:
        """Parses a command string into a structured dict"""
        parts = command_str.strip().split()
        if not parts:
            return {"command": "help"}

        command = parts[0]
        args = parts[1:]
        return {"command": command, "args": args}

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a parsed command"""
        try:
            cmd = command.get("command", "noop")
            handler = self.commands.get(cmd)
            if handler:
                return handler(command)
            return {"status": "error", "message": f"Unknown command: {cmd}"}
        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _start_process(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Start een nieuw proces via command args"""
        if len(self.running_processes) >= self.max_concurrent_processes:
            return {"status": "error", "message": "Maximum aantal gelijktijdige processen bereikt"}

        args = command.get("args", [])
        if not args:
            return {"status": "error", "message": "Geen procesnaam opgegeven"}

        process_name = args[0]
        process_id = str(uuid.uuid4())[:8]

        try:
            self.running_processes[process_id] = {
                "name": process_name,
                "start_time": time.time(),
                "status": "running"
            }
            self.logger.info(f"Starten proces: {process_name} met ID {process_id}")
            return {"status": "success", "message": f"Proces {process_name} gestart met ID: {process_id}"}
        except Exception as e:
            self.logger.error(f"Fout bij starten van proces {process_name}: {e}")
            return {"status": "error", "message": str(e)}

    def _stop_process(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Stop een lopend proces via ID"""
        args = command.get("args", [])
        if not args:
            return {"status": "error", "message": "Geen proces-ID opgegeven"}

        process_id = args[0]
        if process_id not in self.running_processes:
            return {"status": "error", "message": f"Geen lopend proces met ID {process_id} gevonden"}

        try:
            process_info = self.running_processes[process_id]
            process_info["status"] = "stopped"
            process_info["end_time"] = time.time()
            process_info["runtime"] = process_info["end_time"] - process_info["start_time"]
            self.running_processes.pop(process_id)

            self.logger.info(f"Gestopt proces: {process_info['name']} (ID: {process_id})")
            return {"status": "success", "message": f"Proces {process_info['name']} gestopt"}
        except Exception as e:
            self.logger.error(f"Fout bij stoppen van proces {process_id}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_status(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Toon status van lopende processen"""
        if not self.running_processes:
            return {"status": "success", "message": "Geen actieve processen"}

        status_lines = []
        for pid, info in self.running_processes.items():
            runtime = time.time() - info["start_time"]
            status_lines.append(
                f"ID: {pid}, Naam: {info['name']}, Status: {info['status']}, Runtime: {runtime:.1f}s"
            )
        return {"status": "success", "message": "\n".join(status_lines)}

    def _show_help(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Toon beschikbare commando's"""
        help_text = """
Beschikbare commando's:
- start <procesnaam>: Start een nieuw proces
- stop <proces_id>: Stop een lopend proces
- status: Toon status van alle processen
- help: Toon deze hulp
"""
        return {"status": "success", "message": help_text}