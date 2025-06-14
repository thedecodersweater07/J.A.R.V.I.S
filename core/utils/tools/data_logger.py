# dit is een data logger script
#voor het loggen van data activiteiten en gebeurtenissen en opslaan van data 

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# voor data auditing en monitoring
#logging configuratie
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLogger():
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"data_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None) -> None:
        """Log an event with optional data"""
        if data is None:
            data = {}
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(f"{log_entry}\n")
        
        logger.info(f"Logged event: {log_entry}")
    def log_error(self, error_message: str, data: Dict[str, Any] = None) -> None:
        """Log an error with optional data"""
        if data is None:
            data = {}
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "message": error_message,
            "data": data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(f"{log_entry}\n")
        
        logger.error(f"Logged error: {log_entry}")
    def log_info(self, info_message: str, data: Dict[str, Any] = None) -> None: 
        """Log an informational message with optional data"""
        if data is None:
            data = {}
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "info",
            "message": info_message,
            "data": data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(f"{log_entry}\n")
        
        logger.info(f"Logged info: {log_entry}")
    def get_log_file(self) -> Path:
        """Return the path to the log file"""
        return self.log_file
    def read_log(self) -> str:
        """Read the log file and return its contents"""
        if not self.log_file.exists():
            return "Log file does not exist."
        
        with open(self.log_file, 'r') as f:
            return f.read()
    def clear_log(self) -> None:
        """Clear the log file"""
        if self.log_file.exists():
            self.log_file.unlink()
            logger.info("Log file cleared.")
        else:
            logger.warning("Log file does not exist, nothing to clear.")
    def backup_log(self, backup_dir: Path) -> None:
        """Backup the log file to a specified directory"""
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / self.log_file.name
        if self.log_file.exists():
            self.log_file.rename(backup_file)
            logger.info(f"Log file backed up to {backup_file}")
        else:
            logger.warning("Log file does not exist, nothing to backup.")
    def restore_log(self, backup_dir: Path) -> None:
        """Restore the log file from a backup directory"""
        backup_file = backup_dir / self.log_file.name
        if backup_file.exists():
            backup_file.rename(self.log_file)
            logger.info(f"Log file restored from {backup_file}")
        else:
            logger.warning("Backup log file does not exist, nothing to restore.")
    def get_log_summary(self) -> Dict[str, Any]:    
        """Get a summary of the log file"""
        if not self.log_file.exists():
            return {"message": "Log file does not exist."}
        
        summary = {
            "log_file": str(self.log_file),
            "last_modified": datetime.fromtimestamp(self.log_file.stat().st_mtime).isoformat(),
            "size": self.log_file.stat().st_size,
            "entries": []
        }
        
        with open(self.log_file, 'r') as f:
            for line in f:
                summary["entries"].append(json.loads(line.strip()))
        
        return summary
    def get_log_entries(self) -> List[Dict[str, Any]]:
        """Get all log entries as a list of dictionaries"""
        if not self.log_file.exists():
            return []
        
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        
        return entries
    def get_log_entry_count(self) -> int:
        """Get the count of log entries"""
        if not self.log_file.exists():
            return 0
        
        with open(self.log_file, 'r') as f:
            return sum(1 for _ in f)
    def get_log_entry_by_index(self, index: int) -> Dict[str, Any]:
        """Get a specific log entry by index"""
        entries = self.get_log_entries()
        if 0 <= index < len(entries):
            return entries[index]
        else:
            raise IndexError("Log entry index out of range.")
    def search_log(self, keyword: str) -> List[Dict[str, Any]]: 
        """Search log entries for a keyword"""
        entries = self.get_log_entries()
        return [entry for entry in entries if keyword.lower() in json.dumps(entry).lower()]

class DataLoggerError(Exception):
    """Custom exception for DataLogger errors"""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)
def main():
    # Example usage of DataLogger
    log_dir = Path("logs")
    logger = DataLogger(log_dir)
    
    # Log some events
    logger.log_event("data_import", "Data imported successfully", {"file": "data.csv"})
    logger.log_info("Data processing started")
    logger.log_error("Data processing failed", {"error": "File not found"})
    
    # Read log file
    print(logger.read_log())
    
    # Get log summary
    print(logger.get_log_summary())
    # Backup log
    backup_dir = Path("backup")
    logger.backup_log(backup_dir)
    # Restore log
    logger.restore_log(backup_dir)
    # Clear log
    logger.clear_log()
if __name__ == "__main__":
    main()
