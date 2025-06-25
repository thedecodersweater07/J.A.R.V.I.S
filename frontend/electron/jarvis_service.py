"""
Jarvis Service for Electron App
-------------------------------
This script provides a simple IPC interface for the JarvisModel
using stdin/stdout for communication.
"""

import sys
import json
import logging
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("JarvisService")

@dataclass
class JarvisResponse:
    """Standardized response format for Jarvis service"""
    success: bool
    response: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class JarvisService:
    """Main service class for handling Jarvis operations"""
    
    def __init__(self):
        self.jarvis = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self.initialize_jarvis()
    
    def initialize_jarvis(self) -> bool:
        """Initialize the Jarvis model with thread safety"""
        if not JARVIS_AVAILABLE:
            logger.error("JarvisModel is not available. Some features may not work.")
            return False
        
        with self._lock:
            if self.jarvis is not None:
                return True
                
            try:
                from models.jarvis import JarvisModel
                self.jarvis = JarvisModel()
                logger.info("JarvisModel initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize JarvisModel: {e}", exc_info=True)
                return False
    
    def process_input(self, text: str) -> Dict[str, Any]:
        """Process user input through Jarvis with input validation"""
        if not isinstance(text, str) or not text.strip():
            return JarvisResponse(
                success=False,
                response="Please provide valid input text.",
                error="Empty or invalid input"
            ).to_dict()
        
        if not self.jarvis:
            return JarvisResponse(
                success=False,
                response="I'm having trouble initializing my brain. Please try again later.",
                error="JarvisModel not initialized"
            ).to_dict()
        
        try:
            with self._lock:
                response = self.jarvis.process_input(text)
                return JarvisResponse(
                    success=True,
                    response=response.get("response", "I don't have a response for that."),
                    metadata=response
                ).to_dict()
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            return JarvisResponse(
                success=False,
                response="I encountered an error processing your request.",
                error=str(e)
            ).to_dict()
    
    def shutdown(self) -> None:
        """Cleanup resources"""
        with self._lock:
            if hasattr(self.jarvis, 'cleanup'):
                try:
                    self.jarvis.cleanup()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}", exc_info=True)
            self.jarvis = None
        self._shutdown_event.set()

def signal_handler(service: JarvisService, sig, frame) -> None:
    """Handle shutdown signals"""
    logger.info("Shutting down Jarvis service...")
    service.shutdown()
    sys.exit(0)

def main():
    """Main entry point for the service"""
    service = JarvisService()
    
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: signal_handler(service, s, f))
    
    buffer = ""
    try:
        while not service._shutdown_event.is_set():
            # Read from stdin with timeout to allow for graceful shutdown
            try:
                line = sys.stdin.readline()
                if not line:
                    break  # EOF
                
                buffer += line
                
                # Process complete message (ends with blank line)
                if buffer.endswith("\n\n"):
                    process_message(service, buffer.strip())
                    buffer = ""
                    
            except (IOError, KeyboardInterrupt):
                break
                
    finally:
        service.shutdown()

def process_message(service: JarvisService, message_str: str) -> None:
    """Process a single message from stdin"""
    try:
        message = json.loads(message_str)
        msg_type = message.get("type")
        data = message.get("data", {})
        
        response = {"request_id": message.get("request_id")}
        
        if msg_type == "process_input":
            result = service.process_input(str(data.get("text", "")))
            response.update({
                "type": "process_response",
                "data": result
            })
        elif msg_type == "ping":
            response.update({
                "type": "pong",
                "data": {"status": "alive"}
            })
        else:
            response.update({
                "type": "error",
                "error": f"Unknown message type: {msg_type}"
            })
            
        # Send response
        print(json.dumps(response) + "\n\n", flush=True)
        
    except json.JSONDecodeError as e:
        send_error(f"Invalid JSON: {e}", message_str)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        send_error(f"Unexpected error: {e}", message_str)

def send_error(error_msg: str, original_msg: str = "") -> None:
    """Send an error response"""
    error_response = {
        "type": "error",
        "error": error_msg,
        "original_message": original_msg[:1000]  # Limit size to prevent huge logs
    }
    print(json.dumps(error_response) + "\n\n", flush=True)

if __name__ == "__main__":
    main()