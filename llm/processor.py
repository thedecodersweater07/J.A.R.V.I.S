from typing import Dict, Any
import json

class ResponseProcessor:
    def __init__(self):
        self.handlers = {
            "text": self._process_text,
            "json": self._process_json,
            "command": self._process_command
        }
    
    def process_response(self, response: str, response_type: str = "text") -> Any:
        handler = self.handlers.get(response_type, self._process_text)
        return handler(response)
    
    def _process_text(self, response: str) -> str:
        return response.strip()
    
    def _process_json(self, response: str) -> Dict:
        return json.loads(response)
    
    def _process_command(self, response: str) -> Dict:
        # Process command response
        return {"command": response, "status": "processed"}
