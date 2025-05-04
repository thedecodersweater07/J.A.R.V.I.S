import re
from typing import List, Optional

class ResponseFilter:
    def __init__(self):
        self.blocked_patterns = [
            r'(dangerous|harmful|offensive)',
            r'(private|sensitive|confidential)',
            r'(password|credential|api[_\s]*key)'
        ]
        self.max_length = 1000

    def filter(self, response: str) -> str:
        # Safety checks
        if not isinstance(response, str):
            return "Error: Invalid response type"
        
        # Length check
        if len(response) > self.max_length:
            response = response[:self.max_length] + "..."
            
        # Content filtering
        for pattern in self.blocked_patterns:
            response = re.sub(pattern, '[FILTERED]', response, flags=re.IGNORECASE)
            
        return response.strip()
