from typing import List

class TokenManager:
    def __init__(self):
        self.max_context_length = 4096
    
    def count_tokens(self, text: str) -> int:
        # Implement token counting
        return len(text.split())
    
    def truncate_to_fit(self, text: str, max_tokens: int) -> str:
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        # Implement truncation logic
        return " ".join(text.split()[:max_tokens])
