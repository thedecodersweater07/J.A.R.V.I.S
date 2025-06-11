from jarvis.db.manager import DatabaseManager
from jarvis.data.loader import DataLoader

class LLMManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_loader = DataLoader()
        
    async def process_input(self, user_input: str, user_id: str):
        # Process with LLM
        response = await self._generate_response(user_input)
        # Save to database
        self.db.save_conversation(user_id, user_input, response)
        return response
        
    async def _generate_response(self, prompt: str):
        # LLM processing logic here
        return "AI Response"
        
    def save_training_data(self, data):
        self.data_loader.save_training_data(data)
