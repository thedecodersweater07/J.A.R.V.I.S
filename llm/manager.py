from db.manager import DatabaseManager

# Provide a minimal DataLoader fallback to satisfy imports if real implementation is missing
try:
    from data.loader import DataLoader  # type: ignore
except ImportError:  # pragma: no cover
    class DataLoader:  # noqa: D401, D101
        """Fallback DataLoader when data.loader is unavailable."""

        def save_training_data(self, data):  # noqa: D401, D401
            # In fallback we just log / drop the data
            import logging

            logging.getLogger(__name__).warning(
                "DataLoader fallback active – training data not persisted"
            )

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
