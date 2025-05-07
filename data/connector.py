from .db.database import DatabaseManager
from .loader import DataLoader

class DataConnector:
    def __init__(self):
        self.db = DatabaseManager()
        self.loader = DataLoader()
        
    def save_conversation(self, user_id, message, response):
        self.db.save_conversation(user_id, message, response)
        # Also save to training data
        self.loader.save_training_data({
            'user_id': user_id,
            'message': message,
            'response': response
        })
        
    def get_conversation_history(self, user_id):
        return self.db.get_conversations(user_id)
        
    def get_metrics(self):
        return self.db.load_metrics()
