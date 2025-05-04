"""Model persistence interface"""
import numpy as np
from bson.binary import Binary
import pickle
from .database import Database

class ModelStore:
    def __init__(self):
        db = Database.get_instance().get_client()
        self.collection = db['jarvis']['models']
    
    def save_model(self, model_name: str, model):
        """Save model parameters to database"""
        params = model.get_params()
        # Convert numpy arrays to binary for MongoDB storage
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params[key] = Binary(pickle.dumps(value, protocol=2))
                
        self.collection.update_one(
            {'model_name': model_name},
            {'$set': params},
            upsert=True
        )
    
    def load_model(self, model_name: str, model):
        """Load model parameters from database"""
        params = self.collection.find_one({'model_name': model_name})
        if params:
            # Convert binary back to numpy arrays
            for key, value in params.items():
                if isinstance(value, Binary):
                    params[key] = pickle.loads(value)
            model.load_params(params)
        return model
