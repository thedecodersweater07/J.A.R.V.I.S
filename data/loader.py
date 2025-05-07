import pandas as pd
import json
import os

class DataLoader:
    def __init__(self, data_dir="files"):
        self.data_dir = data_dir
        
    def load_csv(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        return pd.read_csv(file_path)
        
    def load_config(self):
        with open(os.path.join(self.data_dir, "config.json"), "r") as f:
            return json.load(f)
            
    def save_training_data(self, data, filename="training_data.csv"):
        file_path = os.path.join(self.data_dir, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            pd.DataFrame(data).to_csv(file_path, index=False)
