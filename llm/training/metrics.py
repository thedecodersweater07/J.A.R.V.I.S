import numpy as np
from typing import Dict, List, Any

class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metric_dict: Dict[str, float]):
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_average(self, metric_name: str) -> float:
        if metric_name in self.metrics:
            return np.mean(self.metrics[metric_name])
        return 0.0

    def save_epoch(self, epoch: int):
        self.history[epoch] = {
            name: np.mean(values) for name, values in self.metrics.items()
        }
        self.metrics = {}
