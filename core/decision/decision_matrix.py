#!/usr/bin/env python3
"""
decision_matrix.py - Evaluates options against weighted criteria to aid decision making.
"""

import json
from typing import Dict, List, Tuple


class DecisionMatrix:
    """A weighted decision matrix to evaluate options based on multiple criteria."""
    
    def __init__(self, name: str = "New Decision"):
        self.name = name
        self.options: List[str] = []
        self.criteria: List[str] = []
        self.weights: Dict[str, float] = {}
        self.scores: Dict[str, Dict[str, float]] = {}
        
    def add_option(self, option: str) -> None:
        """Add a new option to the decision matrix."""
        if option not in self.options:
            self.options.append(option)
            for criterion in self.criteria:
                if criterion not in self.scores:
                    self.scores[criterion] = {}
                self.scores[criterion][option] = 0.0
    
    def add_criterion(self, criterion: str, weight: float = 1.0) -> None:
        """Add a new evaluation criterion with a weight."""
        if criterion not in self.criteria:
            self.criteria.append(criterion)
            self.weights[criterion] = weight
            self.scores[criterion] = {option: 0.0 for option in self.options}
    
    def set_weight(self, criterion: str, weight: float) -> None:
        """Set the weight for an existing criterion."""
        if criterion in self.criteria:
            self.weights[criterion] = weight
    
    def score_option(self, option: str, criterion: str, score: float) -> None:
        """Score an option against a specific criterion."""
        if option in self.options and criterion in self.criteria:
            self.scores[criterion][option] = score
    
    def calculate_results(self) -> Dict[str, float]:
        """Calculate the weighted scores for all options."""
        results = {}
        for option in self.options:
            weighted_sum = 0.0
            for criterion in self.criteria:
                weighted_sum += self.scores[criterion][option] * self.weights[criterion]
            results[option] = weighted_sum
        return results
    
    def get_ranked_options(self) -> List[Tuple[str, float]]:
        """Get options ranked by their score from highest to lowest."""
        results = self.calculate_results()
        return sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    def save_to_file(self, filename: str) -> None:
        """Save the decision matrix to a JSON file."""
        data = {
            "name": self.name,
            "options": self.options,
            "criteria": self.criteria,
            "weights": self.weights,
            "scores": self.scores
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_file(cls, filename: str) -> 'DecisionMatrix':
        """Load a decision matrix from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        matrix = cls(data["name"])
        matrix.options = data["options"]
        matrix.criteria = data["criteria"]
        matrix.weights = data["weights"]
        matrix.scores = data["scores"]
        return matrix