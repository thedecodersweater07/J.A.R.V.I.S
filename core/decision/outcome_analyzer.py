#!/usr/bin/env python3
"""
outcome_analyzer.py - Analyzes potential outcomes of decisions with probability estimation.
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Outcome:
    """Represents a possible outcome of a decision option."""
    description: str
    probability: float  # 0.0 to 1.0
    impact: float  # -10 to 10 (negative to positive)
    confidence: float  # 0.0 to 1.0, certainty in the probability estimate


class OutcomeAnalyzer:
    """Analyzes potential outcomes for decision options."""
    
    def __init__(self, name: str = "Outcome Analysis"):
        self.name = name
        self.options: Dict[str, List[Outcome]] = {}
        
    def add_option(self, option: str) -> None:
        """Add a decision option for outcome analysis."""
        if option not in self.options:
            self.options[option] = []
    
    def add_outcome(self, option: str, outcome: Outcome) -> None:
        """Add a possible outcome for an option."""
        if option in self.options:
            self.options[option].append(outcome)
    
    def remove_outcome(self, option: str, index: int) -> bool:
        """Remove an outcome from an option by its index."""
        if option in self.options and 0 <= index < len(self.options[option]):
            self.options[option].pop(index)
            return True
        return False
    
    def normalize_probabilities(self, option: str) -> None:
        """Normalize outcome probabilities to sum to 1.0."""
        if option in self.options and self.options[option]:
            total_prob = sum(outcome.probability for outcome in self.options[option])
            if total_prob > 0:
                for outcome in self.options[option]:
                    outcome.probability /= total_prob
    
    def calculate_expected_value(self, option: str) -> float:
        """Calculate the expected value (probability-weighted impact) for an option."""
        if option not in self.options:
            return 0.0
        
        return sum(
            outcome.probability * outcome.impact
            for outcome in self.options[option]
        )
    
    def calculate_risk_score(self, option: str) -> float:
        """Calculate a risk score for the option.
        Lower score means lower risk (more desirable)."""
        if option not in self.options:
            return 0.0
        
        # Risk score considers negative outcomes weighted by probability
        risk = 0.0
        for outcome in self.options[option]:
            if outcome.impact < 0:
                # Negative impact contributes to risk
                risk += outcome.probability * abs(outcome.impact)
        
        return risk
    
    def calculate_opportunity_score(self, option: str) -> float:
        """Calculate an opportunity score for the option.
        Higher score means higher opportunity (more desirable)."""
        if option not in self.options:
            return 0.0
        
        # Opportunity score considers positive outcomes weighted by probability
        opportunity = 0.0
        for outcome in self.options[option]:
            if outcome.impact > 0:
                # Positive impact contributes to opportunity
                opportunity += outcome.probability * outcome.impact
        
        return opportunity
    
    def calculate_uncertainty(self, option: str) -> float:
        """Calculate uncertainty score for an option (0.0-1.0).
        Low confidence in outcomes or highly variable outcomes increase uncertainty."""
        if option not in self.options or not self.options[option]:
            return 1.0  # Maximum uncertainty when no outcomes are defined
        
        # Average confidence as a component of certainty
        avg_confidence = sum(o.confidence for o in self.options[option]) / len(self.options[option])
        
        # Entropy as a measure of outcome distribution's uncertainty
        entropy = 0.0
        for outcome in self.options[option]:
            if outcome.probability > 0:
                entropy -= outcome.probability * math.log2(outcome.probability)
        
        # Normalize entropy to 0-1 scale (maximum entropy is log2(n))
        max_entropy = math.log2(len(self.options[option])) if len(self.options[option]) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Combined uncertainty score (weighted average of confidence and entropy)
        uncertainty = (0.6 * (1 - avg_confidence)) + (0.4 * normalized_entropy)
        return min(1.0, max(0.0, uncertainty))
    
    def get_ranked_options(self) -> List[Tuple[str, Dict[str, float]]]:
        """Get options ranked by expected value from highest to lowest."""
        results = []
        for option in self.options:
            metrics = {
                "expected_value": self.calculate_expected_value(option),
                "risk": self.calculate_risk_score(option),
                "opportunity": self.calculate_opportunity_score(option),
                "uncertainty": self.calculate_uncertainty(option)
            }
            results.append((option, metrics))
        
        # Sort by expected value
        return sorted(results, key=lambda x: x[1]["expected_value"], reverse=True)
    
    def save_to_file(self, filename: str) -> None:
        """Save the outcome analysis to a JSON file."""
        data = {
            "name": self.name,
            "options": {
                option: [
                    {
                        "description": outcome.description,
                        "probability": outcome.probability,
                        "impact": outcome.impact,
                        "confidence": outcome.confidence
                    }
                    for outcome in outcomes
                ]
                for option, outcomes in self.options.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)