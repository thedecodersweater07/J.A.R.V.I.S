import random
from typing import Dict, List
import numpy as np

class ScenarioGenerator:
    def __init__(self):
        self.templates = {
            "probleem_oplossen": [
                "Er is een {probleem} in {locatie}. Wat zou je doen?",
                "Systeem {systeem} werkt niet vanwege {reden}. Hoe los je dit op?"
            ],
            "kennis_test": [
                "Leg uit hoe {onderwerp} werkt in relatie tot {context}",
                "Wat is de beste aanpak voor {situatie} wanneer {conditie}?"
            ]
        }
        
    def generate_scenario(self, complexity: float = 0.5) -> Dict[str, any]:
        """Generate a test scenario"""
        scenario_type = random.choice(list(self.templates.keys()))
        template = random.choice(self.templates[scenario_type])
        
        variables = self._generate_variables(complexity)
        scenario = template.format(**variables)
        
        return {
            "type": scenario_type,
            "description": scenario,
            "variables": variables,
            "complexity": complexity
        }
