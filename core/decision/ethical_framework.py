#!/usr/bin/env python3
"""
ethical_framework.py - Evaluates decisions based on ethical principles and consequences.
"""

import json
from typing import Dict, List, Tuple, Set
from enum import Enum


class EthicalPrinciple(Enum):
    """Common ethical principles used for decision analysis."""
    AUTONOMY = "Respect for individual autonomy and freedom"
    BENEFICENCE = "Promotion of wellbeing and benefit"
    NON_MALEFICENCE = "Avoiding harm"
    JUSTICE = "Fair and equitable treatment"
    TRANSPARENCY = "Open and honest communication"
    SUSTAINABILITY = "Environmental and social responsibility"
    DIGNITY = "Respect for human dignity"
    PRIVACY = "Protection of personal information"


class Impact(Enum):
    """Impact levels for ethical analysis."""
    VERY_POSITIVE = 2
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    VERY_NEGATIVE = -2


class StakeholderGroup(Enum):
    """Common stakeholder groups to consider in ethical analysis."""
    USERS = "Direct users or customers"
    EMPLOYEES = "Staff and team members"
    COMMUNITY = "Local community"
    ENVIRONMENT = "Natural environment"
    SHAREHOLDERS = "Investors and owners"
    SUPPLIERS = "Suppliers and partners"
    GOVERNMENT = "Regulatory bodies"
    VULNERABLE_GROUPS = "Vulnerable populations"


class EthicalFramework:
    """Framework for ethical analysis of decisions and options."""
    
    def __init__(self, name: str = "Ethical Analysis"):
        self.name = name
        self.options: List[str] = []
        self.principles: Set[EthicalPrinciple] = set()
        self.stakeholders: Set[StakeholderGroup] = set()
        self.impact_matrix: Dict[str, Dict[EthicalPrinciple, Impact]] = {}
        self.stakeholder_impacts: Dict[str, Dict[StakeholderGroup, Impact]] = {}
        
    def add_option(self, option: str) -> None:
        """Add a decision option for ethical analysis."""
        if option not in self.options:
            self.options.append(option)
            self.impact_matrix[option] = {}
            self.stakeholder_impacts[option] = {}
    
    def add_principle(self, principle: EthicalPrinciple) -> None:
        """Add an ethical principle to consider."""
        self.principles.add(principle)
        for option in self.options:
            if principle not in self.impact_matrix[option]:
                self.impact_matrix[option][principle] = Impact.NEUTRAL
    
    def add_stakeholder(self, stakeholder: StakeholderGroup) -> None:
        """Add a stakeholder group to consider."""
        self.stakeholders.add(stakeholder)
        for option in self.options:
            if stakeholder not in self.stakeholder_impacts[option]:
                self.stakeholder_impacts[option][stakeholder] = Impact.NEUTRAL
    
    def set_principle_impact(self, option: str, principle: EthicalPrinciple, impact: Impact) -> None:
        """Set the impact of an option on an ethical principle."""
        if option in self.options and principle in self.principles:
            self.impact_matrix[option][principle] = impact
    
    def set_stakeholder_impact(self, option: str, stakeholder: StakeholderGroup, impact: Impact) -> None:
        """Set the impact of an option on a stakeholder group."""
        if option in self.options and stakeholder in self.stakeholders:
            self.stakeholder_impacts[option][stakeholder] = impact
    
    def calculate_ethical_score(self, option: str) -> float:
        """Calculate an overall ethical score for an option."""
        if option not in self.options:
            return 0.0
        
        principle_score = 0
        for principle in self.principles:
            if principle in self.impact_matrix[option]:
                principle_score += self.impact_matrix[option][principle].value
        
        stakeholder_score = 0
        for stakeholder in self.stakeholders:
            if stakeholder in self.stakeholder_impacts[option]:
                stakeholder_score += self.stakeholder_impacts[option][stakeholder].value
        
        # Average the scores and normalize to a 0-10 scale
        total_items = len(self.principles) + len(self.stakeholders)
        if total_items == 0:
            return 5.0  # Neutral score if no principles or stakeholders
        
        raw_score = (principle_score + stakeholder_score) / total_items
        # Convert from -2 to 2 scale to 0 to 10 scale
        return (raw_score + 2) * 2.5
    
    def get_ranked_options(self) -> List[Tuple[str, float]]:
        """Get options ranked by their ethical score from highest to lowest."""
        scores = [(option, self.calculate_ethical_score(option)) for option in self.options]
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def save_to_file(self, filename: str) -> None:
        """Save the ethical analysis to a JSON file."""
        data = {
            "name": self.name,
            "options": self.options,
            "principles": [p.name for p in self.principles],
            "stakeholders": [s.name for s in self.stakeholders],
            "impact_matrix": {
                option: {p.name: i.name for p, i in impacts.items()}
                for option, impacts in self.impact_matrix.items()
            },
            "stakeholder_impacts": {
                option: {s.name: i.name for s, i in impacts.items()}
                for option, impacts in self.stakeholder_impacts.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)