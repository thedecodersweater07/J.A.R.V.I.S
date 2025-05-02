#!/usr/bin/env python3
"""
priority_system.py - Prioritizes tasks or options based on urgency, importance, and resources.
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Priority levels for tasks or options."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task or option to be prioritized."""
    id: str
    name: str
    description: str
    importance: float  # 0-10 scale
    urgency: float  # 0-10 scale
    effort: float  # 0-10 scale (resources required)
    dependencies: List[str] = None  # IDs of tasks this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PrioritySystem:
    """System for prioritizing tasks based on multiple factors."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        # Weights for the priority calculation
        self.importance_weight = 0.4
        self.urgency_weight = 0.4
        self.effort_weight = 0.2
        
    def add_task(self, task: Task) -> None:
        """Add a task to the priority system."""
        self.tasks[task.id] = task
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the priority system."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def calculate_priority_score(self, task: Task) -> float:
        """Calculate a priority score based on importance, urgency, and effort."""
        # Higher importance and urgency increase priority
        # Higher effort decreases priority (more resource-intensive)
        score = (
            (task.importance * self.importance_weight) +
            (task.urgency * self.urgency_weight) +
            ((10 - task.effort) * self.effort_weight)
        )
        return score
    
    def get_priority_level(self, score: float) -> Priority:
        """Convert a numerical score to a priority level."""
        if score >= 8.5:
            return Priority.CRITICAL
        elif score >= 6.5:
            return Priority.HIGH
        elif score >= 4.0:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def get_prioritized_tasks(self) -> List[Tuple[Task, float, Priority]]:
        """Get tasks sorted by priority score with their priority levels."""
        result = []
        for task_id, task in self.tasks.items():
            score = self.calculate_priority_score(task)
            priority = self.get_priority_level(score)
            result.append((task, score, priority))
        
        # Sort by priority score in descending order
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def save_to_file(self, filename: str) -> None:
        """Save the priority system to a JSON file."""
        data = {
            "weights": {
                "importance": self.importance_weight,
                "urgency": self.urgency_weight,
                "effort": self.effort_weight
            },
            "tasks": {
                task_id: {
                    "name": task.name,
                    "description": task.description,
                    "importance": task.importance,
                    "urgency": task.urgency,
                    "effort": task.effort,
                    "dependencies": task.dependencies
                }
                for task_id, task in self.tasks.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'PrioritySystem':
        """Load a priority system from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        system = cls()
        system.importance_weight = data["weights"].get("importance", 0.4)
        system.urgency_weight = data["weights"].get("urgency", 0.4)
        system.effort_weight = data["weights"].get("effort", 0.2)
        
        for task_id, task_data in data["tasks"].items():
            task = Task(
                id=task_id,
                name=task_data["name"],
                description=task_data["description"],
                importance=task_data["importance"],
                urgency=task_data["urgency"],
                effort=task_data["effort"],
                dependencies=task_data.get("dependencies", [])
            )
            system.add_task(task)
        
        return system