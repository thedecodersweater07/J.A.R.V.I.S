from typing import Dict, List, Any
from datetime import datetime, timedelta
import win32com.client
from core.memory import RecallEngine
from llm.knowledge import KnowledgeBaseConnector
from .base_agent import BaseAgent
from .task_agent import TaskAgent
from .assistant_agent import AssistantAgent

class PlannerAgent(BaseAgent):
    """Agent responsible for planning and coordinating tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.active_plans = {}
        self.plan_priorities = {}
        
        # Initialize Outlook integration
        self.outlook = win32com.client.Dispatch("Outlook.Application")
        self.namespace = self.outlook.GetNamespace("MAPI")
        self.calendar = self.namespace.GetDefaultFolder(9)  # 9 = Calendar folder
        
        # Initialize other components
        self.recall = RecallEngine()
        self.knowledge = KnowledgeBaseConnector()
        self.task_agent = TaskAgent(config)
        self.assistant = AssistantAgent(config)
    
    def break_down_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Break down a goal into manageable steps."""
        steps = []
        # Analyze goal and create subtasks
        subtasks = self.analyze_goal(goal)
        
        for task in subtasks:
            step = {
                "description": task,
                "status": "pending",
                "dependencies": [],
                "estimated_time": self.estimate_time(task)
            }
            steps.append(step)
        
        return steps

    def analyze_goal(self, goal: str) -> List[str]:
        """Analyze a goal and break it into subtasks."""
        # Basic goal analysis logic
        subtasks = []
        goal_components = goal.split(" and ")
        
        for component in goal_components:
            subtasks.extend(self._identify_subtasks(component))
        
        return subtasks

    def _identify_subtasks(self, goal_component: str) -> List[str]:
        """Identify specific subtasks from a goal component."""
        # Add task identification logic
        return [f"Subtask for: {goal_component}"]

    def estimate_time(self, task: str) -> int:
        """Estimate time needed for a task in minutes."""
        # Basic time estimation
        return 30  # Default 30 minutes

    def set_priority(self, goal: str, priority: int):
        """Set priority level for a plan (1-5, 5 being highest)."""
        if goal in self.active_plans:
            self.plan_priorities[goal] = max(1, min(5, priority))
            self._reorder_plans()

    def _reorder_plans(self):
        """Reorder plans based on priority."""
        self.active_plans = dict(
            sorted(
                self.active_plans.items(),
                key=lambda x: self.plan_priorities.get(x[0], 1),
                reverse=True
            )
        )

    def create_plan(self, goal: str) -> Dict[str, Any]:
        """Create a new plan and sync with systems."""
        steps = self.break_down_goal(goal)
        plan = {
            "goal": goal,
            "steps": steps,
            "status": "created",
            "progress": 0
        }
        self.active_plans[goal] = plan
        
        # Sync with other systems
        self.sync_with_outlook()
        self.knowledge.update_knowledge({
            "type": "plan",
            "content": plan,
            "timestamp": datetime.now().isoformat()
        })
        
        return plan
        
    def update_plan(self, goal: str, progress: float):
        """Update plan progress and notify systems."""
        if goal in self.active_plans:
            self.active_plans[goal]["progress"] = progress
            if progress >= 1.0:
                self.active_plans[goal]["status"] = "completed"
            
            plan = self.active_plans[goal]
            self.notify_stakeholders(plan)
            self.sync_with_outlook()

    def validate_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate a plan's structure and feasibility."""
        required_keys = ["goal", "steps", "status", "progress"]
        if not all(key in plan for key in required_keys):
            return False
            
        # Check if steps are properly defined
        for step in plan["steps"]:
            if not all(key in step for key in ["description", "status"]):
                return False
                
        return True

    def get_plan_status(self, goal: str) -> Dict[str, Any]:
        """Get detailed status of a specific plan."""
        if goal not in self.active_plans:
            return {"error": "Plan not found"}
            
        plan = self.active_plans[goal]
        return {
            "goal": goal,
            "overall_progress": plan["progress"],
            "status": plan["status"],
            "priority": self.plan_priorities.get(goal, 1),
            "steps_completed": sum(1 for step in plan["steps"] if step["status"] == "completed"),
            "total_steps": len(plan["steps"])
        }

    def sync_with_outlook(self):
        """Synchronize plans with Outlook calendar."""
        for goal, plan in self.active_plans.items():
            if plan["status"] != "completed":
                self._create_outlook_tasks(plan)

    def _create_outlook_tasks(self, plan: Dict[str, Any]):
        """Create Outlook calendar items for plan steps."""
        start_time = datetime.now()
        
        for step in plan["steps"]:
            if step["status"] == "pending":
                # Create calendar item
                appointment = self.calendar.Application.CreateItem(1)  # 1 = appointment item
                appointment.Subject = f"JARVIS Task: {step['description']}"
                appointment.Start = start_time
                appointment.Duration = step["estimated_time"]
                appointment.ReminderSet = True
                appointment.ReminderMinutesBeforeStart = 15
                appointment.Save()
                
                start_time += timedelta(minutes=step["estimated_time"])

    def notify_stakeholders(self, plan: Dict[str, Any]):
        """Send notifications to relevant stakeholders."""
        mail = self.outlook.CreateItem(0)  # 0 = mail item
        mail.Subject = f"Plan Update: {plan['goal']}"
        mail.Body = self._generate_plan_summary(plan)
        mail.To = self.config.get("stakeholder_emails", "")
        mail.Send()

    def _generate_plan_summary(self, plan: Dict[str, Any]) -> str:
        """Generate a summary of the plan for notifications."""
        summary = [
            f"Goal: {plan['goal']}",
            f"Status: {plan['status']}",
            f"Progress: {plan['progress']*100}%",
            "\nSteps:",
        ]
        
        for step in plan["steps"]:
            summary.append(f"- {step['description']}: {step['status']}")
            
        return "\n".join(summary)
