from typing import Dict, Any
from .base_agent import BaseAgent

class TaskAgent(BaseAgent):
    """Agent that executes specific tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.active_tasks = {}
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task and return the results."""
        task_id = task.get("id")
        self.active_tasks[task_id] = task
        
        try:
            result = self.process_task(task)
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["result"] = result
        except Exception as e:
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
        return self.active_tasks[task_id]
