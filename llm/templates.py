from typing import Dict, List

class PromptTemplate:
    def __init__(self):
        self.templates = {
            "chat": "{context}\nUser: {user_input}\nAssistant:",
            "system": "You are an AI assistant named Jarvis. {instructions}",
            "command": "Execute the following command: {command}"
        }
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        return template.format(**kwargs)
