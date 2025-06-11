#als van llm hoe het werkt en geconfigureerd is en hoe je het kan gebruiken
from llm import LLM
from llm.manager import LLMManager
from llm.config import LLMConfig

class LLMService:
    """Service class to manage LLM operations"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = LLM(config)
        self.manager = LLMManager()
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM with the provided configuration"""
        try:
            self.llm.initialize()
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")

    def process_input(self, user_input: str, user_id: str) -> str:
        """Process user input and return LLM response"""
        return self.manager.process_input(user_input, user_id)
    def save_training_data(self, data: str):
        """Save training data to the LLM"""
        self.manager.save_training_data(data)
        print("Training data saved successfully")
    def get_config(self) -> LLMConfig:
        """Get the current LLM configuration"""
        return self.config
    def update_config(self, new_config: LLMConfig):
        """Update the LLM configuration"""
        self.config = new_config
        self.llm.update_config(new_config)
        print("LLM configuration updated successfully")
    def get_status(self) -> str:
        """Get the current status of the LLM"""
        return self.llm.get_status()

class LLMServiceManager:
    """Manager class to handle multiple LLMService instances"""
    
    def __init__(self):
        self.services = {}

    def add_service(self, service_name: str, config: LLMConfig):
        """Add a new LLM service"""
        if service_name in self.services:
            raise ValueError(f"Service {service_name} already exists")
        self.services[service_name] = LLMService(config)
        print(f"LLM service '{service_name}' added successfully")

    def get_service(self, service_name: str) -> LLMService:
        """Get an existing LLM service"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} does not exist")
        return self.services[service_name]
    
    def remove_service(self, service_name: str):
        """Remove an existing LLM service"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} does not exist")
        del self.services[service_name]
        print(f"LLM service '{service_name}' removed successfully")
    def list_services(self) -> list:
        """List all available LLM services"""
        return list(self.services.keys())
    
class LLMServiceFactory:
    """Factory class to create LLMService instances"""
    
    @staticmethod
    def create_service(config: LLMConfig) -> LLMService:
        """Create a new LLMService instance"""
        return LLMService(config)
    
    @staticmethod
    def create_manager() -> LLMServiceManager:
        """Create a new LLMServiceManager instance"""
        return LLMServiceManager()
