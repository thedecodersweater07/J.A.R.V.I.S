from models.jarvis import Jarvis
from models.config import JarvisConfig
from typing import Optional, Dict, List, Any


class ConfigModels:
    """
    A unified interface for managing configuration and model operations.
    
    This class provides a single point of access for both configuration
    management and model management operations.
    """
    
    def __init__(self, config_manager: JarvisConfig, model_manager: Jarvis):
        """
        Initialize ConfigModels with managers.
        
        Args:
            config_manager: Instance of ConfigManager for handling configuration
            model_manager: Instance of ModelManager for handling models
        """
        self.config_manager = config_manager
        self.model_manager = model_manager
    
    # Configuration methods
    def get_config(self, key: str) -> Optional[str]:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            
        Returns:
            Configuration value or None if key doesn't exist
        """
        return self.config_manager.get(key)
    
    def set_config(self, key: str, value: str) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key to set
            value: Value to set for the key
        """
        self.config_manager.set(key, value)
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary containing all configuration key-value pairs
        """
        return self.config_manager.get_all() if hasattr(self.config_manager, 'get_all') else {}
    
    def delete_config(self, key: str) -> bool:
        """
        Delete a configuration key.
        
        Args:
            key: Configuration key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        if hasattr(self.config_manager, 'delete'):
            return self.config_manager.delete(key)
        return False
    
    # Model management methods
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model details by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model details dictionary or None if model doesn't exist
        """
        return self.model_manager.get_model(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        return self.model_manager.list_models()
    
    def add_model(self, model_name: str, model_details: Dict[str, Any]) -> bool:
        """
        Add a new model.
        
        Args:
            model_name: Name of the model to add
            model_details: Dictionary containing model configuration
            
        Returns:
            True if model was added successfully, False otherwise
        """
        try:
            self.model_manager.add_model(model_name, model_details)
            return True
        except Exception as e:
            print(f"Error adding model {model_name}: {e}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model by name.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if model was removed successfully, False otherwise
        """
        try:
            self.model_manager.remove_model(model_name)
            return True
        except Exception as e:
            print(f"Error removing model {model_name}: {e}")
            return False
    
    def update_model(self, model_name: str, model_details: Dict[str, Any]) -> bool:
        """
        Update an existing model.
        
        Args:
            model_name: Name of the model to update
            model_details: Dictionary containing updated model configuration
            
        Returns:
            True if model was updated successfully, False otherwise
        """
        try:
            self.model_manager.update_model(model_name, model_details)
            return True
        except Exception as e:
            print(f"Error updating model {model_name}: {e}")
            return False
    
    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        return model_name in self.list_models()
    
    # Utility methods
    def backup_config(self) -> Dict[str, Any]:
        """
        Create a backup of current configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return self.get_all_config()
    
    def restore_config(self, backup: Dict[str, Any]) -> bool:
        """
        Restore configuration from backup.
        
        Args:
            backup: Dictionary containing configuration to restore
            
        Returns:
            True if restore was successful, False otherwise
        """
        try:
            for key, value in backup.items():
                self.set_config(key, str(value))
            return True
        except Exception as e:
            print(f"Error restoring configuration: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of ConfigModels instance."""
        model_count = len(self.list_models())
        return f"ConfigModels(models={model_count}, config_keys={len(self.get_all_config())})"