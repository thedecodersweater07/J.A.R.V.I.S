"""
Data Distributor
===============

Handles the distribution of data across different storage systems.
"""

import os
from typing import Dict, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataDistributor:
    """Distributes data across different storage systems."""
    
    def __init__(self, db_manager):
        """Initialize the data distributor."""
        self.db_manager = db_manager
        self.base_path = db_manager.base_path
        self.db_path = db_manager.db_path
        self.data_path = db_manager.data_path
        
        logger.info("Data Distributor initialized")
    
    def distribute(self, data: Dict[str, Any], data_type: str) -> None:
        """
        Distribute data to appropriate storage locations.
        
        Args:
            data: Data to be distributed
            data_type: Type of data (e.g., 'training', 'processed', 'temp')
        """
        try:
            # Determine storage location based on data type
            storage_path = self._get_storage_path(data_type)
            
            # Store data in appropriate format
            self._store_data(data, storage_path)
            
            # Update database indexes
            self._update_indexes(data, data_type)
            
            logger.info(f"Data distributed successfully to {storage_path}")
        except Exception as e:
            logger.error(f"Error distributing data: {e}")
            raise
    
    def _get_storage_path(self, data_type: str) -> Path:
        """Get appropriate storage path for data type."""
        type_map = {
            'training': self.data_path / 'training',
            'processed': self.data_path / 'processed',
            'temp': self.data_path / 'temp',
            'cache': self.db_path / 'cache'
        }
        
        return type_map.get(data_type, self.data_path / 'temp')
    
    def _store_data(self, data: Dict[str, Any], path: Path) -> None:
        """Store data in appropriate format."""
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data_{timestamp}.json'
        filepath = path / filename
        
        # Store data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data stored at {filepath}")
    
    def _update_indexes(self, data: Dict[str, Any], data_type: str) -> None:
        """Update database indexes with new data location."""
        # TODO: Implement index update logic
        pass
