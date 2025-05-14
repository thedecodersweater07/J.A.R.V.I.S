import logging
import pandas as pd
import numpy as np
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Base class for data validation"""
    
    def __init__(self, name: str):
        """
        Initialize validator
        
        Args:
            name: Name of the validator
        """
        self.name = name
        self.validation_errors: List[str] = []
        
    def validate(self, data: Any) -> bool:
        """
        Validate data
        
        Args:
            data: Data to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        result = self._validate(data)
        if not result:
            logger.warning(f"Validation '{self.name}' failed with {len(self.validation_errors)} errors")
        return result
        
    def _validate(self, data: Any) -> bool:
        """
        Implement validation logic in subclasses
        
        Args:
            data: Data to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        return True
        
    def add_error(self, message: str) -> None:
        """
        Add validation error message
        
        Args:
            message: Error message
        """
        self.validation_errors.append(message)
        logger.debug(f"Validation error in '{self.name}': {message}")
        
    def get_errors(self) -> List[str]:
        """
        Get validation error messages
        
        Returns:
            List of error messages
        """
        return self.validation_errors


class SchemaValidator(DataValidator):
    """Validator for checking data schema"""
    
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        """
        Initialize schema validator
        
        Args:
            schema: Dictionary mapping field names to their specifications
                   Each specification can include:
                   - type: Expected data type
                   - required: Whether the field is required
                   - min/max: Min/max values for numeric fields
                   - allowed_values: Set of allowed values
                   - regex: Regular expression pattern for string fields
        """
        super().__init__("SchemaValidator")
        self.schema = schema
        
    def _validate(self, data: Any) -> bool:
        """
        Validate data against schema
        
        Args:
            data: Data to validate (dict or DataFrame)
            
        Returns:
            True if validation passes, False otherwise
        """
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)
        elif isinstance(data, dict):
            return self._validate_dict(data)
        else:
            self.add_error(f"Unsupported data type: {type(data)}")
            return False
            
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame against schema"""
        is_valid = True
        
        # Check required columns
        required_fields = {field for field, spec in self.schema.items() 
                          if spec.get('required', False)}
        missing_fields = required_fields - set(df.columns)
        if missing_fields:
            for field in missing_fields:
                self.add_error(f"Required column '{field}' is missing")
            is_valid = False
            
        # Check column types and constraints
        for col in df.columns:
            if col not in self.schema:
                continue  # Skip columns not in schema
                
            spec = self.schema[col]
            expected_type = spec.get('type')
            
            # Check type
            if expected_type:
                if expected_type == 'numeric':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        self.add_error(f"Column '{col}' should be numeric")
                        is_valid = False
                elif expected_type == 'string':
                    if not pd.api.types.is_string_dtype(df[col]):
                        self.add_error(f"Column '{col}' should be string type")
                        is_valid = False
                        
            # Check value constraints for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if 'min' in spec and (df[col] < spec['min']).any():
                    self.add_error(f"Column '{col}' has values below minimum {spec['min']}")
                    is_valid = False
                if 'max' in spec and (df[col] > spec['max']).any():
                    self.add_error(f"Column '{col}' has values above maximum {spec['max']}")
                    is_valid = False
                    
            # Check allowed values
            if 'allowed_values' in spec:
                invalid_values = set(df[col].dropna().unique()) - set(spec['allowed_values'])
                if invalid_values:
                    self.add_error(f"Column '{col}' has invalid values: {invalid_values}")
                    is_valid = False
                    
        return is_valid
        
    def _validate_dict(self, data: Dict) -> bool:
        """Validate dictionary against schema"""
        is_valid = True
        
        # Check required fields
        required_fields = {field for field, spec in self.schema.items() 
                          if spec.get('required', False)}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            for field in missing_fields:
                self.add_error(f"Required field '{field}' is missing")
            is_valid = False
            
        # Check field types and constraints
        for field, value in data.items():
            if field not in self.schema:
                continue  # Skip fields not in schema
                
            spec = self.schema[field]
            expected_type = spec.get('type')
            
            # Check type
            if expected_type:
                if expected_type == 'numeric' and not isinstance(value, (int, float)):
                    self.add_error(f"Field '{field}' should be numeric")
                    is_valid = False
                elif expected_type == 'string' and not isinstance(value, str):
                    self.add_error(f"Field '{field}' should be string type")
                    is_valid = False
                    
            # Check value constraints for numeric fields
            if isinstance(value, (int, float)):
                if 'min' in spec and value < spec['min']:
                    self.add_error(f"Field '{field}' value {value} is below minimum {spec['min']}")
                    is_valid = False
                if 'max' in spec and value > spec['max']:
                    self.add_error(f"Field '{field}' value {value} is above maximum {spec['max']}")
                    is_valid = False
                    
            # Check allowed values
            if 'allowed_values' in spec and value not in spec['allowed_values']:
                self.add_error(f"Field '{field}' has invalid value: {value}")
                is_valid = False
                
        return is_valid


class DataIntegrityValidator(DataValidator):
    """Validator for checking data integrity constraints"""
    
    def __init__(self, constraints: List[Dict[str, Any]]):
        """
        Initialize data integrity validator
        
        Args:
            constraints: List of constraint specifications
                Each constraint can include:
                - type: Type of constraint ('unique', 'foreign_key', 'check')
                - columns: Columns involved in the constraint
                - condition: Condition for 'check' constraints
        """
        super().__init__("DataIntegrityValidator")
        self.constraints = constraints
        
    def _validate(self, data: Any) -> bool:
        """
        Validate data integrity
        
        Args:
            data: Data to validate (DataFrame)
            
        Returns:
            True if validation passes, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            self.add_error(f"Data integrity validation requires DataFrame, got {type(data)}")
            return False
            
        is_valid = True
        
        for constraint in self.constraints:
            constraint_type = constraint.get('type')
            columns = constraint.get('columns', [])
            
            # Ensure all columns exist
            if not all(col in data.columns for col in columns):
                missing = [col for col in columns if col not in data.columns]
                self.add_error(f"Columns {missing} not found in data")
                is_valid = False
                continue
                
            # Check constraint based on type
            if constraint_type == 'unique':
                if not data[columns].drop_duplicates().shape[0] == data.shape[0]:
                    self.add_error(f"Uniqueness constraint violated for columns {columns}")
                    is_valid = False
                    
            elif constraint_type == 'not_null':
                for col in columns:
                    if data[col].isnull().any():
                        self.add_error(f"Not-null constraint violated for column {col}")
                        is_valid = False
                        
            elif constraint_type == 'check':
                condition = constraint.get('condition')
                if condition and callable(condition):
                    if not condition(data):
                        self.add_error(f"Check constraint violated: {constraint.get('name', 'unnamed')}")
                        is_valid = False
                        
        return is_valid


class ValidationRegistry:
    """Registry for managing validators"""
    
    def __init__(self):
        """Initialize registry"""
        self.validators: Dict[str, DataValidator] = {}
        
    def register(self, dataset_name: str, validator: DataValidator) -> None:
        """
        Register validator for dataset
        
        Args:
            dataset_name: Name of the dataset
            validator: Validator instance
        """
        self.validators[dataset_name] = validator
        logger.debug(f"Registered validator {validator.name} for dataset {dataset_name}")
        
    def validate(self, dataset_name: str, data: Any) -> Tuple[bool, List[str]]:
        """
        Validate data using registered validator
        
        Args:
            dataset_name: Name of the dataset
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if dataset_name not in self.validators:
            logger.warning(f"No validator registered for dataset {dataset_name}")
            return True, []
            
        validator = self.validators[dataset_name]
        is_valid = validator.validate(data)
        return is_valid, validator.get_errors()
