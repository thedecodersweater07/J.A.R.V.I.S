import logging
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data and return transformed result"""
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the processor"""
        pass


class DataFrameProcessor(DataProcessor):
    """Base class for processors that work with pandas DataFrames"""
    
    def process(self, data: Any) -> pd.DataFrame:
        """Process data ensuring it's a DataFrame"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(data)}")
        return self._process_dataframe(data)
        
    @abstractmethod
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and return transformed DataFrame"""
        pass


class MissingValueHandler(DataFrameProcessor):
    """Handle missing values in DataFrames"""
    
    def __init__(self, strategy: str = 'mean', columns: Optional[List[str]] = None):
        """
        Initialize missing value handler
        
        Args:
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'fill')
            columns: Columns to apply the strategy to (None for all columns)
        """
        self.strategy = strategy
        self.columns = columns
        self._stats = {}  # Store statistics for each column
        
    @property
    def name(self) -> str:
        return f"MissingValueHandler({self.strategy})"
        
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Determine columns to process
        cols = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            if df[col].isna().sum() == 0:
                continue  # No missing values
                
            if self.strategy == 'mean':
                value = df[col].mean()
                result[col] = df[col].fillna(value)
                self._stats[col] = value
            elif self.strategy == 'median':
                value = df[col].median()
                result[col] = df[col].fillna(value)
                self._stats[col] = value
            elif self.strategy == 'mode':
                value = df[col].mode()[0]
                result[col] = df[col].fillna(value)
                self._stats[col] = value
            elif self.strategy == 'drop':
                # Only drop rows where specified columns have NaN
                result = result.dropna(subset=[col])
            elif self.strategy == 'fill':
                # Fill with a specified value (0 by default)
                value = 0
                result[col] = df[col].fillna(value)
                self._stats[col] = value
                
        return result


class Normalizer(DataFrameProcessor):
    """Normalize numerical columns in DataFrames"""
    
    def __init__(self, method: str = 'minmax', columns: Optional[List[str]] = None):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('minmax', 'zscore')
            columns: Columns to normalize (None for all numerical columns)
        """
        self.method = method
        self.columns = columns
        self._stats = {}  # Store statistics for each column
        
    @property
    def name(self) -> str:
        return f"Normalizer({self.method})"
        
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns"""
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Determine columns to process
        cols = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            if self.method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:  # Avoid division by zero
                    result[col] = (df[col] - min_val) / (max_val - min_val)
                    self._stats[col] = {'min': min_val, 'max': max_val}
            elif self.method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    result[col] = (df[col] - mean) / std
                    self._stats[col] = {'mean': mean, 'std': std}
                    
        return result


class DataPipeline:
    """Pipeline for processing data through multiple processors"""
    
    def __init__(self, name: str = "DataPipeline"):
        """
        Initialize data pipeline
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.processors: List[DataProcessor] = []
        
    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """
        Add processor to pipeline
        
        Args:
            processor: DataProcessor instance
            
        Returns:
            Self for method chaining
        """
        self.processors.append(processor)
        logger.debug(f"Added processor {processor.name} to pipeline {self.name}")
        return self
        
    def process(self, data: Any) -> Any:
        """
        Process data through all processors in the pipeline
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        result = data
        for processor in self.processors:
            try:
                logger.debug(f"Applying processor {processor.name}")
                result = processor.process(result)
            except Exception as e:
                logger.error(f"Error in processor {processor.name}: {str(e)}")
                raise
                
        return result
        
    def __str__(self) -> str:
        """String representation of the pipeline"""
        return f"DataPipeline({self.name}) with {len(self.processors)} processors"
