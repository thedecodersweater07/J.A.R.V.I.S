import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class MLDataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean data for ML"""
        df = data.copy()
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df[col] = self._normalize_column(df[col], col)
            
        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = self._encode_column(df[col], col)
            
        return df
        
    def _normalize_column(self, series: pd.Series, col_name: str) -> pd.Series:
        if col_name not in self.scalers:
            self.scalers[col_name] = StandardScaler()
            return pd.Series(self.scalers[col_name].fit_transform(series.values.reshape(-1, 1)).flatten())
        return pd.Series(self.scalers[col_name].transform(series.values.reshape(-1, 1)).flatten())
        
    def _encode_column(self, series: pd.Series, col_name: str) -> pd.Series:
        if col_name not in self.encoders:
            self.encoders[col_name] = LabelEncoder()
            return pd.Series(self.encoders[col_name].fit_transform(series.fillna('UNKNOWN')))
        return pd.Series(self.encoders[col_name].transform(series.fillna('UNKNOWN')))
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle numerical missing values
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        # Handle categorical missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('UNKNOWN')
        
        return df
