import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import re
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class SecureDataHandler:
    def __init__(self):
        self.sensitive_patterns = [
            r"password",
            r"wachtwoord",
            r"prive",
            r"\b[Pp][Ii][Nn]\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{9}\b",  # BSN
            r"\b(?:\+31|0)\d{9}\b"  # Phone numbers
        ]
        self._init_encryption()
        
    def _init_encryption(self):
        """Initialize encryption key"""
        key_file = Path("security/keys/data.key")
        if key_file.exists():
            with open(key_file, "rb") as f:
                self.key = f.read()
        else:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.key)
        self.fernet = Fernet(self.key)
        
    def sanitize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove sensitive information from data"""
        df = data.copy()
        
        # Replace sensitive patterns with [REDACTED]
        for column in df.select_dtypes(['object']).columns:
            for pattern in self.sensitive_patterns:
                df[column] = df[column].astype(str).apply(
                    lambda x: re.sub(pattern, '[REDACTED]', x, flags=re.IGNORECASE)
                )
                
        return df
        
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode())
        
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        try:
            return self.fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return ""
            
    def is_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive information"""
        return any(
            re.search(pattern, text, re.IGNORECASE) 
            for pattern in self.sensitive_patterns
        )
