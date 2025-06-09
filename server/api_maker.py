"""
Enhanced API Key Manager for JARVIS Server
Provides secure API key generation, validation, and management
"""
import secrets
import string
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("jarvis.api_key_manager")

class KeyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_hash: str
    name: str
    server_name: str
    permissions: List[str]
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: int = 1000  # requests per hour
    allowed_ips: Optional[List[str]] = None

class APIKeyManager:
    """
    Enhanced API Key Manager with security features
    - Secure key generation with cryptographic randomness
    - Key rotation and expiration
    - Usage tracking and rate limiting
    - Permission-based access control
    - IP whitelisting support
    """
    
    def __init__(self, server_name: str, storage_path: Optional[str] = None):
        self.server_name = server_name
        self.storage_path = Path(storage_path or "config/api_keys.json")
        self.api_keys: Dict[str, APIKey] = {}
        self.key_index: Dict[str, str] = {}  # hash -> key_id mapping
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing keys
        self._load_keys()
        
        logger.info(f"API Key Manager initialized for server: {server_name}")

    def generate_api_key(self, 
                        name: str,
                        permissions: List[str] = None,
                        expires_hours: Optional[int] = None,
                        rate_limit: int = 1000,
                        allowed_ips: Optional[List[str]] = None,
                        length: int = 64) -> Dict[str, str]:
        """
        Generate a new API key with enhanced security
        
        Args:
            name: Human-readable name for the key
            permissions: List of permissions for this key
            expires_hours: Hours until expiration (None for no expiration)
            rate_limit: Requests per hour limit
            allowed_ips: List of allowed IP addresses
            length: Length of the generated key
            
        Returns:
            Dictionary with key information (includes plaintext key - store securely!)
        """
        if length < 32:
            raise ValueError("API key length must be at least 32 characters")
            
        # Generate cryptographically secure random key
        alphabet = string.ascii_letters + string.digits + '-_'
        raw_key = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # Add prefix for identification
        api_key = f"jarvis_{self.server_name}_{raw_key}"
        
        # Generate unique key ID
        key_id = self._generate_key_id()
        
        # Hash the key for storage (never store plaintext)
        key_hash = self._hash_key(api_key)
        
        # Set expiration if specified
        expires_at = None
        if expires_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        # Create API key object
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            server_name=self.server_name,
            permissions=permissions or ["basic"],
            status=KeyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips
        )
        
        # Store the key
        self.api_keys[key_id] = api_key_obj
        self.key_index[key_hash] = key_id
        
        # Save to persistent storage
        self._save_keys()
        
        logger.info(f"Generated new API key '{name}' with ID: {key_id}")
        
        return {
            "key_id": key_id,
            "api_key": api_key,  # Only return plaintext once!
            "name": name,
            "permissions": permissions or ["basic"],
            "expires_at": expires_at.isoformat() if expires_at else None,
            "rate_limit": rate_limit
        }

    def validate_api_key(self, api_key: str, 
                        required_permission: Optional[str] = None,
                        client_ip: Optional[str] = None) -> Dict[str, any]:
        """
        Validate API key and check permissions
        
        Args:
            api_key: The API key to validate
            required_permission: Required permission for the operation
            client_ip: Client IP address for IP filtering
            
        Returns:
            Validation result with key information or error
        """
        if not api_key or len(api_key) < 32:
            return {"valid": False, "error": "Invalid key format"}
        
        # Hash the provided key
        key_hash = self._hash_key(api_key)
        
        # Find the key
        key_id = self.key_index.get(key_hash)
        if not key_id:
            return {"valid": False, "error": "Key not found"}
        
        api_key_obj = self.api_keys.get(key_id)
        if not api_key_obj:
            return {"valid": False, "error": "Key not found"}
        
        # Check key status
        if api_key_obj.status != KeyStatus.ACTIVE:
            return {"valid": False, "error": f"Key is {api_key_obj.status.value}"}
        
        # Check expiration
        if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
            api_key_obj.status = KeyStatus.EXPIRED
            self._save_keys()
            return {"valid": False, "error": "Key has expired"}
        
        # Check IP restrictions
        if api_key_obj.allowed_ips and client_ip:
            if client_ip not in api_key_obj.allowed_ips:
                return {"valid": False, "error": "IP address not allowed"}
        
        # Check permissions
        if required_permission:
            if required_permission not in api_key_obj.permissions and "admin" not in api_key_obj.permissions:
                return {"valid": False, "error": "Insufficient permissions"}
        
        # Update usage statistics
        api_key_obj.last_used = datetime.utcnow()
        api_key_obj.usage_count += 1
        self._save_keys()
        
        return {
            "valid": True,
            "key_id": key_id,
            "name": api_key_obj.name,
            "permissions": api_key_obj.permissions,
            "rate_limit": api_key_obj.rate_limit,
            "usage_count": api_key_obj.usage_count
        }

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id not in self.api_keys:
            return False
        
        self.api_keys[key_id].status = KeyStatus.REVOKED
        self._save_keys()
        
        logger.info(f"Revoked API key: {key_id}")
        return True

    def list_api_keys(self, include_revoked: bool = False) -> List[Dict[str, any]]:
        """List all API keys (excluding sensitive data)"""
        keys = []
        for key_obj in self.api_keys.values():
            if not include_revoked and key_obj.status == KeyStatus.REVOKED:
                continue
                
            keys.append({
                "key_id": key_obj.key_id,
                "name": key_obj.name,
                "permissions": key_obj.permissions,
                "status": key_obj.status.value,
                "created_at": key_obj.created_at.isoformat(),
                "expires_at": key_obj.expires_at.isoformat() if key_obj.expires_at else None,
                "last_used": key_obj.last_used.isoformat() if key_obj.last_used else None,
                "usage_count": key_obj.usage_count,
                "rate_limit": key_obj.rate_limit
            })
        
        return keys

    def rotate_api_key(self, key_id: str) -> Optional[Dict[str, str]]:
        """
        Rotate an existing API key (generate new key, keep same permissions)
        """
        if key_id not in self.api_keys:
            return None
        
        old_key = self.api_keys[key_id]
        
        # Generate new key with same settings
        new_key_info = self.generate_api_key(
            name=f"{old_key.name} (rotated)",
            permissions=old_key.permissions,
            expires_hours=None if not old_key.expires_at else 
                         int((old_key.expires_at - datetime.utcnow()).total_seconds() / 3600),
            rate_limit=old_key.rate_limit,
            allowed_ips=old_key.allowed_ips
        )
        
        # Revoke old key
        old_key.status = KeyStatus.REVOKED
        self._save_keys()
        
        logger.info(f"Rotated API key: {key_id} -> {new_key_info['key_id']}")
        return new_key_info

    def get_key_stats(self) -> Dict[str, any]:
        """Get statistics about API keys"""
        total = len(self.api_keys)
        active = sum(1 for k in self.api_keys.values() if k.status == KeyStatus.ACTIVE)
        expired = sum(1 for k in self.api_keys.values() if k.status == KeyStatus.EXPIRED)
        revoked = sum(1 for k in self.api_keys.values() if k.status == KeyStatus.REVOKED)
        
        return {
            "total_keys": total,
            "active_keys": active,
            "expired_keys": expired,
            "revoked_keys": revoked,
            "server_name": self.server_name
        }

    def cleanup_expired_keys(self) -> int:
        """Remove expired keys and return count of cleaned up keys"""
        cleaned = 0
        current_time = datetime.utcnow()
        
        for key_obj in self.api_keys.values():
            if (key_obj.expires_at and current_time > key_obj.expires_at and 
                key_obj.status == KeyStatus.ACTIVE):
                key_obj.status = KeyStatus.EXPIRED
                cleaned += 1
        
        if cleaned > 0:
            self._save_keys()
            logger.info(f"Cleaned up {cleaned} expired keys")
        
        return cleaned

    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        timestamp = int(datetime.utcnow().timestamp())
        random_part = secrets.token_hex(8)
        return f"key_{timestamp}_{random_part}"

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _load_keys(self):
        """Load API keys from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                for key_data in data.get('keys', []):
                    # Convert datetime strings back to datetime objects
                    key_data['created_at'] = datetime.fromisoformat(key_data['created_at'])
                    if key_data.get('expires_at'):
                        key_data['expires_at'] = datetime.fromisoformat(key_data['expires_at'])
                    if key_data.get('last_used'):
                        key_data['last_used'] = datetime.fromisoformat(key_data['last_used'])
                    
                    # Convert status string to enum
                    key_data['status'] = KeyStatus(key_data['status'])
                    
                    # Create APIKey object
                    api_key_obj = APIKey(**key_data)
                    self.api_keys[api_key_obj.key_id] = api_key_obj
                    self.key_index[api_key_obj.key_hash] = api_key_obj.key_id
                    
                logger.info(f"Loaded {len(self.api_keys)} API keys from storage")
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")

    def _save_keys(self):
        """Save API keys to storage"""
        try:
            # Convert to serializable format
            keys_data = []
            for key_obj in self.api_keys.values():
                key_dict = asdict(key_obj)
                # Convert datetime objects to ISO format strings
                key_dict['created_at'] = key_obj.created_at.isoformat()
                if key_obj.expires_at:
                    key_dict['expires_at'] = key_obj.expires_at.isoformat()
                if key_obj.last_used:
                    key_dict['last_used'] = key_obj.last_used.isoformat()
                # Convert enum to string
                key_dict['status'] = key_obj.status.value
                keys_data.append(key_dict)
            
            data = {
                'server_name': self.server_name,
                'updated_at': datetime.utcnow().isoformat(),
                'keys': keys_data
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = APIKeyManager("JARVIS_TEST")
    
    # Generate a new API key
    key_info = manager.generate_api_key(
        name="Test Application",
        permissions=["ai_query", "system_status"],
        expires_hours=24,
        rate_limit=500
    )
    
    print(f"Generated API key: {key_info['api_key']}")
    print(f"Key ID: {key_info['key_id']}")
    
    # Validate the key
    validation = manager.validate_api_key(
        key_info['api_key'],
        required_permission="ai_query"
    )
    
    print(f"Validation result: {validation}")
    
    # List all keys
    keys = manager.list_api_keys()
    print(f"All keys: {keys}")
    
    # Get statistics
    stats = manager.get_key_stats()
    print(f"Key statistics: {stats}")