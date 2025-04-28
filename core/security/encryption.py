# encryption.py
# Encryption and decryption utilities for the Jarvis security suite

import logging
import base64
import os
import hashlib
import hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import hashes

class EncryptionSystem:
    """Provides encryption and decryption capabilities for Jarvis"""
    
    def __init__(self):
        self.logger = logging.getLogger("EncryptionSystem")
        self.key_store = {}  # For storing generated keys
        self.logger.info("Encryption System initialized")
    
    def generate_symmetric_key(self, key_id, key_size=256):
        """Generate a new symmetric key"""
        if key_size not in [128, 192, 256]:
            self.logger.error(f"Invalid key size: {key_size}")
            return False
            
        try:
            # Generate a random key
            key = os.urandom(key_size // 8)
            
            # Store the key
            self.key_store[key_id] = {
                'type': 'symmetric',
                'key': key,
                'size': key_size
            }
            
            self.logger.info(f"Generated {key_size}-bit symmetric key: {key_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error generating symmetric key: {str(e)}")
            return False
    
    def generate_key_pair(self, key_id, key_size=2048):
        """Generate a new RSA key pair"""
        if key_size not in [1024, 2048, 3072, 4096]:
            self.logger.error(f"Invalid key size: {key_size}")
            return False
            
        try:
            # Generate a new RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()
            
            # Store the key pair
            self.key_store[key_id] = {
                'type': 'asymmetric',
                'private_key': private_key,
                'public_key': public_key,
                'size': key_size
            }
            
            self.logger.info(f"Generated {key_size}-bit RSA key pair: {key_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error generating RSA key pair: {str(e)}")
            return False
    
    def symmetric_encrypt(self, key_id, plaintext, iv=None):
        """Encrypt data using a symmetric key"""
        if key_id not in self.key_store or self.key_store[key_id]['type'] != 'symmetric':
            self.logger.error(f"Symmetric key not found: {key_id}")
            return None
            
        try:
            # Convert plaintext to bytes if it's a string
            if isinstance(plaintext, str):
                plaintext = plaintext.encode('utf-8')
                
            # Get the key
            key = self.key_store[key_id]['key']
            
            # Generate IV if not provided
            if iv is None:
                iv = os.urandom(16)  # 16 bytes for AES
                
            # Create and apply padding
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            
            # Encrypt the data
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Encode as base64 for easy storage/transmission
            result = {
                'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                'iv': base64.b64encode(iv).decode('utf-8')
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return None
    
    def symmetric_decrypt(self, key_id, ciphertext, iv):
        """Decrypt data using a symmetric key"""
        if key_id not in self.key_store or self.key_store[key_id]['type'] != 'symmetric':
            self.logger.error(f"Symmetric key not found: {key_id}")
            return None
            
        try:
            # Decode base64 if inputs are strings
            if isinstance(ciphertext, str):
                ciphertext = base64.b64decode(ciphertext)
            if isinstance(iv, str):
                iv = base64.b64decode(iv)
                
            # Get the key
            key = self.key_store[key_id]['key']
            
            # Decrypt the data
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return plaintext
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            return None
    
    def compute_hmac(self, key_id, data):
        """Compute HMAC for data integrity verification"""
        if key_id not in self.key_store or self.key_store[key_id]['type'] != 'symmetric':
            self.logger.error(f"Symmetric key not found for HMAC: {key_id}")
            return None
            
        try:
            # Convert data to bytes if it's a string
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            # Get the key
            key = self.key_store[key_id]['key']
            
            # Compute HMAC
            h = hmac.new(key, data, hashlib.sha256)
            
            return base64.b64encode(h.digest()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"HMAC computation error: {str(e)}")
            return None