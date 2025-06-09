import socket
import secrets
import string

class APIKeyManager:
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.api_keys = {}

    def generate_api_key(self, length=32):
        alphabet = string.ascii_letters + string.digits
        key = ''.join(secrets.choice(alphabet) for _ in range(length))
        self.api_keys[self.server_name] = key
        return key

    def get_api_key(self):
        return self.api_keys.get(self.server_name, None)

    def list_all_keys(self):
        return self.api_keys

# Voorbeeldgebruik
if __name__ == "__main__":
    manager = APIKeyManager("Server_XYZ")
    new_key = manager.generate_api_key()
    print(f"API key for {manager.server_name}: {new_key}")



