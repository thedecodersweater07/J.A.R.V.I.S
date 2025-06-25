import yaml
import json
from typing import Any

def load_yaml(path: str) -> Any:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)
