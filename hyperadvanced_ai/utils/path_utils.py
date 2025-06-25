import os
from typing import Optional

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def get_basename(path: str) -> str:
    return os.path.basename(path)

def get_extension(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    return ext if ext else None
