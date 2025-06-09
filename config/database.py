import os
from pathlib import Path

# Use absolute paths and create directories
ROOT_DIR = Path(__file__).parent.parent
DB_ROOT = ROOT_DIR / "data" / "db"
DB_ROOT.mkdir(parents=True, exist_ok=True)

# Initialize all required database directories
DB_DIRS = ["auth", "knowledge", "memory", "feedback", "cache"]
for dir_name in DB_DIRS:
    (DB_ROOT / dir_name).mkdir(exist_ok=True)

DATABASE_PATHS = {
    "auth": str(DB_ROOT / "auth" / "auth.db"),
    "knowledge": str(DB_ROOT / "knowledge" / "knowledge.db"),
    "memory": str(DB_ROOT / "memory" / "memory.db"),
    "feedback": str(DB_ROOT / "feedback" / "feedback.db"),
    "cache": str(DB_ROOT / "cache" / "cache.db")
}
