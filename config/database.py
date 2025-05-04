import os

DB_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db")
os.makedirs(DB_ROOT, exist_ok=True)

DATABASE_PATHS = {
    "knowledge": os.path.join(DB_ROOT, "knowledge.db"),
    "memory": os.path.join(DB_ROOT, "memory.db"),
    "feedback": os.path.join(DB_ROOT, "feedback.db"),
    "cache": os.path.join(DB_ROOT, "cache.db")
}
