from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

@dataclass
class User:
    id: str
    username: str
    role: str = "user"
    created_at: datetime = datetime.now()
    last_login: Optional[datetime] = None
    status: str = "active"
    permissions: List[str] = None
    preferences: Dict = None

    def __post_init__(self):
        self.permissions = self.permissions or []
        self.preferences = self.preferences or {}

    @property
    def is_active(self) -> bool:
        return self.status == "active"
