from pydantic import BaseModel, Field

class RouteConfig(BaseModel):
    path: str
    pattern: str = Field(default=".*")  # Using pattern instead of regex
    methods: list[str] = ["GET"]

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    routes: list[RouteConfig] = []
