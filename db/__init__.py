"""Database module initialization"""
from datetime import datetime
from typing import Optional, TypeVar, Any
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session as _Session
from pathlib import Path
import os

# Type variable for SQLAlchemy Session
Session = _Session
db_session = TypeVar('db_session', bound=_Session)

# Create a base class for models
Base = declarative_base()

# Define the AIRequestLog model
class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    input_text = Column(String, nullable=False)
    response_text = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection setup
def init_db(db_path: Optional[str] = None) -> Session:
    """Initialize the database connection and create tables
    
    Args:
        db_path: Optional path to the SQLite database file. If None, uses a default path.
        
    Returns:
        Session: A SQLAlchemy session object
    """
    if db_path is None:
        # Default to a database in the project's data directory
        db_path = str(Path(__file__).parent.parent.parent / "data" / "db" / "jarvis.db")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create SQLAlchemy engine
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return SessionLocal()

__all__ = [
    'Base',
    'AIRequestLog',
    'init_db'
]
