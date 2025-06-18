"""
Database models and utilities for JARVIS.

This module provides database models and utilities for JARVIS, including
conversation history, user settings, and other persistent data.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pathlib import Path
import os

# Create base class for SQLAlchemy models
Base = declarative_base()

class Conversation(Base):
    """Stores conversation history."""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36), unique=True, nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = Column('metadata', JSON)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """Stores individual messages in a conversation."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36), ForeignKey('conversations.conversation_id'), nullable=False)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column('metadata', JSON)
    
    conversation = relationship("Conversation", back_populates="messages")

class UserSettings(Base):
    """Stores user-specific settings and preferences."""
    __tablename__ = 'user_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), unique=True, nullable=False)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemLog(Base):
    """System logs and events."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False)  # 'info', 'warning', 'error', 'debug'
    source = Column(String(100))
    message = Column(Text, nullable=False)
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<SystemLog(level='{self.level}', source='{self.source}', message='{self.message[:50]}...')>"

class Database:
    """Database connection and session management."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize the database connection.
        
        Args:
            db_url: SQLAlchemy database URL. If not provided, uses SQLite in the data directory.
        """
        if db_url is None:
            # Default to SQLite in the data directory
            db_path = Path(__file__).parent.parent / 'data' / 'db'
            db_path.mkdir(parents=True, exist_ok=True)
            db_url = f'sqlite:///{db_path}/jarvis.db'
        
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close the database connection."""
        self.engine.dispose()

# Singleton database instance
_db_instance = None

def get_database() -> Database:
    """Get the database instance (singleton pattern)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
