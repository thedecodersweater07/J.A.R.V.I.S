"""
SQL database module for JARVIS.

This module provides database access and management functionality using SQLAlchemy.
"""

from .database_manager import DatabaseManager

# Import models
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

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

__all__ = [
    'DatabaseManager',
    'Base',
    'Conversation',
    'Message',
    'UserSettings',
    'SystemLog'
]
