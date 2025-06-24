"""Request log model for tracking AI interactions"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AIRequestLog(Base):
    """Model for storing AI request logs"""
    __tablename__ = 'ai_request_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    input_text = Column(String, nullable=False)
    response_text = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    extra_data = Column(JSON, default={})  # Renamed from 'metadata' to avoid SQLAlchemy reserved name
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIRequestLog(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"
