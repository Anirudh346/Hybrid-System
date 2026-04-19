from datetime import datetime
from typing import List
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Comparison(Base):
    """Device comparison history (for future use)"""
    __tablename__ = "comparisons"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True)
    device_ids = Column(JSON, default=[])  # Store device IDs as JSON array
    name = Column(String(512), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Comparison(id={self.id}, user_id={self.user_id}, devices={len(self.device_ids)})>"

