from beanie import Document
from pydantic import Field
from datetime import datetime
from beanie import PydanticObjectId
from typing import Optional


class Favorite(Document):
    """User favorites collection"""
    
    user_id: Optional[PydanticObjectId] = None
    device_id: Optional[PydanticObjectId] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional note from user
    note: Optional[str] = ""
    
    class Settings:
        name = "favorites"
        indexes = [
            [("user_id", 1), ("device_id", 1)],  # Compound unique index
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "device_id": "507f1f77bcf86cd799439012",
                "note": "Great camera!"
            }
        }
