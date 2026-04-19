from beanie import Document
from pydantic import EmailStr, Field
from datetime import datetime
from typing import Optional


class User(Document):
    """User collection model"""
    
    email: EmailStr
    password_hash: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # User preferences
    preferences: dict = Field(default_factory=dict)
    
    class Settings:
        name = "users"
        indexes = [
            [("email", 1)],
        ]
        # Beanie will create unique index automatically for email field
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            }
        }
