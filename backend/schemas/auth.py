from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# Request schemas
class UserRegister(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenRefresh(BaseModel):
    """Refresh token request"""
    refresh_token: str


# Response schemas
class Token(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User data response"""
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    preferences: dict = {}
    
    class Config:
        from_attributes = True


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[str] = None
