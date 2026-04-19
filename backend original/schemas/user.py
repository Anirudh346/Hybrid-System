from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class FavoriteCreate(BaseModel):
    """Create favorite request"""
    device_id: str
    note: Optional[str] = ""


class FavoriteResponse(BaseModel):
    """Favorite response with device details"""
    id: str
    device_id: str
    user_id: str
    added_at: datetime
    note: str
    # Populated device info
    device: Optional[dict] = None
    
    class Config:
        from_attributes = True


class SavedSearchCreate(BaseModel):
    """Create saved search request"""
    name: str
    filters: dict


class SavedSearchResponse(BaseModel):
    """Saved search response"""
    id: str
    user_id: str
    name: str
    filters: dict
    created_at: datetime
    last_used: datetime
    use_count: int
    
    class Config:
        from_attributes = True


class SavedSearchUpdate(BaseModel):
    """Update saved search"""
    name: Optional[str] = None
    filters: Optional[dict] = None
