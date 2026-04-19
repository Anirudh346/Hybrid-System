from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import List, Optional
from beanie import PydanticObjectId


class SavedSearch(Document):
    """Saved search filters collection"""
    
    user_id: Optional[PydanticObjectId] = None
    name: str
    
    # Filter criteria
    filters: dict = Field(default_factory=dict)
    # Example: {
    #   "device_type": ["mobile"],
    #   "brand": ["Apple", "Samsung"],
    #   "min_price": 500,
    #   "max_price": 1000,
    #   "processor": "Snapdragon"
    # }
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime = Field(default_factory=datetime.utcnow)
    use_count: int = 0
    
    class Settings:
        name = "saved_searches"
        indexes = [
            "user_id",
            [("user_id", 1), ("created_at", -1)],
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "name": "Gaming Phones Under $800",
                "filters": {
                    "device_type": ["mobile"],
                    "max_price": 800,
                    "processor": "Snapdragon 8 Gen"
                },
                "use_count": 5
            }
        }
