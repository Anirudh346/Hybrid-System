from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import List
from beanie import PydanticObjectId


class Comparison(Document):
    """Device comparison history collection"""
    
    user_id: PydanticObjectId
    device_ids: List[PydanticObjectId] = Field(default_factory=list)
    
    # Optional comparison name
    name: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "comparisons"
        indexes = [
            "user_id",
            [("user_id", 1), ("created_at", -1)],
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "device_ids": [
                    "507f1f77bcf86cd799439012",
                    "507f1f77bcf86cd799439013",
                    "507f1f77bcf86cd799439014"
                ],
                "name": "Flagship Comparison 2024"
            }
        }
