from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional
from beanie import PydanticObjectId


class PriceHistory(Document):
    """Price tracking history collection"""
    
    device_id: PydanticObjectId
    price: float
    currency: str = "EUR"
    source: str  # "amazon", "flipkart", "gsmarena"
    url: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional metadata
    variant_id: Optional[str] = None  # Which variant this price is for
    availability: bool = True
    
    class Settings:
        name = "price_history"
        indexes = [
            "device_id",
            [("device_id", 1), ("scraped_at", -1)],  # For time-series queries
            "scraped_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "507f1f77bcf86cd799439012",
                "price": 1199.99,
                "currency": "EUR",
                "source": "amazon",
                "url": "https://amazon.in/product/...",
                "variant_id": "256GB-8GB"
            }
        }
