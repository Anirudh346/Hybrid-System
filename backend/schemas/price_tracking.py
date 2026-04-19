from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PriceTrackRequest(BaseModel):
    """Subscribe to price tracking"""
    device_id: str
    target_price: Optional[float] = None  # Alert when price drops below this


class PriceTrackResponse(BaseModel):
    """Price tracking subscription response"""
    success: bool
    message: str


class PriceHistoryResponse(BaseModel):
    """Price history data point"""
    price: float
    currency: str
    source: str
    scraped_at: datetime
    variant_id: Optional[str] = None


class PriceChartResponse(BaseModel):
    """Price chart data for a device"""
    device_id: str
    device_name: str
    current_price: Optional[float] = None
    history: List[PriceHistoryResponse]
    lowest_price: Optional[float] = None
    highest_price: Optional[float] = None
