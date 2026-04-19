from fastapi import APIRouter, HTTPException, Depends
from typing import List
from beanie import PydanticObjectId
from datetime import datetime, timedelta

from schemas.price_tracking import (
    PriceTrackRequest,
    PriceTrackResponse,
    PriceHistoryResponse,
    PriceChartResponse
)
from models.price_history import PriceHistory
from models.device import Device
from models.user import User
from utils.auth import get_current_user

router = APIRouter()


@router.post("/subscribe", response_model=PriceTrackResponse)
async def subscribe_price_tracking(
    request: PriceTrackRequest,
    current_user: User = Depends(get_current_user)
):
    """Subscribe to price tracking for a device"""
    
    try:
        device_id = PydanticObjectId(request.device_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    # Check if device exists
    device = await Device.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # TODO: Store subscription in a new PriceAlert collection
    # For now, return success
    
    return PriceTrackResponse(
        success=True,
        message=f"Successfully subscribed to price tracking for {device.model_name}"
    )


@router.get("/{device_id}/history", response_model=PriceChartResponse)
async def get_price_history(device_id: str, days: int = 30):
    """Get price history for a device"""
    
    try:
        device_obj_id = PydanticObjectId(device_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    # Get device
    device = await Device.get(device_obj_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Get price history from last N days
    since_date = datetime.utcnow() - timedelta(days=days)
    
    price_records = await PriceHistory.find(
        PriceHistory.device_id == device_obj_id,
        PriceHistory.scraped_at >= since_date
    ).sort(PriceHistory.scraped_at).to_list()
    
    history = [
        PriceHistoryResponse(
            price=record.price,
            currency=record.currency,
            source=record.source,
            scraped_at=record.scraped_at,
            variant_id=record.variant_id
        )
        for record in price_records
    ]
    
    # Calculate stats
    current_price = None
    lowest_price = None
    highest_price = None
    
    if price_records:
        prices = [r.price for r in price_records]
        current_price = price_records[-1].price
        lowest_price = min(prices)
        highest_price = max(prices)
    
    return PriceChartResponse(
        device_id=str(device.id),
        device_name=device.model_name,
        current_price=current_price,
        history=history,
        lowest_price=lowest_price,
        highest_price=highest_price
    )


@router.delete("/unsubscribe/{device_id}", response_model=PriceTrackResponse)
async def unsubscribe_price_tracking(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """Unsubscribe from price tracking"""
    
    try:
        device_obj_id = PydanticObjectId(device_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    device = await Device.get(device_obj_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # TODO: Remove subscription from PriceAlert collection
    
    return PriceTrackResponse(
        success=True,
        message=f"Unsubscribed from price tracking for {device.model_name}"
    )
