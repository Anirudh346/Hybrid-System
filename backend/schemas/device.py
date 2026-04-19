from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DeviceVariantSchema(BaseModel):
    """Device variant schema"""
    id: str
    label: str
    storage: str
    storage_in_gb: int
    ram: str
    ram_in_gb: int
    price: Optional[str] = None


class DeviceResponse(BaseModel):
    """Device response schema"""
    id: int
    brand: Optional[str] = None
    model_name: Optional[str] = None
    model_image: Optional[str] = None
    technology: Optional[str] = None
    announced: Optional[str] = None
    status: Optional[str] = None
    dimensions: Optional[str] = None
    weight: Optional[str] = None
    build: Optional[str] = None
    sim: Optional[str] = None
    display_type: Optional[str] = None
    display_size: Optional[str] = None
    display_resolution: Optional[str] = None
    os: Optional[str] = None
    chipset: Optional[str] = None
    cpu: Optional[str] = None
    gpu: Optional[str] = None
    internal_storage: Optional[str] = None
    card_slot: Optional[str] = None
    main_camera_features: Optional[str] = None
    main_camera_video: Optional[str] = None
    selfie_camera_single: Optional[str] = None
    loudspeaker: Optional[str] = None
    jack_35mm: Optional[str] = None
    wlan: Optional[str] = None
    bluetooth: Optional[str] = None
    nfc: Optional[str] = None
    usb: Optional[str] = None
    price: Optional[str] = None
    battery_capacity: Optional[str] = None
    charging: Optional[str] = None
    antutu: Optional[str] = None
    geekbench: Optional[str] = None
    speed: Optional[str] = None
    colors: Optional[str] = None
    sensors: Optional[str] = None
    
    class Config:
        from_attributes = True


class DeviceListResponse(BaseModel):
    """Paginated device list response"""
    devices: List[DeviceResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DeviceFilters(BaseModel):
    """Device filter parameters"""
    device_type: Optional[List[str]] = None
    brand: Optional[List[str]] = None
    search: Optional[str] = None
    processor: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class BrandResponse(BaseModel):
    """Brand list response"""
    brands: List[str]


class ChipsetResponse(BaseModel):
    """Chipset list response"""
    chipsets: List[str]


class UploadResponse(BaseModel):
    """CSV upload response"""
    success: bool
    message: str
    devices_count: int
