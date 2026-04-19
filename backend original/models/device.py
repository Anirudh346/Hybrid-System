from sqlalchemy import Column, Integer, String, Text
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from models.base import Base


class Device(Base):
    """SQLAlchemy Device model - mapped to actual device_catalog.devices table (103 columns)"""
    __tablename__ = "devices"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Core device fields (longtext columns)
    brand = Column(Text, nullable=True, index=True)
    model_name = Column(Text, nullable=True)
    model_image = Column(Text, nullable=True)
    technology = Column(Text, nullable=True)
    announced = Column(Text, nullable=True)
    status = Column(Text, nullable=True)
    dimensions = Column(Text, nullable=True)
    weight = Column(Text, nullable=True)
    build = Column(Text, nullable=True)
    sim = Column(Text, nullable=True)
    display_type = Column(Text, nullable=True)
    display_size = Column(Text, nullable=True)
    display_resolution = Column(Text, nullable=True)
    os = Column(Text, nullable=True)
    chipset = Column(Text, nullable=True)
    cpu = Column(Text, nullable=True)
    gpu = Column(Text, nullable=True)
    internal_storage = Column(Text, nullable=True)
    card_slot = Column(Text, nullable=True)
    battery_capacity = Column(Text, nullable=True)
    charging = Column(Text, nullable=True)
    price = Column(Text, nullable=True)
    main_camera_features = Column(Text, nullable=True)
    main_camera_video = Column(Text, nullable=True)
    selfie_camera_single = Column(Text, nullable=True)
    loudspeaker = Column(Text, nullable=True)
    jack_35mm = Column(Text, nullable=True)
    wlan = Column(Text, nullable=True)
    bluetooth = Column(Text, nullable=True)
    nfc = Column(Text, nullable=True)
    usb = Column(Text, nullable=True)
    
    # Network bands
    bands_2g = Column(Text, nullable=True)
    bands_3g = Column(Text, nullable=True)
    bands_4g = Column(Text, nullable=True)
    bands_5g = Column(Text, nullable=True)
    
    # Additional common columns
    speed = Column(Text, nullable=True)
    gprs = Column(Text, nullable=True)
    edge = Column(Text, nullable=True)
    antutu = Column(Text, nullable=True)
    geekbench = Column(Text, nullable=True)
    colors = Column(Text, nullable=True)
    sensors = Column(Text, nullable=True)


class DeviceResponse(BaseModel):
    """Pydantic schema for Device API responses"""
    id: str
    brand: Optional[str] = None
    model_name: Optional[str] = None
    model_image: Optional[str] = None
    device_type: str = 'mobile'
    specs: dict = {}
    variants: list = []
    scraped_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
