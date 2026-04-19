from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

Base = declarative_base()


class Device(Base):
    """Device SQLAlchemy model - Maps to actual MySQL devices table"""
    __tablename__ = "devices"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Core identification
    brand = Column(String(255), index=True, nullable=True)
    model_name = Column(String(512), index=True, nullable=True)
    model_image = Column(Text, nullable=True)
    
    # Basic specs
    technology = Column(Text, nullable=True)
    announced = Column(Text, nullable=True)
    status = Column(Text, nullable=True)
    dimensions = Column(Text, nullable=True)
    weight = Column(Text, nullable=True)
    build = Column(Text, nullable=True)
    sim = Column(Text, nullable=True)
    keyboard = Column(Text, nullable=True)
    
    # Display
    display_type = Column(Text, nullable=True)
    display_size = Column(Text, nullable=True)
    display_resolution = Column(Text, nullable=True)
    display_protection = Column(Text, nullable=True)
    display_contrast = Column(Text, nullable=True)
    
    # Battery
    battery_capacity = Column(Text, nullable=True)
    battery = Column(Text, nullable=True)
    charging = Column(Text, nullable=True)
    standby = Column(Text, nullable=True)
    talk_time = Column(Text, nullable=True)
    music_play = Column(Text, nullable=True)
    battery_old = Column(Text, nullable=True)
    standby_1 = Column(Text, nullable=True)
    talk_time_1 = Column(Text, nullable=True)
    battery_life = Column(Text, nullable=True)
    
    # Performance
    os = Column(Text, nullable=True)
    chipset = Column(Text, nullable=True)
    cpu = Column(Text, nullable=True)
    gpu = Column(Text, nullable=True)
    performance = Column(Text, nullable=True)
    
    # Storage
    card_slot = Column(Text, nullable=True)
    internal_storage = Column(Text, nullable=True)
    
    # Camera
    main_camera_features = Column(Text, nullable=True)
    main_camera_video = Column(Text, nullable=True)
    selfie_camera_single = Column(Text, nullable=True)
    selfie_features = Column(Text, nullable=True)
    selfie_single_1 = Column(Text, nullable=True)
    selfie_video = Column(Text, nullable=True)
    triple_camera = Column(Text, nullable=True)
    quad_camera = Column(Text, nullable=True)
    dual_camera = Column(Text, nullable=True)
    camera = Column(Text, nullable=True)
    
    # Audio
    loudspeaker = Column(Text, nullable=True)
    loudspeaker_1 = Column(Text, nullable=True)
    loudspeaker_2 = Column(Text, nullable=True)
    loudspeaker_lufs = Column(Text, nullable=True)
    audio_quality = Column(Text, nullable=True)
    jack_35mm = Column(Text, nullable=True)
    alert_types = Column(Text, nullable=True)
    
    # Connectivity
    wlan = Column(Text, nullable=True)
    bluetooth = Column(Text, nullable=True)
    nfc = Column(Text, nullable=True)
    usb = Column(Text, nullable=True)
    positioning = Column(Text, nullable=True)
    radio = Column(Text, nullable=True)
    infrared_port = Column(Text, nullable=True)
    
    # Features & Sensors
    sensors = Column(Text, nullable=True)
    messaging = Column(Text, nullable=True)
    browser = Column(Text, nullable=True)
    clock = Column(Text, nullable=True)
    alarm = Column(Text, nullable=True)
    games = Column(Text, nullable=True)
    languages = Column(Text, nullable=True)
    java = Column(Text, nullable=True)
    colors = Column(Text, nullable=True)
    
    # Telephony/System
    phonebook = Column(Text, nullable=True)
    call_records = Column(Text, nullable=True)
    
    # Pricing & Models
    models = Column(Text, nullable=True)
    price = Column(Text, nullable=True)
    variants = Column(Text, nullable=True)
    
    # Other specs
    speed = Column(Text, nullable=True)
    gprs = Column(Text, nullable=True)
    edge = Column(Text, nullable=True)
    antutu = Column(Text, nullable=True)
    geekbench = Column(Text, nullable=True)
    gfxbench = Column(Text, nullable=True)
    bands_2g = Column(Text, nullable=True)
    bands_3g = Column(Text, nullable=True)
    bands_4g = Column(Text, nullable=True)
    bands_5g = Column(Text, nullable=True)
    
    # Health & Durability
    sar = Column(Text, nullable=True)
    sar_eu = Column(Text, nullable=True)
    energy = Column(Text, nullable=True)
    free_fall = Column(Text, nullable=True)
    repairability = Column(Text, nullable=True)
    
    # Benchmarks
    display = Column(Text, nullable=True)
    
    # Metadata (scraping info)
    # scraped_at and updated_at would go here but not in actual database
    
    # Extra columns (placeholder columns in database)
    unknown_1 = Column(Text, nullable=True)
    unknown_1_1 = Column(Text, nullable=True)
    extra_col = Column(Text, nullable=True)
    extra_col2 = Column(Text, nullable=True)
    extra_col3 = Column(Text, nullable=True)
    extra_col4 = Column(Text, nullable=True)
    extra_col5 = Column(Text, nullable=True)
    extra_col6 = Column(Text, nullable=True)
    extra_col7 = Column(Text, nullable=True)
    extra_col8 = Column(Text, nullable=True)
    extra_col9 = Column(Text, nullable=True)
    extra_col10 = Column(Text, nullable=True)
    extra_col11 = Column(Text, nullable=True)
    extra_col12 = Column(Text, nullable=True)
    extra_col13 = Column(Text, nullable=True)


class DeviceVariant(BaseModel):
    """Device storage/RAM variant"""
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
    brand: str
    model_name: str
    model_image: Optional[str] = None
    device_type: str
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
    scraped_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
