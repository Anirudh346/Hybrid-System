from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from datetime import datetime
from sqlalchemy import or_

from schemas.device import DeviceResponse, DeviceListResponse
from models.device import Device
from database import SessionLocal
from ml.nlp_parser import NLPQueryParser
from pydantic import BaseModel

router = APIRouter()


def _to_device_response(device: Device) -> DeviceResponse:
    return DeviceResponse(
        id=str(device.id),
        brand=device.brand,
        model_name=device.model_name,
        model_image=device.model_image,
        device_type='mobile',
        technology=device.technology,
        announced=device.announced,
        status=device.status,
        dimensions=device.dimensions,
        weight=device.weight,
        build=device.build,
        sim=device.sim,
        display_type=device.display_type,
        display_size=device.display_size,
        display_resolution=device.display_resolution,
        os=device.os,
        chipset=device.chipset,
        cpu=device.cpu,
        gpu=device.gpu,
        internal_storage=device.internal_storage,
        card_slot=device.card_slot,
        main_camera_features=device.main_camera_features,
        main_camera_video=device.main_camera_video,
        selfie_camera_single=device.selfie_camera_single,
        loudspeaker=device.loudspeaker,
        jack_35mm=device.jack_35mm,
        wlan=device.wlan,
        bluetooth=device.bluetooth,
        nfc=device.nfc,
        usb=device.usb,
        price=device.price,
        battery_capacity=device.battery_capacity,
        charging=device.charging,
        antutu=device.antutu,
        geekbench=device.geekbench,
        speed=device.speed,
        colors=device.colors,
        sensors=device.sensors,
        specs={},
        variants=[],
        scraped_at=datetime.now(),
        updated_at=datetime.now(),
    )


@router.get("", response_model=DeviceListResponse)
async def get_devices(
    brand: Optional[List[str]] = Query(None),
    search: Optional[str] = Query(None),
    chipset: Optional[List[str]] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=10000),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get devices with filtering and pagination"""
    
    try:
        session = SessionLocal()
        
        # Build query
        query = session.query(Device)
        
        # Apply filters
        if brand:
            query = query.filter(Device.brand.in_(brand))
        
        if search:
            query = query.filter(
                (Device.brand.ilike(f"%{search}%")) | 
                (Device.model_name.ilike(f"%{search}%"))
            )
        
        # Apply chipset filter - match if any chipset keyword appears in device chipset
        if chipset:
            chip_filters = []
            for chip in chipset:
                chip_filters.append(Device.chipset.ilike(f"%{chip}%"))
            query = query.filter(or_(*chip_filters))
        
        if min_price is not None:
            query = query.filter(Device.price >= min_price)
        
        if max_price is not None:
            query = query.filter(Device.price <= max_price)
        
        # Allow legacy limit param to override page_size
        if limit is not None:
            page_size = limit

        # Get total count
        total = query.count()
        
        # Apply pagination
        skip = (page - 1) * page_size
        devices = query.offset(skip).limit(page_size).all()
        
        session.close()
        
        # Convert to response format
        device_responses = [_to_device_response(device) for device in devices]
        
        total_pages = (total + page_size - 1) // page_size
        
        return DeviceListResponse(
            devices=device_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching devices: {str(e)}")


@router.get("/brands", response_model=List[str])
async def get_brands():
    """Get unique list of all brands"""
    try:
        session = SessionLocal()
        brands = session.query(Device.brand).distinct().all()
        session.close()
        
        brands_list = sorted([b[0] for b in brands if b[0]])
        return brands_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching brands: {str(e)}")


@router.get("/chipsets", response_model=List[str])
async def get_chipsets():
    """Extract unique chipset manufacturers from database"""
    try:
        session = SessionLocal()
        # Get all unique chipsets from database
        chipsets = session.query(Device.chipset).distinct().all()
        session.close()
        
        # Extract manufacturer names from chipset strings
        manufacturers = set()
        chipset_keywords = {
            "Qualcomm": ["Qualcomm", "Snapdragon", "MSM"],
            "MediaTek": ["MediaTek", "Mediatek", "Helio", "Dimensity", "MT"],
            "Apple": ["Apple", "A-series"],
            "Samsung": ["Samsung", "Exynos"],
            "HiSilicon": ["HiSilicon", "Kirin"],
            "Google": ["Google", "Tensor"],
            "UNISOC": ["UNISOC", "Spreadtrum"],
            "NVIDIA": ["NVIDIA", "Tegra"],
            "Intel": ["Intel", "Atom"],
            "Allwinner": ["Allwinner"],
            "Rockchip": ["Rockchip"],
            "Leadcore Technology": ["Leadcore"],
        }
        
        for chipset_tuple in chipsets:
            if not chipset_tuple[0]:
                continue
            chipset_str = chipset_tuple[0].lower()
            
            # Find matching manufacturer
            for manufacturer, keywords in chipset_keywords.items():
                if any(kw.lower() in chipset_str for kw in keywords):
                    manufacturers.add(manufacturer)
                    break
        
        return sorted(list(manufacturers))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chipsets: {str(e)}")


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str):
    """Get single device by ID"""
    
    try:
        session = SessionLocal()
        device = session.query(Device).filter(Device.id == int(device_id)).first()
        session.close()
        
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        return _to_device_response(device)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching device: {str(e)}")


class NLPSearchRequest(BaseModel):
    """NLP search request schema"""
    query: str


@router.post("/search/nlp", response_model=DeviceListResponse)
async def nlp_search(request: NLPSearchRequest, limit: int = Query(50, ge=1, le=10000)):
    """
    Search devices using natural language query (NLP-powered)
    
    Example queries:
    - "best gaming phone under $500"
    - "affordable Samsung phone with good battery"
    - "high-end iPhone alternative"
    - "budget tablet for reading"
    """
    
    try:
        # Parse natural language query
        parser = NLPQueryParser()
        preferences = parser.parse(request.query)
        
        session = SessionLocal()
        query = session.query(Device)
        
        # Apply brand filter if brands were mentioned
        if preferences.get('brand_preference') and len(preferences['brand_preference']) > 0:
            brand_filters = []
            for brand in preferences['brand_preference']:
                brand_filters.append(Device.brand.ilike(f"%{brand}%"))
            query = query.filter(or_(*brand_filters))
        
        # Apply budget filter if budget was mentioned
        if preferences.get('budget'):
            try:
                # Extract numeric price from device.price field
                # This assumes price is stored as a string that starts with a number
                import re as regex_module
                
                # For now, do a simple price filtering - this would need refinement based on your price format
                # query = query.filter(Device.price <= preferences['budget'])
                pass
            except:
                pass
        
        # Apply device type filter if specified
        if preferences.get('device_type') and len(preferences['device_type']) > 0:
            # Note: your Device model doesn't have device_type field, so we skip for now
            # You could infer from model_name or add device_type column
            pass
        
        # Apply use case specific filters
        use_case = preferences.get('use_case', '').lower()
        
        if use_case == 'gaming':
            # Prefer devices with good chipsets for gaming
            query = query.filter(Device.chipset.ilike('%Snapdragon%') | 
                                Device.chipset.ilike('%Dimensity%') |
                                Device.chipset.ilike('%Apple%') |
                                Device.chipset.ilike('%Exynos%'))
        elif use_case == 'photography':
            # Prefer devices with good cameras
            query = query.filter(Device.main_camera_features.isnot(None))
        elif use_case == 'battery':
            # Prefer devices with larger batteries
            query = query.filter(Device.battery_capacity.isnot(None))
        elif use_case == 'display':
            # Prefer devices with good displays (AMOLED, high refresh rate)
            query = query.filter(Device.display_type.ilike('%AMOLED%') | 
                                Device.display_type.ilike('%OLED%') |
                                Device.display_size.isnot(None))
        
        # Get total count and apply limit
        total = query.count()
        devices = query.limit(limit).all()
        
        session.close()
        
        # Convert to response format
        device_responses = [_to_device_response(device) for device in devices]
        
        # Simple pagination for NLP results
        page_size = limit
        total_pages = 1 if total == 0 else (total + page_size - 1) // page_size
        
        return DeviceListResponse(
            devices=device_responses,
            total=total,
            page=1,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in NLP search: {str(e)}")
