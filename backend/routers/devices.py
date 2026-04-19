from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from typing import List, Optional
import re
import io
import csv
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_
from dateutil import parser as date_parser

from schemas.device import (
    DeviceResponse,
    DeviceListResponse,
    DeviceFilters,
    BrandResponse,
    ChipsetResponse,
    UploadResponse,
    DeviceVariantSchema
)
from models.device import Device, DeviceVariant
from database import SessionLocal

# Setup logging
logger = logging.getLogger(__name__)

# Lazy load NLP models to avoid blocking server startup
# from ml.advanced_nlp_parser import advanced_parser, AdvancedNLPParser
# from ml.recommender import DeviceRecommender

router = APIRouter()

# Global variables for lazy-loaded NLP components
_nlp_parser = None
_recommender = None


def get_nlp_parser():
    """Lazy load the NLP parser"""
    global _nlp_parser
    if _nlp_parser is None:
        from ml.advanced_nlp_parser import advanced_parser
        _nlp_parser = advanced_parser
    return _nlp_parser


def get_recommender():
    """Lazy load the recommender"""
    global _recommender
    if _recommender is None:
        from ml.recommender import DeviceRecommender
        _recommender = DeviceRecommender(use_semantic=False)
    return _recommender


def nlp_service_preload():
    """
    Pre-load NLP service on application startup
    This initializes BERT models so first NLP request doesn't have to wait
    """
    from ml.nlp_service import get_nlp_service
    nlp_service = get_nlp_service()
    logger.info("🔄 Pre-loading NLP service...")
    success = nlp_service.initialize()
    if success:
        logger.info("✅ NLP service pre-loaded successfully")
    else:
        logger.warning(f"⚠️  NLP service pre-load skipped: {nlp_service.load_error}")
        logger.info("   NLP will use fallback keyword parsing for initial requests")


def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions to extract numeric specs from string fields
def extract_ram_gb(internal_storage: str) -> int:
    """
    Extract RAM in GB from internal storage field
    Format: "256 GB 12GB RAM" -> RAM = 12
    """
    if not internal_storage:
        return 4
    
    internal_lower = internal_storage.lower()
    
    # Look for pattern: number followed by GB/G and RAM keyword
    # This matches "12GB RAM", "12 GB RAM", "12GB RAM", etc.
    ram_match = re.search(r'(\d+)\s*(?:gb|g)\s+ram', internal_lower)
    if ram_match:
        return int(ram_match.group(1))
    
    # Fallback: look for "RAM: 12GB" format
    ram_match = re.search(r'ram\s*[:\s]+(\d+)\s*(?:gb|g)', internal_lower)
    if ram_match:
        return int(ram_match.group(1))
    
    return 4


def extract_storage_gb(internal_storage: str) -> int:
    """
    Extract storage in GB from internal storage field
    Format: "256 GB 12GB RAM" -> Storage = 256
    """
    if not internal_storage:
        return 64
    
    internal_lower = internal_storage.lower()
    
    # Look for the first number followed by GB that is NOT followed by RAM
    # This handles "256GB 12GB RAM" -> 256 is storage, 12 is RAM
    all_numbers = re.finditer(r'(\d+)\s*(?:gb|g)', internal_lower)
    
    for match in all_numbers:
        num = int(match.group(1))
        # Check if this number is followed by RAM keyword within next 20 chars
        end_pos = match.end()
        following_text = internal_storage[end_pos:end_pos+20].lower()
        
        # If this number is NOT part of RAM specification, it's storage
        if 'ram' not in following_text:
            return num
    
    # Fallback: get the first number that has GB
    match = re.search(r'(\d+)\s*(?:gb|g)', internal_lower)
    if match:
        return int(match.group(1))
    
    return 64


def extract_battery_mah(battery_capacity: str) -> int:
    """
    Extract battery capacity in mAh from battery_capacity field
    Format: "4000 mAh" or "4000mAh" or "Li-Ion 4000 mAh"
    """
    if not battery_capacity:
        return 4000
    
    battery_lower = battery_capacity.lower()
    
    # Look for pattern: number followed by mAh
    mah_match = re.search(r'(\d+)\s*m?ah', battery_lower)
    if mah_match:
        value = int(mah_match.group(1))
        # Filter out unrealistic values (battery should be > 1000 mAh and < 10000 mAh for most phones)
        if 1000 <= value <= 10000:
            return value
    
    return 4000


def extract_camera_mp(camera: str) -> float:
    """Extract primary camera megapixels"""
    if not camera:
        return 12.0
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*MP', camera, re.I)
    if matches:
        return max(float(m) for m in matches)
    return 12.0


def extract_display_inches(display_size: str) -> float:
    """Extract display size in inches"""
    if not display_size:
        return 6.5
    match = re.search(r'(\d+\.\d+)', display_size)
    if match:
        return float(match.group(1))
    return 6.5


def convert_price_to_usd(price_str: str) -> tuple[float, str]:
    """Convert price to USD with approximate exchange rates"""
    if not price_str:
        return 0.0, ""
    
    # Check for informational text that should be preserved
    price_lower = price_str.lower()
    informational_keywords = ['coming soon', 'discontinued', 'not available', 'tba', 'announced']
    if any(keyword in price_lower for keyword in informational_keywords):
        # Preserve original informational text
        return 0.0, price_str
    
    # Approximate exchange rates (as of 2024)
    exchange_rates = {
        'EUR': 1.10,    # 1 EUR = 1.10 USD
        'INR': 0.012,   # 1 INR = 0.012 USD (₹)
        'GBP': 1.27,    # 1 GBP = 1.27 USD (£)
        'CNY': 0.14,    # 1 CNY = 0.14 USD (¥/yuan)
        'JPY': 0.0067,  # 1 JPY = 0.0067 USD (¥/yen)
        'AUD': 0.65,    # 1 AUD = 0.65 USD
        'CAD': 0.74,    # 1 CAD = 0.74 USD
        'RUB': 0.011,   # 1 RUB = 0.011 USD (₽)
        'BRL': 0.20,    # 1 BRL = 0.20 USD (R$)
        'USD': 1.0,     # Base currency
    }
    
    # Extract numeric value
    numeric_match = re.search(r'([\d,]+\.?\d*)', price_str)
    if not numeric_match:
        return 0.0, ""
    
    numeric_str = numeric_match.group(1).replace(',', '')
    try:
        numeric_value = float(numeric_str)
    except:
        return 0.0, ""
    
    # Detect currency and convert
    if '€' in price_str or 'eur' in price_lower:
        usd_value = numeric_value * exchange_rates['EUR']
    elif '₹' in price_str or 'inr' in price_lower or 'rs' in price_lower:
        usd_value = numeric_value * exchange_rates['INR']
    elif '£' in price_str or 'gbp' in price_lower:
        usd_value = numeric_value * exchange_rates['GBP']
    elif '¥' in price_str:
        # Distinguish CNY from JPY by value (JPY values are typically much higher)
        if numeric_value > 10000:
            usd_value = numeric_value * exchange_rates['JPY']
        else:
            usd_value = numeric_value * exchange_rates['CNY']
    elif 'yuan' in price_lower or 'cny' in price_lower:
        usd_value = numeric_value * exchange_rates['CNY']
    elif 'yen' in price_lower or 'jpy' in price_lower:
        usd_value = numeric_value * exchange_rates['JPY']
    elif 'aud' in price_lower or 'a$' in price_str:
        usd_value = numeric_value * exchange_rates['AUD']
    elif 'cad' in price_lower or 'c$' in price_str:
        usd_value = numeric_value * exchange_rates['CAD']
    elif '₽' in price_str or 'rub' in price_lower:
        usd_value = numeric_value * exchange_rates['RUB']
    elif 'r$' in price_lower or 'brl' in price_lower:
        usd_value = numeric_value * exchange_rates['BRL']
    elif '$' in price_str or 'usd' in price_lower:
        usd_value = numeric_value
    else:
        # Default: assume USD
        usd_value = numeric_value
    
    # Filter out invalid prices (less than $1)
    if usd_value < 1.0:
        return 0.0, ""
    
    # Format as USD string
    usd_str = f"${usd_value:,.2f}"
    return usd_value, usd_str


def extract_price_numeric(price: str) -> float:
    """Extract numeric price from price string and convert to USD"""
    usd_value, _ = convert_price_to_usd(price)
    return usd_value


def parse_announced_date(announced: str) -> datetime:
    """Parse announced date string to datetime for sorting"""
    if not announced:
        return datetime(1970, 1, 1)  # Default to epoch for missing dates
    
    try:
        # Handle "2025, May 16" or "2024, January" format
        if ',' in announced:
            announced = announced.strip()
            # Try full parsing
            try:
                return date_parser.parse(announced, fuzzy=True)
            except:
                pass
            
            # Handle "YYYY, QX" format (quarters)
            if 'q' in announced.lower():
                year_part = announced.split(',')[0].strip()
                quarter_match = re.search(r'q(\d)', announced.lower())
                if quarter_match:
                    year = int(year_part)
                    quarter = int(quarter_match.group(1))
                    month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                    return datetime(year, month, 1)
        
        # Try general parsing
        return date_parser.parse(announced, fuzzy=True)
    except:
        # If all parsing fails, return epoch
        return datetime(1970, 1, 1)


def has_5g(bands_5g: str) -> bool:
    """Check if device has 5G support"""
    if not bands_5g:
        return False
    return '5g' in bands_5g.lower() or len(bands_5g.strip()) > 0


def extract_refresh_rate(display_type: str) -> int:
    """Extract display refresh rate"""
    if not display_type:
        return 60
    match = re.search(r'(\d+)\s*Hz', display_type, re.I)
    if match:
        return int(match.group(1))
    return 60


@router.get("", response_model=DeviceListResponse)
async def get_devices(
    device_type: Optional[List[str]] = Query(None),
    brand: Optional[List[str]] = Query(None),
    search: Optional[str] = Query(None),
    processor: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """Get devices with filtering and pagination"""
    
    query = db.query(Device)
    
    # Apply filters
    if device_type:
        query = query.filter(Device.device_type.in_(device_type))
    
    if brand:
        query = query.filter(Device.brand.in_(brand))
    
    if search:
        # Search in model_name or brand (case-insensitive)
        query = query.filter(
            or_(
                Device.model_name.ilike(f"%{search}%"),
                Device.brand.ilike(f"%{search}%")
            )
        )
    
    if processor:
        # Search in chipset or CPU
        query = query.filter(
            or_(
                Device.chipset.ilike(f"%{processor}%"),
                Device.cpu.ilike(f"%{processor}%")
            )
        )
    
    if min_price is not None:
        # For price filtering, we'd need to parse the price field
        # This is simplified - actual implementation would extract numeric price
        pass
    
    if max_price is not None:
        pass
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    skip = (page - 1) * page_size
    devices = query.offset(skip).limit(page_size).all()
    
    # Convert to response format
    device_responses = []
    for device in devices:
        # Convert price to USD
        _, price_usd_str = convert_price_to_usd(device.price or '')
        
        response = DeviceResponse(
            id=device.id,
            brand=device.brand,
            model_name=device.model_name,
            model_image=device.model_image,
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
            price=price_usd_str,  # USD converted price
            battery_capacity=device.battery_capacity,
            charging=device.charging,
            antutu=device.antutu,
            geekbench=device.geekbench,
            speed=device.speed,
            colors=device.colors,
            sensors=device.sensors,
        )
        device_responses.append(response)
    
    total_pages = (total + page_size - 1) // page_size
    
    return DeviceListResponse(
        devices=device_responses,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.get("/brands", response_model=BrandResponse)
async def get_brands(db: Session = Depends(get_db)):
    """Get unique list of all brands"""
    
    brands = db.query(Device.brand).distinct().all()
    brands_list = sorted([b[0] for b in brands if b[0]])
    
    return BrandResponse(brands=brands_list)


@router.get("/chipsets", response_model=ChipsetResponse)
async def get_chipsets():
    """Get predefined list of major chipset manufacturers"""
    
    chipsets = [
        "Qualcomm",
        "MediaTek",
        "Apple",
        "Samsung",
        "HiSilicon",
        "Google",
        "UNISOC",
        "NVIDIA",
        "Intel",
        "Leadcore Technology"
    ]
    
    return ChipsetResponse(chipsets=chipsets)


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str, db: Session = Depends(get_db)):
    """Get single device by ID"""
    
    try:
        device_id_int = int(device_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    device = db.query(Device).filter(Device.id == device_id_int).first()
    
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Convert price to USD
    _, price_usd_str = convert_price_to_usd(device.price or '')
    
    return DeviceResponse(
        id=device.id,
        brand=device.brand,
        model_name=device.model_name,
        model_image=device.model_image,
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
        price=price_usd_str,  # USD converted price
        battery_capacity=device.battery_capacity,
        charging=device.charging,
        antutu=device.antutu,
        geekbench=device.geekbench,
        speed=device.speed,
        colors=device.colors,
        sensors=device.sensors,
    )


@router.post("/search/nlp", response_model=DeviceListResponse)
async def nlp_search(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(50, ge=1, le=10000),
    debug: bool = Query(False, description="Enable debug logging"),
    db: Session = Depends(get_db)
):
    """
    Search devices using NLP-powered natural language queries with intelligent fallback
    
    Examples:
    - "best gaming phone under $500"
    - "affordable Samsung with good battery"
    - "5G phone with 256GB storage"
    
    Query parameters:
    - query: Natural language search query (required)
    - limit: Maximum results to return (default 50)
    - debug: Enable detailed logging (default false)
    """
    
    try:
        # Enable debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"🔍 NLP search query: '{query}'")
        
        # Initialize NLP service if needed
        from ml.nlp_service import get_nlp_service
        nlp_service = get_nlp_service()
        
        if not nlp_service.is_loaded:
            logger.info("🔄 Initializing NLP components on first request...")
            nlp_service.initialize()
        
        # Parse query with fallback to keyword matching
        logger.debug("📝 Parsing query with NLP service...")
        preferences = nlp_service.parse_query(query, use_fallback=True)
        
        if debug:
            logger.debug(f"📋 Parsed preferences: {preferences}")
        
        # Override device_type to always be mobile (we only have mobile devices)
        preferences['device_type'] = ['mobile']
        
        # Load devices from database (filter by availability status)
        logger.debug("📱 Loading devices from database...")
        all_db_devices = db.query(Device).filter(
            (Device.status == None) |
            (Device.status == '') |
            (Device.status.ilike('%available%'))
        ).all()
        
        logger.info(f"📊 Loaded {len(all_db_devices)} available devices from database")
        
        if not all_db_devices:
            logger.warning("⚠️  No available devices found in database")
            return DeviceListResponse(
                devices=[],
                total=0,
                page=1,
                page_size=limit,
                total_pages=0
            )
        
        # Convert devices to dictionaries for recommender
        logger.debug("🔄 Converting devices for recommender...")
        devices_data = []
        device_id_map = {}
        
        for device in all_db_devices:
            ram_gb = extract_ram_gb(device.internal_storage or '')
            storage_gb = extract_storage_gb(device.internal_storage or '')
            battery_mah = extract_battery_mah(device.battery_capacity or '')
            main_camera_mp = extract_camera_mp(device.main_camera_features or '')
            selfie_camera_mp = extract_camera_mp(device.selfie_camera_single or '')
            display_inches = extract_display_inches(device.display_size or '')
            refresh_rate = extract_refresh_rate(device.display_type or '')
            price_numeric, price_usd_str = convert_price_to_usd(device.price or '')
            has_5g_support = has_5g(device.bands_5g or '')
            announced_date = parse_announced_date(device.announced or '')
            
            device_dict = {
                'id': device.id,
                'brand': device.brand or '',
                'model_name': device.model_name or '',
                'price': price_usd_str,  # Now in USD
                'price_original': device.price or '0',  # Keep original for reference
                'announced': device.announced or '',
                'announced_date': announced_date,  # Parsed date for sorting
                'battery_capacity': device.battery_capacity or '',
                'display_size': device.display_size or '',
                'chipset': device.chipset or '',
                'cpu': device.cpu or '',
                'gpu': device.gpu or '',
                'internal_storage': device.internal_storage or '',
                'main_camera_features': device.main_camera_features or '',
                'selfie_camera_single': device.selfie_camera_single or '',
                'os': device.os or '',
                'display_type': device.display_type or '',
                'wlan': device.wlan or '',
                'bluetooth': device.bluetooth or '',
                'nfc': device.nfc or '',
                'bands_2g': device.bands_2g or '',
                'bands_3g': device.bands_3g or '',
                'bands_4g': device.bands_4g or '',
                'bands_5g': device.bands_5g or '',
                'build': device.build or '',
                'sim': device.sim or '',
                'card_slot': device.card_slot or '',
                'charging': device.charging or '',
                'weight': device.weight or '',
                'dimensions': device.dimensions or '',
                'specs': {
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'battery_mah': battery_mah,
                    'main_camera_mp': main_camera_mp,
                    'selfie_camera_mp': selfie_camera_mp,
                    'display_size_inches': display_inches,
                    'refresh_rate_hz': refresh_rate,
                    'price': price_numeric,  # Numeric USD value
                    'has_5g': has_5g_support,
                    'has_nfc': 'yes' in (device.nfc or '').lower(),
                    'has_wireless_charging': 'wireless' in (device.charging or '').lower(),
                    'has_fast_charging': any(x in (device.charging or '').lower() for x in ['fast', 'quick', 'w']),
                    'has_dual_sim': 'dual' in (device.sim or '').lower(),
                    'has_expandable_storage': 'microsd' in (device.card_slot or '').lower() or 'yes' in (device.card_slot or '').lower(),
                    'os': device.os or '',
                    'chipset': device.chipset or '',
                    'display_type': device.display_type or '',
                },
            }
            devices_data.append(device_dict)
            device_id_map[device.id] = device
        
        logger.debug(f"✅ Prepared {len(devices_data)} devices")
        
        # Get recommender and fit with all devices
        logger.debug("🤖 Initializing recommender...")
        try:
            recommender = get_recommender()
            recommender.fit(devices_data)
            
            logger.debug(f"📊 Running recommendations with preferences...")
            recommendations = recommender.recommend_by_preferences(
                preferences=preferences,
                top_n=limit,
                use_mcdm=False
            )
            
            logger.info(f"✅ Recommender returned {len(recommendations)} recommendations")
            
        except Exception as recommender_error:
            logger.error(f"❌ Recommender error: {str(recommender_error)}")
            logger.info("↩️  Falling back to keyword-based brand matching...")
            
            # Fallback: use brand preference if available
            if preferences.get('brand_preference'):
                brand_prefs = [b.lower() for b in preferences.get('brand_preference', [])]
                matching = [
                    (d['id'], 0.5, "Fallback: Brand match")
                    for d in devices_data 
                    if d['brand'].lower() in brand_prefs
                ][:limit]
                recommendations = matching
                logger.info(f"🔄 Fallback found {len(recommendations)} devices by brand")
            else:
                # Last resort: return top devices by popularity/rating
                recommendations = [(d['id'], 0.3, "Fallback: Popular device") for d in devices_data[:limit]]
                logger.info(f"🔄 Fallback returning top {len(recommendations)} popular devices")
        
        # Sort recommendations by announced date (newest first)
        logger.debug("📅 Sorting recommendations by release date...")
        recommendations_with_dates = []
        for device_id, score, explanation in recommendations:
            if isinstance(device_id, str):
                device_id = int(device_id)
            device_dict = next((d for d in devices_data if d['id'] == device_id), None)
            if device_dict:
                recommendations_with_dates.append((
                    device_id, 
                    score, 
                    explanation, 
                    device_dict.get('announced_date', datetime(1970, 1, 1))
                ))
        
        # Sort by date descending (newest first)
        recommendations_with_dates.sort(key=lambda x: x[3], reverse=True)
        logger.info(f"✅ Sorted {len(recommendations_with_dates)} recommendations by date")
        
        # Convert recommendations to DeviceResponse objects
        logger.debug("🔄 Converting recommendations to responses...")
        device_responses = []
        for device_id, score, explanation, announced_date in recommendations_with_dates:
            device = device_id_map.get(device_id)
            if device:
                # Convert price to USD for response
                _, price_usd_str = convert_price_to_usd(device.price or '')
                device_responses.append(DeviceResponse(
                    id=device.id,
                    brand=device.brand,
                    model_name=device.model_name,
                    model_image=device.model_image,
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
                    price=price_usd_str,
                    battery_capacity=device.battery_capacity,
                    charging=device.charging,
                    antutu=device.antutu,
                    geekbench=device.geekbench,
                    speed=device.speed,
                    colors=device.colors,
                    sensors=device.sensors,
                ))
        
        logger.info(f"🎯 Returning {len(device_responses)} device recommendations")
        if debug:
            logger.debug(f"Device IDs returned: {[d.id for d in device_responses]}")
        
        return DeviceListResponse(
            devices=device_responses,
            total=len(device_responses),
            page=1,
            page_size=limit,
            total_pages=1
        )
        
    except Exception as e:
        import traceback
        logger.error(f"💥 Unexpected error in NLP search: {str(e)}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"NLP search error: {str(e)}"
        )
