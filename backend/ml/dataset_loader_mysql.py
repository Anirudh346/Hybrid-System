"""
Dataset loader for loading phone specifications from MySQL database
Loads and processes device data from MySQL device_catalog.devices table
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger = logging.getLogger(__name__)

# Database Configuration (update with your MySQL credentials)
DATABASE_URL = "mysql+pymysql://root:123@localhost:3306/device_catalog"

Base = declarative_base()


class DeviceORM(Base):
    """ORM Model for devices table"""
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(255), index=True)
    model_name = Column(String(512), index=True)
    model_image = Column(Text, nullable=True)
    device_type = Column(String(50), default='mobile')
    
    # Specifications
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
    
    # Additional specs
    speed = Column(Text, nullable=True)
    gprs = Column(Text, nullable=True)
    edge = Column(Text, nullable=True)
    antutu = Column(Text, nullable=True)
    geekbench = Column(Text, nullable=True)
    colors = Column(Text, nullable=True)
    sensors = Column(Text, nullable=True)
    
    # Metadata
    scraped_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)


class SpecificationExtractor:
    """Extract and normalize phone specifications"""
    
    def extract_numeric(self, value: str) -> float:
        """Extract first numeric value from string"""
        if not value:
            return 0.0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        return float(match.group(1)) if match else 0.0
    
    def extract_ram_gb(self, value: str) -> int:
        """Extract RAM in GB"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+)\s*(?:gb|g)', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_storage_gb(self, value: str) -> int:
        """Extract storage in GB"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        if 'tb' in value_str:
            match = re.search(r'(\d+)\s*tb', value_str)
            if match:
                return int(match.group(1)) * 1024
        
        match = re.search(r'(\d+)\s*gb', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_battery_mah(self, value: str) -> int:
        """Extract battery capacity in mAh"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+)\s*(?:mah|ma h)', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_camera_mp(self, value: str) -> float:
        """Extract max camera megapixels"""
        if not value:
            return 0.0
        
        value_str = str(value).lower()
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*mp', value_str)
        if matches:
            return max(float(m) for m in matches)
        return 0.0
    
    def extract_display_inches(self, value: str) -> float:
        """Extract display size in inches"""
        if not value:
            return 0.0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:inch|")', value_str)
        return float(match.group(1)) if match else 0.0
    
    def extract_refresh_rate(self, value: str) -> int:
        """Extract display refresh rate"""
        if not value:
            return 60
        
        value_str = str(value).lower()
        match = re.search(r'(\d+)\s*hz', value_str)
        return int(match.group(1)) if match else 60
    
    def extract_price(self, value: str) -> float:
        """Extract price from any format"""
        if not value:
            return 0.0
        
        value_str = str(value).strip()
        
        # Handle multi-currency format by splitting on '/'
        if '/' in value_str:
            prices = value_str.split('/')
            
            # Try to find USD price first
            for price_part in prices:
                if '$' in price_part:
                    numeric = re.sub(r'[^\d.,]', '', price_part.strip())
                    numeric = numeric.replace(',', '')
                    if numeric:
                        try:
                            return float(numeric)
                        except ValueError:
                            continue
            
            # Use first available price
            for price_part in prices:
                numeric = re.sub(r'[^\d.,]', '', price_part.strip())
                numeric = numeric.replace(',', '')
                if numeric:
                    try:
                        return float(numeric)
                    except ValueError:
                        continue
        
        # Single price format
        numeric = re.sub(r'[^\d.,]', '', value_str)
        numeric = numeric.replace(',', '')
        
        try:
            return float(numeric) if numeric else 0.0
        except ValueError:
            return 0.0
    
    def has_feature(self, value: str, features: List[str]) -> bool:
        """Check if value contains any of the features"""
        if not value:
            return False
        
        value_str = str(value).lower()
        return any(feature.lower() in value_str for feature in features)
    
    def extract_specs(self, device_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized specs from device dictionary"""
        specs = {}
        
        # Extract numeric specs
        specs['ram_gb'] = self.extract_ram_gb(device_dict.get('internal_storage', ''))
        specs['storage_gb'] = self.extract_storage_gb(device_dict.get('internal_storage', ''))
        specs['battery_mah'] = self.extract_battery_mah(device_dict.get('battery_capacity', ''))
        specs['main_camera_mp'] = self.extract_camera_mp(device_dict.get('main_camera_features', ''))
        specs['selfie_camera_mp'] = self.extract_camera_mp(device_dict.get('selfie_camera_single', ''))
        specs['display_size_inches'] = self.extract_display_inches(device_dict.get('display_size', ''))
        specs['refresh_rate_hz'] = self.extract_refresh_rate(device_dict.get('display_type', ''))
        specs['price'] = self.extract_price(device_dict.get('price', ''))
        specs['weight_g'] = self.extract_numeric(device_dict.get('weight', ''))
        
        # Feature flags
        specs['has_5g'] = self.has_feature(device_dict.get('bands_5g', ''), ['5g', '5', 'sub6', 'mmwave'])
        specs['has_nfc'] = self.has_feature(device_dict.get('nfc', ''), ['yes', 'nfc'])
        specs['has_wireless_charging'] = self.has_feature(device_dict.get('charging', ''), ['wireless', 'inductive'])
        specs['has_fast_charging'] = self.has_feature(device_dict.get('charging', ''), ['fast', 'quick', '25w', '30w', '65w', '120w'])
        specs['has_dual_sim'] = self.has_feature(device_dict.get('sim', ''), ['dual', '2'])
        specs['has_expandable_storage'] = self.has_feature(device_dict.get('card_slot', ''), ['yes', 'microsd', 'expandable'])
        specs['has_jack_35mm'] = self.has_feature(device_dict.get('jack_35mm', ''), ['yes'])
        
        # Text specs
        specs['os'] = str(device_dict.get('os', '')).strip()
        specs['chipset'] = str(device_dict.get('chipset', '')).strip()
        specs['display_type'] = str(device_dict.get('display_type', '')).strip()
        specs['design_material'] = str(device_dict.get('build', '')).strip()
        
        return specs


class PhoneDatasetLoader:
    """Load phone data from MySQL database"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize loader with MySQL database connection
        
        Args:
            database_url: MySQL connection string (defaults to predefined URL)
        """
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.Session = None
        self.extractor = SpecificationExtractor()
        self.devices = []
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"✓ Connected to MySQL database: {self.database_url.split('@')[1]}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to MySQL: {str(e)}")
            raise
    
    def load_all_devices(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all devices from MySQL database
        
        Args:
            limit: Maximum number of devices to load (for testing)
            
        Returns:
            List of device dictionaries with extracted specs
        """
        session = self.Session()
        devices = []
        
        try:
            # Query all devices
            query = session.query(DeviceORM)
            
            if limit:
                query = query.limit(limit)
            
            device_records = query.all()
            logger.info(f"✓ Loaded {len(device_records)} devices from MySQL")
            
            for db_device in device_records:
                # Convert ORM object to dictionary
                device_dict = {
                    'id': db_device.id,
                    'brand': db_device.brand,
                    'model_name': db_device.model_name,
                    'model_image': db_device.model_image,
                    'device_type': db_device.device_type or 'mobile',
                    
                    # Specs
                    'technology': db_device.technology or '',
                    'announced': db_device.announced or '',
                    'status': db_device.status or '',
                    'dimensions': db_device.dimensions or '',
                    'weight': db_device.weight or '',
                    'build': db_device.build or '',
                    'sim': db_device.sim or '',
                    'display_type': db_device.display_type or '',
                    'display_size': db_device.display_size or '',
                    'display_resolution': db_device.display_resolution or '',
                    'os': db_device.os or '',
                    'chipset': db_device.chipset or '',
                    'cpu': db_device.cpu or '',
                    'gpu': db_device.gpu or '',
                    'internal_storage': db_device.internal_storage or '',
                    'card_slot': db_device.card_slot or '',
                    'battery_capacity': db_device.battery_capacity or '',
                    'charging': db_device.charging or '',
                    'price': db_device.price or '',
                    'main_camera_features': db_device.main_camera_features or '',
                    'main_camera_video': db_device.main_camera_video or '',
                    'selfie_camera_single': db_device.selfie_camera_single or '',
                    'loudspeaker': db_device.loudspeaker or '',
                    'jack_35mm': db_device.jack_35mm or '',
                    'wlan': db_device.wlan or '',
                    'bluetooth': db_device.bluetooth or '',
                    'nfc': db_device.nfc or '',
                    'usb': db_device.usb or '',
                    'bands_2g': db_device.bands_2g or '',
                    'bands_3g': db_device.bands_3g or '',
                    'bands_4g': db_device.bands_4g or '',
                    'bands_5g': db_device.bands_5g or '',
                    'speed': db_device.speed or '',
                    'gprs': db_device.gprs or '',
                    'edge': db_device.edge or '',
                    'antutu': db_device.antutu or '',
                    'geekbench': db_device.geekbench or '',
                    'colors': db_device.colors or '',
                    'sensors': db_device.sensors or '',
                }
                
                # Extract specifications
                specs = self.extractor.extract_specs(device_dict)
                
                device = {
                    'id': device_dict['id'],
                    'brand': device_dict['brand'],
                    'model_name': device_dict['model_name'],
                    'model_image': device_dict['model_image'],
                    'device_type': device_dict['device_type'],
                    'specs': specs,
                    'raw_specs': {k: v for k, v in device_dict.items() if v},
                }
                
                devices.append(device)
            
            self.devices = devices
            logger.info(f"✓ Successfully loaded {len(devices)} devices from MySQL with extracted specs")
            return devices
            
        except Exception as e:
            logger.error(f"✗ Error loading devices: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_devices_by_feature(self, feature: str, value_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """Filter devices by specific feature"""
        filtered = []
        
        for device in self.devices:
            specs = device['specs']
            
            if feature not in specs:
                continue
            
            value = specs[feature]
            
            if value_range:
                min_val, max_val = value_range
                if isinstance(value, (int, float)) and min_val <= value <= max_val:
                    filtered.append(device)
            elif value:
                filtered.append(device)
        
        return filtered
    
    def get_devices_by_brand(self, brand: str) -> List[Dict[str, Any]]:
        """Get devices from specific brand"""
        brand_lower = brand.lower()
        return [d for d in self.devices if d['brand'].lower() == brand_lower]
    
    def get_devices_by_price_range(self, min_price: float, max_price: float) -> List[Dict[str, Any]]:
        """Get devices within price range"""
        return self.get_devices_by_feature('price', (min_price, max_price))
    
    def get_devices_by_ram(self, min_ram: int, max_ram: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get devices with minimum RAM"""
        if max_ram is None:
            max_ram = 16
        return self.get_devices_by_feature('ram_gb', (min_ram, max_ram))
    
    def get_flagship_devices(self, top_n: int = 50) -> List[Dict[str, Any]]:
        """Get top flagship devices by score"""
        scored_devices = []
        
        for device in self.devices:
            specs = device['specs']
            
            score = 0.0
            score += specs.get('ram_gb', 0) * 5
            score += specs.get('storage_gb', 0) / 50
            score += specs.get('main_camera_mp', 0)
            score += specs.get('battery_mah', 0) / 1000
            score += specs.get('refresh_rate_hz', 60) / 10
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_budget_devices(self, max_price: float = 500, top_n: int = 50) -> List[Dict[str, Any]]:
        """Get best budget devices"""
        budget_devices = [d for d in self.devices 
                         if 0 < d['specs'].get('price', 0) <= max_price]
        
        scored_devices = []
        
        for device in budget_devices:
            specs = device['specs']
            
            score = 0.0
            if specs.get('price', 0) > 0:
                score += specs.get('ram_gb', 0) * 20 / specs['price']
                score += specs.get('storage_gb', 0) / specs['price']
                score += specs.get('main_camera_mp', 0) * 2 / specs['price']
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_gaming_devices(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get best gaming phones"""
        scored_devices = []
        
        for device in self.devices:
            specs = device['specs']
            
            score = 0.0
            score += specs.get('ram_gb', 0) * 10
            score += specs.get('refresh_rate_hz', 60) * 1.5
            
            chipset = specs.get('chipset', '').lower()
            if any(x in chipset for x in ['snapdragon 8', 'apple a1', 'exynos 2', 'dimensity 9']):
                score += 100
            elif any(x in chipset for x in ['snapdragon 7', 'exynos', 'dimensity 8']):
                score += 50
            
            if specs.get('battery_mah', 0) > 5000:
                score += 20
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.devices:
            return {}
        
        prices = [d['specs'].get('price', 0) for d in self.devices if d['specs'].get('price', 0) > 0]
        rams = [d['specs'].get('ram_gb', 0) for d in self.devices if d['specs'].get('ram_gb', 0) > 0]
        batteries = [d['specs'].get('battery_mah', 0) for d in self.devices if d['specs'].get('battery_mah', 0) > 0]
        
        return {
            'total_devices': len(self.devices),
            'brands': len(set(d['brand'] for d in self.devices)),
            'price': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'avg': sum(prices) / len(prices) if prices else 0,
            },
            'ram': {
                'min': min(rams) if rams else 0,
                'max': max(rams) if rams else 0,
                'avg': sum(rams) / len(rams) if rams else 0,
            },
            'battery': {
                'min': min(batteries) if batteries else 0,
                'max': max(batteries) if batteries else 0,
                'avg': sum(batteries) / len(batteries) if batteries else 0,
            }
        }
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("✓ Database connection closed")


# Global loader instance
dataset_loader = PhoneDatasetLoader()
