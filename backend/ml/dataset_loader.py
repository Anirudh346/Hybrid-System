"""
Dataset loader and feature extractor for phone specifications
Loads and processes GSMArena phone datasets for ML-based recommendations
"""

import pandas as pd
import numpy as np
import re
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpecificationExtractor:
    """Extract and normalize phone specifications from dataset"""
    
    # Mapping of common spec columns to standardized names
    SPEC_MAPPINGS = {
        'Chipset': ['Chipset', 'Processor', 'CPU'],
        'CPU': ['CPU', 'Processor'],
        'GPU': ['GPU', 'Adreno', 'Mali', 'PowerVR'],
        'RAM': ['RAM', 'Memory'],
        'Storage': ['Internal', 'Storage', 'ROM'],
        'Display': ['Display', 'Type', 'Size'],
        'Resolution': ['Resolution', 'Screen Resolution'],
        'Refresh Rate': ['Refresh', 'Refresh Rate', 'Hz'],
        'Battery': ['Battery', 'mAh'],
        'Camera': ['Camera', 'Main Camera', 'Single'],
        'Selfie Camera': ['Selfie', 'Selfie camera', 'Front'],
        'OS': ['OS', 'Operating System'],
        'Weight': ['Weight'],
        'Dimensions': ['Dimensions'],
        'Price': ['Price'],
        'Charging': ['Charging', 'Fast charge'],
        'Network': ['5G', '4G', '3G', 'Technology'],
        'Connectivity': ['Bluetooth', 'NFC', 'WLAN', 'USB'],
    }
    
    def __init__(self):
        self.extracted_features = {}
    
    def extract_numeric(self, value: str) -> float:
        """Extract first numeric value from string"""
        if not value or pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        return float(match.group(1)) if match else 0.0
    
    def extract_ram_gb(self, value: str) -> int:
        """Extract RAM in GB"""
        if not value or pd.isna(value):
            return 0
        
        value_str = str(value).lower()
        # Look for patterns like "8GB", "12 GB", etc.
        match = re.search(r'(\d+)\s*(?:gb|g)', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_storage_gb(self, value: str) -> int:
        """Extract storage in GB"""
        if not value or pd.isna(value):
            return 0
        
        value_str = str(value).lower()
        # Look for patterns like "256GB", "1TB", etc.
        if 'tb' in value_str:
            match = re.search(r'(\d+)\s*tb', value_str)
            if match:
                return int(match.group(1)) * 1024
        
        match = re.search(r'(\d+)\s*gb', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_battery_mah(self, value: str) -> int:
        """Extract battery capacity in mAh"""
        if not value or pd.isna(value):
            return 0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+)\s*(?:mah|ma h)', value_str)
        return int(match.group(1)) if match else 0
    
    def extract_camera_mp(self, value: str) -> float:
        """Extract camera megapixels"""
        if not value or pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*mp', value_str)
        return float(match.group(1)) if match else 0.0
    
    def extract_max_camera_mp(self, value: str) -> float:
        """Extract maximum camera megapixels from string (handles multiple cameras like '13 MP + 5 MP + 2 MP')"""
        if not value or pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        # Find all MP values in the string
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*mp', value_str)
        if matches:
            # Return the maximum MP value
            return max(float(m) for m in matches)
        return 0.0
    
    def extract_camera_mp_from_columns(self, row: Dict[str, Any]) -> float:
        """Extract maximum camera MP from all camera columns (Single, Dual, Triple, Quad)"""
        camera_columns = ['Main Camera', 'Single', 'Dual', 'Triple', 'Quad']
        max_mp = 0.0
        
        for col in camera_columns:
            if col in row:
                mp = self.extract_max_camera_mp(row.get(col, ''))
                max_mp = max(max_mp, mp)
        
        return max_mp
    
    def extract_display_inches(self, value: str) -> float:
        """Extract display size in inches"""
        if not value or pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*inches?', value_str)
        return float(match.group(1)) if match else 0.0
    
    def extract_ram_storage_from_internal(self, internal_str: str) -> Tuple[int, int]:
        """
        Extract RAM and Storage from Internal column
        Handles formats like:
        - "256GB 12GB RAM" -> storage=256GB, ram=12GB
        - "256GB/12GB RAM" -> storage=256GB, ram=12GB
        - "256GB 12GB" -> storage=256GB, ram=12GB (larger is storage)
        - "8GB RAM, 128GB storage" -> ram=8GB, storage=128GB
        
        Args:
            internal_str: String from Internal column
            
        Returns:
            Tuple of (ram_gb, storage_gb)
        """
        if not internal_str or pd.isna(internal_str):
            return 0, 0
        
        value_str = str(internal_str).lower().strip()
        ram_gb = 0
        storage_gb = 0
        
        # Strategy 1: Look for explicit "RAM" keyword
        ram_pattern = r'(\d+)\s*gb\s+ram'
        ram_match = re.search(ram_pattern, value_str)
        if ram_match:
            ram_gb = int(ram_match.group(1))
        
        # Strategy 2: Look for explicit "storage" keyword
        storage_pattern = r'(\d+)\s*(?:gb|tb)\s+(?:storage|internal)'
        storage_match = re.search(storage_pattern, value_str)
        if storage_match:
            storage_str = storage_match.group(1)
            storage_gb = int(storage_str)
            if 'tb' in value_str[storage_match.start():storage_match.end()]:
                storage_gb *= 1024
        
        # Strategy 3: If RAM found via explicit keyword, extract storage as the OTHER number
        if ram_gb > 0 and storage_gb == 0:
            # Find all GB/TB values and extract the one that's not RAM
            all_numbers = re.findall(r'(\d+)\s*(?:gb|tb)', value_str)
            if all_numbers:
                for num_str in all_numbers:
                    num = int(num_str)
                    if num != ram_gb and storage_gb == 0:
                        # Check if it's TB
                        if re.search(rf'{num_str}\s*tb', value_str):
                            storage_gb = num * 1024
                        else:
                            storage_gb = num
                        break
        
        # Strategy 4: If no explicit keywords found, parse numbers intelligently
        if ram_gb == 0 and storage_gb == 0:
            # Find all numbers with GB/TB units
            gb_matches = re.findall(r'(\d+)\s*gb', value_str)
            tb_matches = re.findall(r'(\d+)\s*tb', value_str)
            
            # Handle TB to GB conversion
            if tb_matches:
                storage_gb = int(tb_matches[0]) * 1024
            
            # If we have multiple GB values
            if len(gb_matches) >= 2:
                values = sorted([int(m) for m in gb_matches], reverse=True)
                # Larger value is usually storage, smaller is RAM
                # But if both are large (>24GB), the first is storage, second might also be storage
                if values[0] >= 32:  # Definitely storage
                    storage_gb = values[0]
                    if values[1] <= 24:  # Reasonable RAM value
                        ram_gb = values[1]
                    # If second value is also large, it's likely duplicate or variant info, ignore
                else:
                    storage_gb = values[0]
                    ram_gb = values[1]
            elif len(gb_matches) == 1:
                value = int(gb_matches[0])
                # Single value - typically storage if > 64, RAM if <= 16
                if value <= 16:  # Likely RAM
                    ram_gb = value
                elif value >= 64:  # Likely storage
                    storage_gb = value
                else:  # Ambiguous - assume storage
                    storage_gb = value
        
        # Validation: RAM should never exceed 24GB in real phones
        if ram_gb > 24:
            # If RAM is unrealistic, it's probably storage that was mislabeled
            if storage_gb == 0:
                storage_gb = ram_gb
            ram_gb = 0
        
        # Validation: If both have the same large value, keep only storage
        if ram_gb == storage_gb and ram_gb >= 32:
            ram_gb = 0
        
        return ram_gb, storage_gb
    
    def extract_refresh_rate(self, value: str) -> int:
        """Extract display refresh rate"""
        if not value or pd.isna(value):
            return 60  # Default to 60Hz
        
        value_str = str(value).lower()
        match = re.search(r'(\d+)\s*hz', value_str)
        return int(match.group(1)) if match else 60
    
    def extract_price(self, value: str) -> float:
        """Extract price from multi-currency format (e.g., '₹72,999 / $899.99 / £849.00 / €949.00')
        
        Prefers USD price, fallback to first available price
        """
        if not value or pd.isna(value):
            return 0.0
        
        value_str = str(value).strip()
        
        # Handle multi-currency format by splitting on '/'
        if '/' in value_str:
            prices = value_str.split('/')
            
            # First, try to find USD price ($ symbol)
            for price_part in prices:
                if '$' in price_part:
                    # Extract numeric value from this part
                    numeric = re.sub(r'[^\d.,]', '', price_part.strip())
                    numeric = numeric.replace(',', '')
                    if numeric:
                        try:
                            # Handle decimal prices like 899.99
                            if '.' in numeric:
                                return float(numeric)
                            else:
                                return float(numeric)
                        except ValueError:
                            continue
            
            # If no USD price found, try first available price
            for price_part in prices:
                numeric = re.sub(r'[^\d.,]', '', price_part.strip())
                numeric = numeric.replace(',', '')
                if numeric:
                    try:
                        if '.' in numeric:
                            return float(numeric)
                        else:
                            return float(numeric)
                    except ValueError:
                        continue
            
            return 0.0
        
        # Single price format (no /)
        numeric = re.sub(r'[^\d.,]', '', value_str)
        numeric = numeric.replace(',', '')
        
        try:
            return float(numeric) if numeric else 0.0
        except ValueError:
            return 0.0
    
    def has_feature(self, value: str, features: List[str]) -> bool:
        """Check if value contains any of the features"""
        if not value or pd.isna(value):
            return False
        
        value_str = str(value).lower()
        return any(feature.lower() in value_str for feature in features)
    
    def extract_specs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized specs from a row"""
        specs = {}
        
        # Extract RAM and Storage from Internal column (Samsung format: "256GB 12GB RAM")
        internal_val = row.get('Internal', '') or row.get('internal', '')
        ram_gb = 0
        storage_gb = 0
        
        if internal_val:
            ram_gb, storage_gb = self.extract_ram_storage_from_internal(str(internal_val))
        
        # Fallback: try dedicated RAM and Storage columns if available
        if ram_gb == 0 and row.get('RAM', ''):
            ram_gb = self.extract_ram_gb(row.get('RAM', ''))
        if storage_gb == 0 and row.get('Storage', ''):
            storage_gb = self.extract_storage_gb(row.get('Storage', ''))
        
        specs['ram_gb'] = ram_gb
        specs['storage_gb'] = storage_gb
        
        # Extract battery from Type_1 column (Samsung format), fallback to Battery column
        battery_mah = 0
        type_1_val = row.get('Type_1', '') or row.get('type_1', '')
        if type_1_val:
            battery_mah = self.extract_battery_mah(str(type_1_val))
        
        if battery_mah == 0:
            battery_val = row.get('Battery', '') or row.get('battery', '')
            if battery_val:
                battery_mah = self.extract_battery_mah(str(battery_val))
        
        specs['battery_mah'] = battery_mah
        specs['main_camera_mp'] = self.extract_camera_mp_from_columns(row)
        specs['selfie_camera_mp'] = self.extract_camera_mp(row.get('Selfie camera', row.get('Selfie', '')))
        specs['display_size_inches'] = self.extract_display_inches(row.get('Size', ''))
        specs['refresh_rate_hz'] = self.extract_refresh_rate(row.get('Refresh', row.get('Type_1', '')))
        specs['price'] = self.extract_price(row.get('Price', ''))
        
        # Feature flags
        specs['has_5g'] = self.has_feature(row.get('5G bands', ''), ['5g', '5', 'sub6', 'mmwave'])
        specs['has_nfc'] = self.has_feature(row.get('NFC', ''), ['yes', 'nfc'])
        specs['has_wireless_charging'] = self.has_feature(row.get('Charging', ''), ['wireless', 'inductive'])
        specs['has_fast_charging'] = self.has_feature(row.get('Charging', ''), ['fast', 'quick', '25w', '30w', '65w', '120w'])
        specs['has_dual_sim'] = self.has_feature(row.get('SIM', ''), ['dual', '2'])
        specs['has_expandable_storage'] = self.has_feature(row.get('Card slot', ''), ['yes', 'microsd', 'expandable'])
        specs['has_jack_35mm'] = self.has_feature(row.get('3.5mm jack', ''), ['yes'])
        
        # Text specs
        specs['os'] = str(row.get('OS', '')).strip()
        specs['chipset'] = str(row.get('Chipset', '')).strip()
        specs['display_type'] = str(row.get('Type', '')).strip()
        specs['design_material'] = str(row.get('Build', '')).strip()
        specs['weight_g'] = self.extract_numeric(row.get('Weight', ''))
        
        return specs


class PhoneDatasetLoader:
    """Load and process phone datasets from CSV files"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            dataset_path: Path to GSMArenaDataset folder
        """
        if dataset_path is None:
            # Try to find default dataset path
            base_path = Path(__file__).parent.parent.parent
            dataset_path = str(base_path / "DatasetPhones" / "GSMArenaDataset")
        
        self.dataset_path = Path(dataset_path)
        self.extractor = SpecificationExtractor()
        self.devices = []
        
        if not self.dataset_path.exists():
            logger.warning(f"Dataset path not found: {self.dataset_path}")
    
    def load_csv_files(self, pattern: str = "*.csv", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all CSV files from dataset folder
        
        Args:
            pattern: Glob pattern for files to load
            limit: Maximum number of devices to load (for testing)
        
        Returns:
            List of device dictionaries
        """
        devices = []
        
        if not self.dataset_path.exists():
            logger.error(f"Dataset path not found: {self.dataset_path}")
            return devices
        
        csv_files = sorted(self.dataset_path.glob(pattern))
        
        # Prioritize high-end brands likely to have modern specs
        priority_brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Google', 'Oppo', 
                          'Vivo', 'Realme', 'Motorola', 'Sony', 'Asus']
        priority_files = [f for f in csv_files if any(brand.lower() in f.stem.lower() for brand in priority_brands)]
        other_files = [f for f in csv_files if f not in priority_files]
        csv_files = priority_files + other_files
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        device_count = 0
        
        for csv_file in csv_files:
            logger.debug(f"Loading {csv_file.name}...")
            
            try:
                # Use csv.DictReader for Samsung.csv due to ragged rows (89 header cols, 90-108 data cols)
                # DictReader handles extra columns gracefully with restkey parameter
                if csv_file.name.lower() == 'samsung.csv':
                    logger.debug(f"  Using csv.DictReader for {csv_file.name} (ragged rows)")
                    try:
                        rows_list = []
                        with open(csv_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f, restkey='_extra_fields')
                            rows_list = list(reader)
                        
                        if not rows_list:
                            logger.warning(f"  No rows loaded from {csv_file.name}")
                            continue
                        
                        logger.debug(f"  → Loaded {len(rows_list)} rows from {csv_file.name}")
                        
                        # Convert to DataFrame for consistency with rest of code
                        # Remove the restkey column if it exists
                        for row in rows_list:
                            row.pop('_extra_fields', None)
                        
                        df = pd.DataFrame(rows_list)
                    except Exception as csv_error:
                        logger.error(f"Error using csv.DictReader for {csv_file.name}: {csv_error}")
                        continue
                else:
                    # Use pandas for other files
                    try:
                        df = pd.read_csv(
                            csv_file,
                            encoding='utf-8',
                            dtype=str
                        )
                    except Exception as first_error:
                        # Fall back to python engine for ragged rows
                        logger.debug(f"  C engine failed for {csv_file.name}, trying python engine")
                        try:
                            df = pd.read_csv(
                                csv_file,
                                encoding='utf-8',
                                dtype=str,
                                on_bad_lines='skip',
                                engine='python'
                            )
                        except Exception as python_error:
                            logger.error(f"Error loading {csv_file.name}: {python_error}")
                            continue

                # Fill NaN with empty strings for safe .get() access
                df = df.fillna("")
                
                # Log available columns for debugging
                logger.debug(f"  → Loaded {len(df)} rows with {len(df.columns)} columns")
                
                # Do NOT add empty columns - let actual CSV data flow to extraction functions
                # The extraction functions (extract_specs) will handle missing columns gracefully
                
                devices_from_file = 0
                for _, row in df.iterrows():
                    # Skip rows without required fields (Brand and Model Name must exist)
                    brand = row.get('Brand', '').strip()
                    model_name = row.get('Model Name', '').strip()
                    status = row.get('Status', '').strip()
                    
                    if not brand or not model_name:
                        continue
                    

                    # Include devices with 'available' status or empty/null status
                    # Only skip if explicitly discontinued/cancelled
                    skip_statuses = ['discontinued', 'cancelled']
                    if status and any(skip_status in status.lower() for skip_status in skip_statuses):
                        continue
                    
                    # Extract specifications
                    specs = self.extractor.extract_specs(row.to_dict())
                    
                    # Detect device type
                    model_lower = model_name.lower()
                    if 'watch' in model_lower or 'band' in model_lower:
                        device_type = 'smartwatch'
                    elif 'tablet' in model_lower or 'ipad' in model_lower or 'tab' in model_lower:
                        device_type = 'tablet'
                    else:
                        device_type = 'mobile'
                    
                    device = {
                        'id': f"{brand.lower()}-{model_name.lower().replace(' ', '-')}",
                        'brand': brand,
                        'model_name': model_name,
                        'model_image': row.get('Model Image', ''),
                        'device_type': device_type,
                        'specs': specs,
                        'raw_specs': {k: v for k, v in row.items() 
                                     if pd.notna(v) and str(v).strip()},
                    }
                    
                    devices.append(device)
                    device_count += 1
                    
                    if limit and device_count >= limit:
                        logger.info(f"Reached device limit: {device_count}")
                        return devices
                
                logger.info(f"Loaded {len(devices)} devices total from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {str(e)}")
                continue
        
        self.devices = devices
        logger.info(f"✅ Successfully loaded {len(devices)} devices from {len(csv_files)} CSV files")
        return devices
    
    def get_devices_by_feature(self, feature: str, value_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        Filter devices by specific feature
        
        Args:
            feature: Feature name (e.g., 'ram_gb', 'price', 'battery_mah')
            value_range: Tuple of (min, max) for numeric features
        
        Returns:
            List of matching devices
        """
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
            elif value:  # For boolean or any truthy values
                filtered.append(device)
        
        return filtered
    
    def get_devices_by_price_range(self, min_price: float, max_price: float) -> List[Dict[str, Any]]:
        """Get devices within price range"""
        return self.get_devices_by_feature('price', (min_price, max_price))
    
    def get_devices_by_ram(self, min_ram: int, max_ram: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get devices with minimum RAM (and optional maximum)"""
        if max_ram is None:
            max_ram = 16  # Reasonable upper limit
        return self.get_devices_by_feature('ram_gb', (min_ram, max_ram))
    
    def get_devices_by_brand(self, brand: str) -> List[Dict[str, Any]]:
        """Get devices from specific brand"""
        brand_lower = brand.lower()
        return [d for d in self.devices if d['brand'].lower() == brand_lower]
    
    def get_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        """Get devices of specific type (mobile, tablet, smartwatch)"""
        return [d for d in self.devices if d['device_type'] == device_type]
    
    def get_flagship_devices(self, top_n: int = 50) -> List[Dict[str, Any]]:
        """Get top flagship devices by score (high specs + price)"""
        scored_devices = []
        
        for device in self.devices:
            specs = device['specs']
            
            # Calculate flagship score
            score = 0.0
            score += specs.get('ram_gb', 0) * 5
            score += specs.get('storage_gb', 0) / 50
            score += specs.get('main_camera_mp', 0)
            score += specs.get('battery_mah', 0) / 1000
            score += specs.get('refresh_rate_hz', 60) / 10
            
            scored_devices.append((device, score))
        
        # Sort by score and return top N
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_budget_devices(self, max_price: float = 500, top_n: int = 50) -> List[Dict[str, Any]]:
        """Get best budget devices by score"""
        budget_devices = [d for d in self.devices 
                         if 0 < d['specs'].get('price', 0) <= max_price]
        
        scored_devices = []
        
        for device in budget_devices:
            specs = device['specs']
            
            # Calculate budget score (value for money)
            score = 0.0
            if specs.get('price', 0) > 0:
                score += specs.get('ram_gb', 0) * 20 / specs['price']
                score += specs.get('storage_gb', 0) / specs['price']
                score += specs.get('main_camera_mp', 0) * 2 / specs['price']
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_gaming_devices(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get best gaming phones by performance specs"""
        scored_devices = []
        
        for device in self.devices:
            specs = device['specs']
            
            # Gaming score based on performance specs
            score = 0.0
            score += specs.get('ram_gb', 0) * 10
            score += specs.get('refresh_rate_hz', 60) * 1.5
            
            # Prefer devices with high-end chipsets
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
    
    def get_camera_phones(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get best camera phones"""
        scored_devices = []
        
        for device in self.devices:
            specs = device['specs']
            
            # Camera score
            score = 0.0
            score += specs.get('main_camera_mp', 0) * 0.5
            score += specs.get('selfie_camera_mp', 0) * 0.2
            
            # Bonus for having multiple cameras
            if specs.get('main_camera_mp', 0) > 0:
                score += 20
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_devices[:top_n]]
    
    def get_battery_phones(self, min_battery: int = 5000, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get phones with best battery life"""
        battery_devices = [d for d in self.devices 
                          if d['specs'].get('battery_mah', 0) >= min_battery]
        
        battery_devices.sort(
            key=lambda d: d['specs'].get('battery_mah', 0),
            reverse=True
        )
        
        return battery_devices[:top_n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.devices:
            return {}
        
        prices = [d['specs'].get('price', 0) for d in self.devices if d['specs'].get('price', 0) > 0]
        rams = [d['specs'].get('ram_gb', 0) for d in self.devices if d['specs'].get('ram_gb', 0) > 0]
        batteries = [d['specs'].get('battery_mah', 0) for d in self.devices if d['specs'].get('battery_mah', 0) > 0]
        
        return {
            'total_devices': len(self.devices),
            'by_type': {
                'mobile': len([d for d in self.devices if d['device_type'] == 'mobile']),
                'tablet': len([d for d in self.devices if d['device_type'] == 'tablet']),
                'smartwatch': len([d for d in self.devices if d['device_type'] == 'smartwatch']),
            },
            'brands': len(set(d['brand'] for d in self.devices)),
            'price': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'avg': np.mean(prices) if prices else 0,
            },
            'ram': {
                'min': min(rams) if rams else 0,
                'max': max(rams) if rams else 0,
                'avg': np.mean(rams) if rams else 0,
            },
            'battery': {
                'min': min(batteries) if batteries else 0,
                'max': max(batteries) if batteries else 0,
                'avg': np.mean(batteries) if batteries else 0,
            }
        }


# Global loader instance
dataset_loader = PhoneDatasetLoader()
