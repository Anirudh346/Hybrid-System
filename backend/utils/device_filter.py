"""
Feature-based device filtering and recommendation utilities
Provides high-level functions for filtering devices by specifications
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UseCase(Enum):
    """Predefined use cases for phone recommendations"""
    GAMING = "gaming"
    PHOTOGRAPHY = "photography"
    BATTERY = "battery"
    DISPLAY = "display"
    BUDGET = "budget"
    GENERAL = "general"
    VIDEO = "video"
    PRODUCTIVITY = "productivity"


@dataclass
class SpecRequirements:
    """Device specification requirements"""
    min_ram_gb: Optional[int] = None
    min_storage_gb: Optional[int] = None
    min_camera_mp: Optional[float] = None
    min_battery_mah: Optional[int] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_refresh_rate: Optional[int] = None
    min_weight_g: Optional[float] = None
    max_weight_g: Optional[float] = None
    device_type: Optional[str] = None
    brands_include: Optional[List[str]] = None
    brands_exclude: Optional[List[str]] = None
    require_5g: bool = False
    require_nfc: bool = False
    require_wireless_charging: bool = False
    require_fast_charging: bool = False
    require_expandable_storage: bool = False
    require_3_5mm_jack: bool = False


class DeviceFilter:
    """Advanced device filtering based on specifications"""
    
    @staticmethod
    def filter_by_specs(
        devices: List[Dict[str, Any]],
        requirements: SpecRequirements
    ) -> List[Dict[str, Any]]:
        """
        Filter devices based on specification requirements
        
        Args:
            devices: List of device dictionaries
            requirements: SpecRequirements object
        
        Returns:
            Filtered list of devices
        """
        filtered = devices
        
        # Brand filters
        if requirements.brands_exclude:
            brands_lower = [b.lower() for b in requirements.brands_exclude]
            filtered = [d for d in filtered 
                       if d.get('brand', '').lower() not in brands_lower]
        
        if requirements.brands_include:
            brands_lower = [b.lower() for b in requirements.brands_include]
            filtered = [d for d in filtered 
                       if d.get('brand', '').lower() in brands_lower]
        
        # Device type filter
        if requirements.device_type:
            filtered = [d for d in filtered 
                       if d.get('device_type') == requirements.device_type]
        
        # Numeric spec filters
        for device in filtered[:]:  # Create a copy to iterate
            specs = device.get('specs', {})
            
            if requirements.min_ram_gb and specs.get('ram_gb', 0) < requirements.min_ram_gb:
                filtered.remove(device)
                continue
            
            if requirements.min_storage_gb and specs.get('storage_gb', 0) < requirements.min_storage_gb:
                filtered.remove(device)
                continue
            
            if requirements.min_camera_mp and specs.get('main_camera_mp', 0) < requirements.min_camera_mp:
                filtered.remove(device)
                continue
            
            if requirements.min_battery_mah and specs.get('battery_mah', 0) < requirements.min_battery_mah:
                filtered.remove(device)
                continue
            
            if requirements.min_refresh_rate and specs.get('refresh_rate_hz', 60) < requirements.min_refresh_rate:
                filtered.remove(device)
                continue
            
            if requirements.max_price and specs.get('price', 0) > requirements.max_price > 0:
                filtered.remove(device)
                continue
            
            if requirements.min_price and specs.get('price', 0) > 0 and specs.get('price', 0) < requirements.min_price:
                filtered.remove(device)
                continue
            
            if requirements.min_weight_g and specs.get('weight_g', 0) > 0 and specs.get('weight_g', 0) < requirements.min_weight_g:
                filtered.remove(device)
                continue
            
            if requirements.max_weight_g and specs.get('weight_g', 0) > requirements.max_weight_g:
                filtered.remove(device)
                continue
            
            # Feature flags
            if requirements.require_5g and not specs.get('has_5g', False):
                filtered.remove(device)
                continue
            
            if requirements.require_nfc and not specs.get('has_nfc', False):
                filtered.remove(device)
                continue
            
            if requirements.require_wireless_charging and not specs.get('has_wireless_charging', False):
                filtered.remove(device)
                continue
            
            if requirements.require_fast_charging and not specs.get('has_fast_charging', False):
                filtered.remove(device)
                continue
            
            if requirements.require_expandable_storage and not specs.get('has_expandable_storage', False):
                filtered.remove(device)
                continue
            
            if requirements.require_3_5mm_jack and not specs.get('has_jack_35mm', False):
                filtered.remove(device)
                continue
        
        return filtered
    
    @staticmethod
    def get_predefined_requirements(use_case: UseCase) -> SpecRequirements:
        """Get predefined spec requirements for common use cases"""
        
        if use_case == UseCase.GAMING:
            return SpecRequirements(
                min_ram_gb=8,
                min_refresh_rate=120,
                min_battery_mah=5000,
            )
        
        elif use_case == UseCase.PHOTOGRAPHY:
            return SpecRequirements(
                min_camera_mp=48,
                min_ram_gb=6,
            )
        
        elif use_case == UseCase.BATTERY:
            return SpecRequirements(
                min_battery_mah=5500,
                require_fast_charging=True,
            )
        
        elif use_case == UseCase.DISPLAY:
            return SpecRequirements(
                min_refresh_rate=90,
            )
        
        elif use_case == UseCase.BUDGET:
            return SpecRequirements(
                max_price=500,
            )
        
        elif use_case == UseCase.VIDEO:
            return SpecRequirements(
                min_ram_gb=8,
                min_storage_gb=128,
                min_refresh_rate=60,
            )
        
        elif use_case == UseCase.PRODUCTIVITY:
            return SpecRequirements(
                min_ram_gb=8,
                min_storage_gb=128,
                require_5g=True,
            )
        
        return SpecRequirements()
    
    @staticmethod
    def score_device_for_use_case(
        device: Dict[str, Any],
        use_case: UseCase
    ) -> float:
        """
        Score a device for a specific use case (0-100)
        
        Args:
            device: Device dictionary
            use_case: UseCase enum value
        
        Returns:
            Score from 0 to 100
        """
        specs = device.get('specs', {})
        score = 50.0  # Base score
        
        if use_case == UseCase.GAMING:
            # Weight: RAM (30%), Refresh Rate (30%), Chipset (25%), Battery (15%)
            ram = specs.get('ram_gb', 0)
            score += min(ram / 12.0 * 30, 30)
            
            refresh = specs.get('refresh_rate_hz', 60)
            score += min((refresh - 60) / 60.0 * 30, 30)
            
            chipset = specs.get('chipset', '').lower()
            if any(x in chipset for x in ['snapdragon 8', 'apple a1', 'exynos 2', 'dimensity 9']):
                score += 25
            elif any(x in chipset for x in ['snapdragon 7', 'exynos 1', 'dimensity 8']):
                score += 15
            
            battery = specs.get('battery_mah', 0)
            score += min(battery / 7000.0 * 15, 15)
        
        elif use_case == UseCase.PHOTOGRAPHY:
            # Weight: Main Camera (40%), Secondary features (20%), RAM (20%), Price value (20%)
            main_cam = specs.get('main_camera_mp', 0)
            score += min(main_cam / 200.0 * 40, 40)
            
            selfie_cam = specs.get('selfie_camera_mp', 0)
            score += min(selfie_cam / 50.0 * 10, 10)
            
            if specs.get('has_fast_charging', False):
                score += 10
            
            ram = specs.get('ram_gb', 0)
            score += min(ram / 12.0 * 20, 20)
        
        elif use_case == UseCase.BATTERY:
            # Weight: Battery capacity (50%), Fast charging (30%), Weight (10%), Price (10%)
            battery = specs.get('battery_mah', 0)
            score += min(battery / 7000.0 * 50, 50)
            
            if specs.get('has_fast_charging', False):
                score += 30
            
            weight = specs.get('weight_g', 0)
            if weight > 0 and weight < 200:
                score += 10
            
            price = specs.get('price', 0)
            if price > 0 and price < 800:
                score += 10
        
        elif use_case == UseCase.DISPLAY:
            # Weight: Refresh rate (40%), Display size (30%), Type (20%), Resolution (10%)
            refresh = specs.get('refresh_rate_hz', 60)
            score += min((refresh - 60) / 60.0 * 40, 40)
            
            size = specs.get('display_size_inches', 6.1)
            score += min(abs(size - 6.5) / 2.0 * 30, 30)
            
            display_type = specs.get('display_type', '').lower()
            if any(x in display_type for x in ['amoled', 'oled']):
                score += 20
        
        elif use_case == UseCase.BUDGET:
            # Weight: Value for money (60%), Features (40%)
            price = specs.get('price', 1)
            if price > 0:
                value = (specs.get('ram_gb', 0) * 10 + 
                        specs.get('storage_gb', 0) / 50.0 +
                        specs.get('main_camera_mp', 0)) / price
                score += min(value / 2.0 * 60, 60)
            
            score += min(specs.get('ram_gb', 0) / 6.0 * 15, 15)
            score += min(specs.get('storage_gb', 0) / 128.0 * 15, 15)
            score += min(specs.get('battery_mah', 0) / 5000.0 * 10, 10)
        
        elif use_case == UseCase.VIDEO:
            # Weight: RAM (25%), Storage (25%), Refresh rate (20%), Camera (20%), Battery (10%)
            ram = specs.get('ram_gb', 0)
            score += min(ram / 12.0 * 25, 25)
            
            storage = specs.get('storage_gb', 0)
            score += min(storage / 256.0 * 25, 25)
            
            refresh = specs.get('refresh_rate_hz', 60)
            score += min((refresh - 60) / 60.0 * 20, 20)
            
            camera = specs.get('main_camera_mp', 0)
            score += min(camera / 100.0 * 20, 20)
            
            battery = specs.get('battery_mah', 0)
            score += min(battery / 5000.0 * 10, 10)
        
        elif use_case == UseCase.PRODUCTIVITY:
            # Weight: RAM (30%), Storage (25%), 5G (20%), Display (15%), Battery (10%)
            ram = specs.get('ram_gb', 0)
            score += min(ram / 12.0 * 30, 30)
            
            storage = specs.get('storage_gb', 0)
            score += min(storage / 256.0 * 25, 25)
            
            if specs.get('has_5g', False):
                score += 20
            
            refresh = specs.get('refresh_rate_hz', 60)
            score += min((refresh - 60) / 60.0 * 15, 15)
            
            battery = specs.get('battery_mah', 0)
            score += min(battery / 5000.0 * 10, 10)
        
        return min(max(score, 0), 100)


class ComparisonHelper:
    """Helper functions for device comparison"""
    
    @staticmethod
    def compare_devices(
        devices: List[Dict[str, Any]],
        spec_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a comparison matrix for multiple devices
        
        Args:
            devices: List of devices to compare
            spec_keys: List of spec keys to compare (if None, uses common specs)
        
        Returns:
            Comparison dictionary
        """
        if not spec_keys:
            spec_keys = [
                'ram_gb', 'storage_gb', 'main_camera_mp', 'selfie_camera_mp',
                'battery_mah', 'refresh_rate_hz', 'price', 'weight_g'
            ]
        
        comparison = {
            'devices': [],
            'specs': {}
        }
        
        for device in devices:
            comparison['devices'].append({
                'id': device.get('id'),
                'brand': device.get('brand'),
                'model': device.get('model_name')
            })
        
        specs = [d.get('specs', {}) for d in devices]
        
        for key in spec_keys:
            values = [s.get(key, 0) for s in specs]
            comparison['specs'][key] = {
                'values': values,
                'min': min(v for v in values if v),
                'max': max(v for v in values if v),
                'avg': sum(v for v in values if v) / len([v for v in values if v]) if any(values) else 0
            }
        
        return comparison
    
    @staticmethod
    def get_best_device_for_spec(
        devices: List[Dict[str, Any]],
        spec_key: str
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best device for a specific specification
        
        Args:
            devices: List of devices
            spec_key: Specification key to evaluate
        
        Returns:
            Tuple of (device, value)
        """
        best_device = None
        best_value = -float('inf')
        
        for device in devices:
            value = device.get('specs', {}).get(spec_key, 0)
            if value and value > best_value:
                best_device = device
                best_value = value
        
        return (best_device, best_value) if best_device else (None, 0.0)
