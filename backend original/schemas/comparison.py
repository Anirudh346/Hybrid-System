from pydantic import BaseModel
from typing import List, Dict, Any


class ComparisonRequest(BaseModel):
    """Create comparison request"""
    device_ids: List[str]
    name: str = ""


class ComparisonDeviceResponse(BaseModel):
    """Device data for comparison"""
    id: str
    brand: str
    model_name: str
    model_image: str = None
    device_type: str
    specs: Dict[str, Any] = {}


class ComparisonResponse(BaseModel):
    """Comparison response"""
    id: str
    devices: List[ComparisonDeviceResponse]
    name: str
    created_at: str
    
    # Organized comparison data
    comparison_table: Dict[str, List[Any]] = {}


class ComparisonListResponse(BaseModel):
    """List of user comparisons"""
    comparisons: List[ComparisonResponse]
