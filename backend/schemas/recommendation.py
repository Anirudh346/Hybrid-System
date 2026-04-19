from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class RecommendationRequest(BaseModel):
    """Recommendation request schema"""
    
    # Natural language query
    query: Optional[str] = None
    
    # Structured preferences
    budget: Optional[float] = None
    device_type: Optional[List[str]] = None
    use_case: Optional[str] = None  # gaming, photography, battery, display, budget, etc.
    brand_preference: Optional[List[str]] = None
    
    # Feature-based filtering
    min_ram_gb: Optional[int] = None
    min_storage_gb: Optional[int] = None
    min_camera_mp: Optional[float] = None
    min_battery_mah: Optional[int] = None
    max_price: Optional[float] = None
    
    # Feature requirements
    require_5g: Optional[bool] = False
    require_nfc: Optional[bool] = False
    require_wireless_charging: Optional[bool] = False
    prefer_fast_charging: Optional[bool] = False
    prefer_expandable_storage: Optional[bool] = False
    
    # Number of recommendations
    top_n: int = 10
    
    # Enable detailed XAI explanations
    explain: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "best gaming phone under $800",
                "budget": 800,
                "device_type": ["mobile"],
                "use_case": "gaming",
                "brand_preference": ["Samsung", "OnePlus"],
                "min_ram_gb": 8,
                "min_camera_mp": 48,
                "min_battery_mah": 5000,
                "require_5g": True,
                "prefer_fast_charging": True,
                "top_n": 10,
                "explain": True
            }
        }


class FeatureContribution(BaseModel):
    """Individual feature contribution to recommendation"""
    feature_name: str
    value: Any
    contribution_score: float
    importance: float
    explanation: str


class DeviceExplanation(BaseModel):
    """Detailed XAI explanation for a recommendation"""
    overall_score: float
    feature_contributions: List[FeatureContribution]
    match_summary: str
    top_reasons: List[str]
    comparable_alternatives: List[Dict[str, Any]]
    confidence: float
    counterfactual: Optional[str] = None


class DeviceRecommendation(BaseModel):
    """Single device recommendation"""
    id: str
    brand: str
    model_name: str
    model_image: Optional[str] = None
    device_type: str
    score: float
    reason: str  # Simple reason (backward compatible)
    specs: Dict[str, Any] = {}
    explanation: Optional[DeviceExplanation] = None  # Detailed XAI explanation


class RecommendationResponse(BaseModel):
    """Recommendation response schema"""
    recommendations: List[DeviceRecommendation]
    parsed_preferences: Dict[str, Any]
    total_candidates: int
