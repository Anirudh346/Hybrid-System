from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from config import settings
import re

from schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    DeviceRecommendation,
    DeviceExplanation,
    FeatureContribution
)
from models.device import Device
from ml.simple_recommender import simple_recommender
# Parsers are imported lazily inside handlers to avoid heavy import-time dependencies

# Helper function to extract price from string
def extract_price(price_str: str) -> float:
    """Extract numeric price from string like '1199' or '$1, 199'"""
    if not price_str:
        return 0
    # Remove currency symbols and spaces
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(cleaned) if cleaned else 0
    except ValueError:
        return 0

# Schemas for parse endpoint
class ParseRequest(BaseModel):
    query: str

class ParseResponse(BaseModel):
    parsed_preferences: dict
    query: str

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse_query(request: ParseRequest):
    """
    Parse natural language query into structured preferences.
    
    This endpoint extracts device preferences from natural language without running full recommendations.
    Useful for search bar integration, filters, and query understanding.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # First try lightweight parser (no heavy deps)
        from ml.simple_parser import simple_parser
        preferences = simple_parser.parse(request.query)

        # If advanced parsers are available, enhance the result (optional)
        try:
            from ml.advanced_nlp_parser import AdvancedNLPParser
            from ml.nlp_parser import NLPQueryParser

            adv = AdvancedNLPParser()
            parsed = adv.parse_complex_query(request.query)

            nlp = NLPQueryParser()
            preferences = nlp.enhance_preferences(parsed)
        except Exception:
            # Ignore advanced parser failures; simple parser result is fine
            pass

        return ParseResponse(
            parsed_preferences=preferences,
            query=request.query
        )
    except Exception as e:
        print(f"Error in parse_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing query: {str(e)}")


@router.post("", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get AI-powered device recommendations with Explainable AI (XAI)
    
    Supports:
    - Natural language: "best gaming phone under $800"
    - Complex queries: "cheap flagship for gaming AND photography, not Samsung"
    - Structured preferences: budget, device_type, use_case, brand_preference
    - Detailed explanations: feature contributions, alternatives, confidence scores
    
    The system automatically uses Advanced NLP for complex queries with:
    - Conflicting requirements ("cheap flagship")
    - Multiple use cases ("gaming AND photography")
    - Negations ("not Samsung", "without notch")
    - Trade-offs ("sacrifice camera for battery")
    - Context references ("better than iPhone 12")
    - Implicit preferences ("I travel a lot" → battery + dual SIM)
    """
    
    # Parse natural language query if provided
    preferences = {}
    
    if request.query:
        # Use lightweight parser by default to avoid heavy ML imports at startup
        try:
            from ml.simple_parser import simple_parser
            preferences = simple_parser.parse(request.query)
        except Exception:
            preferences = {'query': request.query}

        # Try to enhance using advanced parsers if available (optional)
        try:
            from ml.advanced_nlp_parser import AdvancedNLPParser
            from ml.nlp_parser import NLPQueryParser

            adv = AdvancedNLPParser()
            parsed = adv.parse_complex_query(request.query)

            nlp = NLPQueryParser()
            preferences = nlp.enhance_preferences(parsed)
        except Exception:
            # keep the simple parser result if advanced parsing fails
            pass
    
    # Override with explicit preferences if provided
    if request.budget is not None:
        preferences['budget'] = request.budget
    
    if request.device_type:
        preferences['device_type'] = request.device_type
    
    if request.use_case:
        preferences['use_case'] = request.use_case
    
    if request.brand_preference:
        preferences['brand_preference'] = request.brand_preference
    
    # Get all devices from MySQL database (device_catalog.devices)
    all_devices = []
    try:
        from database import SessionLocal
        session = SessionLocal()
        all_devices_orm = session.query(Device).all()
        session.close()
        
        # Convert ORM objects to dict format for recommender
        all_devices = [
            {
                'id': str(d.id),
                'brand': d.brand or 'Unknown',
                'model_name': d.model_name or 'Unknown',
                'device_type': 'mobile',  # Default to mobile
                'price': float(extract_price(d.price)) if d.price else 0,
                'model_image': d.model_image,
                'technology': d.technology,
                'status': d.status,
                'os': d.os,
                'chipset': d.chipset,
                'display': d.display_type,
                'battery': d.battery,
                'camera': d.main_camera_features
            }
            for d in all_devices_orm
        ]
    except Exception as e:
        print(f"Error fetching devices from MySQL: {e}")
        return RecommendationResponse(
            recommendations=[],
            parsed_preferences=preferences,
            total_candidates=0
        )
    
    if not all_devices:
        raise HTTPException(status_code=404, detail="No devices found in database")
    
    # Train/update recommender
    simple_recommender.fit(all_devices)
    
    # Get recommendations (cap at 10 for NLP queries)
    effective_top_n = 10 if request.query else request.top_n
    
    # Get recommendations
    recommendations = simple_recommender.recommend_by_preferences(
        preferences,
        top_n=effective_top_n
    )
    
    if not recommendations:
        return RecommendationResponse(
            recommendations=[],
            parsed_preferences=preferences,
            total_candidates=0
        )
    
    # Fetch full device details for recommendations
    device_dict_map = {d['id']: d for d in all_devices}
    
    result_devices = []
    for device_id, score in recommendations:
        device_dict = device_dict_map.get(device_id)
        
        if device_dict:
            # Generate basic recommendation reason
            reason = _generate_reason_dict(device_dict, preferences, score)
            
            # Generate detailed XAI explanation if requested
            explanation = None
            if request.explain:
                # Import the explainer lazily to avoid import-time side-effects
                from ml.xai_explainer import xai_explainer

                xai_result = xai_explainer.explain_recommendation(
                    device=device_dict,
                    preferences=preferences,
                    score=score,
                    all_devices=all_devices
                )
                
                # Convert to Pydantic model
                explanation = DeviceExplanation(
                    overall_score=xai_result.overall_score,
                    feature_contributions=[
                        FeatureContribution(
                            feature_name=fc.feature_name,
                            value=fc.value,
                            contribution_score=fc.contribution_score,
                            importance=fc.importance,
                            explanation=fc.explanation
                        )
                        for fc in xai_result.feature_contributions
                    ],
                    match_summary=xai_result.match_summary,
                    top_reasons=xai_result.top_reasons,
                    comparable_alternatives=xai_result.comparable_alternatives,
                    confidence=xai_result.confidence,
                    counterfactual=xai_result.counterfactual
                )
            
            result_devices.append(
                DeviceRecommendation(
                    id=device_dict['id'],
                    brand=device_dict.get('brand'),
                    model_name=device_dict.get('model_name'),
                    model_image=device_dict.get('model_image'),
                    device_type=device_dict.get('device_type', 'mobile'),
                    score=round(score, 3),
                    reason=reason,
                    specs={},  # No specs in simple schema
                    explanation=explanation
                )
            )
    
    return RecommendationResponse(
        recommendations=result_devices,
        parsed_preferences=preferences,
        total_candidates=len(all_devices)
    )

def _generate_reason_dict(device_dict: dict, preferences: dict, score: float) -> str:
    """Generate human-readable reason for recommendation from dict"""
    
    reasons = []
    
    # Brand match
    if 'brand_preference' in preferences and preferences['brand_preference']:
        if device_dict.get('brand') in preferences['brand_preference']:
            reasons.append(f"Matches your preferred brand ({device_dict.get('brand')})")
    
    # Use case match
    if 'use_case' in preferences and preferences['use_case']:
        use_case = preferences['use_case'].lower()
        
        if use_case == 'gaming':
            chipset = (device_dict.get('chipset') or '').lower()
            if any(kw in chipset for kw in ['snapdragon', 'dimensity', 'a17', 'a18']):
                reasons.append("Powerful processor for gaming")
            else:
                reasons.append("Good gaming potential")
        elif use_case == 'photography':
            camera = device_dict.get('camera') or ''
            if camera:
                reasons.append(f"Photography-focused device")
        elif use_case == 'battery':
            reasons.append("Battery optimized")
    
    # Price match
    if 'budget' in preferences and preferences['budget']:
        price = device_dict.get('price', 0)
        if 0 < price <= preferences['budget']:
            reasons.append(f"Within your budget (${price:.0f})")
    
    # High relevance score
    if score > 0.7:
        reasons.append("Highly relevant to your search")
    elif score > 0.5:
        reasons.append("Good match for your requirements")
    
    if not reasons:
        reasons.append("Recommended based on your preferences")
    
    return ". ".join(reasons) + "."


