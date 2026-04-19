from fastapi import APIRouter, HTTPException
from typing import List

from schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    DeviceRecommendation,
    DeviceExplanation,
    FeatureContribution
)
from models.device import Device
from ml.recommender import recommender
from ml.nlp_parser import nlp_parser
from ml.advanced_nlp_parser import advanced_parser
from ml.xai_explainer import xai_explainer

router = APIRouter()


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
        # Use advanced parser for complex queries
        # Detects: conflicts, negations, trade-offs, context, implicit preferences
        parsed = advanced_parser.parse_complex_query(request.query)
        
        # Enhance with BERT NER for additional entity extraction
        bert_parsed = nlp_parser.enhance_preferences(parsed)
        preferences = bert_parsed
    
    # Override with explicit preferences if provided
    if request.budget is not None:
        preferences['budget'] = request.budget
    
    if request.device_type:
        preferences['device_type'] = request.device_type
    
    if request.use_case:
        preferences['use_case'] = request.use_case
    
    if request.brand_preference:
        preferences['brand_preference'] = request.brand_preference
    
    # Get all devices from database
    all_devices = await Device.find_all().to_list()
    
    if not all_devices:
        raise HTTPException(status_code=404, detail="No devices found in database")
    
    # Convert to dict format for ML model
    device_dicts = [
        {
            'id': str(device.id),
            'brand': device.brand,
            'model_name': device.model_name,
            'model_image': device.model_image,
            'device_type': device.device_type,
            'specs': device.specs,
            'variants': [v.dict() for v in device.variants]
        }
        for device in all_devices
    ]
    
    # Train/update recommender
    recommender.fit(device_dicts)
    
    # Get recommendations
    recommendations = recommender.recommend_by_preferences(
        preferences,
        top_n=request.top_n
    )
    
    if not recommendations:
        return RecommendationResponse(
            recommendations=[],
            parsed_preferences=preferences,
            total_candidates=0
        )
    
    # Fetch full device details for recommendations
    device_map = {str(device.id): device for device in all_devices}
    device_dict_map = {d['id']: d for d in device_dicts}
    
    result_devices = []
    for device_id, score in recommendations:
        device = device_map.get(device_id)
        device_dict = device_dict_map.get(device_id)
        
        if device and device_dict:
            # Generate basic recommendation reason
            reason = _generate_reason(device, preferences, score)
            
            # Generate detailed XAI explanation if requested
            explanation = None
            if request.explain:
                xai_result = xai_explainer.explain_recommendation(
                    device=device_dict,
                    preferences=preferences,
                    score=score,
                    all_devices=device_dicts
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
                    id=str(device.id),
                    brand=device.brand,
                    model_name=device.model_name,
                    model_image=device.model_image,
                    device_type=device.device_type,
                    score=round(score, 3),
                    reason=reason,
                    specs=device.specs,
                    explanation=explanation
                )
            )
    
    return RecommendationResponse(
        recommendations=result_devices,
        parsed_preferences=preferences,
        total_candidates=len(all_devices)
    )


def _generate_reason(device: Device, preferences: dict, score: float) -> str:
    """Generate human-readable reason for recommendation"""
    
    reasons = []
    
    # Brand match
    if 'brand_preference' in preferences and preferences['brand_preference']:
        if device.brand in preferences['brand_preference']:
            reasons.append(f"Matches your preferred brand ({device.brand})")
    
    # Use case match
    if 'use_case' in preferences and preferences['use_case']:
        use_case = preferences['use_case']
        specs = device.specs
        
        if use_case == 'gaming':
            if 'Chipset' in specs and any(kw in specs['Chipset'].lower() for kw in ['snapdragon', 'dimensity', 'bionic']):
                reasons.append("Powerful processor for gaming")
        
        elif use_case == 'photography':
            if 'Main Camera' in specs or 'Camera' in specs:
                reasons.append("Excellent camera capabilities")
        
        elif use_case == 'battery':
            if 'Battery' in specs or 'Type_1' in specs:
                reasons.append("Long battery life")
    
    # Budget match
    if 'budget' in preferences and preferences['budget']:
        reasons.append(f"Within your budget")
    
    # High relevance score
    if score > 0.7:
        reasons.append("Highly relevant to your search")
    elif score > 0.5:
        reasons.append("Good match for your requirements")
    
    if not reasons:
        reasons.append("Recommended based on your preferences")
    
    return ". ".join(reasons) + "."
