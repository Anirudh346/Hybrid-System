"""
Explainable AI (XAI) Module for Device Recommendations

Provides detailed explanations for why devices are recommended, including:
- Feature contribution analysis
- Confidence scores
- Comparable alternatives
- Counterfactual explanations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class FeatureContribution:
    """Individual feature contribution to recommendation score"""
    feature_name: str
    value: Any
    contribution_score: float  # How much this feature contributed (0-1)
    importance: float  # Overall importance weight
    explanation: str  # Human-readable explanation


@dataclass
class Explanation:
    """Complete explanation for a recommendation"""
    device_id: str
    overall_score: float
    feature_contributions: List[FeatureContribution]
    match_summary: str
    top_reasons: List[str]
    comparable_alternatives: List[Dict[str, Any]]
    confidence: float  # 0-1, how confident the system is
    counterfactual: Optional[str] = None  # "If you changed X, you'd get Y"


class XAIExplainer:
    """Explainable AI module for device recommendations"""
    
    def __init__(self):
        # Feature importance weights (learned from user preferences)
        self.feature_weights = {
            'brand_match': 0.15,
            'price_fit': 0.25,
            'use_case_alignment': 0.30,
            'specs_quality': 0.20,
            'popularity': 0.10
        }
        
        # Spec importance by use case
        self.use_case_specs = {
            'gaming': {
                'Chipset': 0.35,
                'RAM': 0.25,
                'GPU': 0.20,
                'Display': 0.15,
                'Battery': 0.05
            },
            'photography': {
                'Main Camera': 0.40,
                'Selfie camera': 0.20,
                'Chipset': 0.15,
                'Display': 0.15,
                'Internal': 0.10
            },
            'battery': {
                'Battery': 0.50,
                'Charging': 0.30,
                'Chipset': 0.10,
                'Display': 0.10
            },
            'display': {
                'Display': 0.40,
                'Type': 0.30,
                'Size': 0.15,
                'Resolution': 0.15
            }
        }
    
    def explain_recommendation(
        self,
        device: Dict[str, Any],
        preferences: Dict[str, Any],
        score: float,
        all_devices: List[Dict[str, Any]],
        feature_matrix: Optional[np.ndarray] = None,
        similarity_scores: Optional[np.ndarray] = None
    ) -> Explanation:
        """
        Generate comprehensive explanation for why a device was recommended
        
        Args:
            device: The recommended device
            preferences: User preferences that led to recommendation
            score: The recommendation score
            all_devices: All candidate devices for comparison
            feature_matrix: TF-IDF feature matrix (if available)
            similarity_scores: Cosine similarity scores (if available)
        """
        
        # Calculate feature contributions
        contributions = self._calculate_feature_contributions(device, preferences)
        
        # Generate match summary
        summary = self._generate_match_summary(device, preferences, contributions)
        
        # Extract top reasons
        top_reasons = self._extract_top_reasons(contributions, top_n=3)
        
        # Find comparable alternatives
        alternatives = self._find_alternatives(device, all_devices, preferences, n=3)
        
        # Calculate confidence
        confidence = self._calculate_confidence(score, contributions)
        
        # Generate counterfactual
        counterfactual = self._generate_counterfactual(device, preferences, all_devices)
        
        return Explanation(
            device_id=device.get('id', ''),
            overall_score=score,
            feature_contributions=contributions,
            match_summary=summary,
            top_reasons=top_reasons,
            comparable_alternatives=alternatives,
            confidence=confidence,
            counterfactual=counterfactual
        )
    
    def _calculate_feature_contributions(
        self,
        device: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> List[FeatureContribution]:
        """Calculate how much each feature contributed to the recommendation"""
        
        contributions = []
        specs = device.get('specs', {})
        
        # 1. Brand Match
        if 'brand_preference' in preferences and preferences['brand_preference']:
            brand = device.get('brand', '').lower()
            brand_prefs = [b.lower() for b in preferences['brand_preference']]
            is_match = brand in brand_prefs
            
            contributions.append(FeatureContribution(
                feature_name='Brand Match',
                value=device.get('brand', 'Unknown'),
                contribution_score=1.0 if is_match else 0.0,
                importance=self.feature_weights['brand_match'],
                explanation=f"{'Matches' if is_match else 'Does not match'} your preferred brand"
            ))
        
        # 2. Price Fit
        if 'budget' in preferences and preferences['budget']:
            budget = preferences['budget']
            price = self._extract_price(device)
            
            if price > 0:
                price_fit = max(0, 1 - abs(price - budget) / budget)
                within_budget = price <= budget
                
                contributions.append(FeatureContribution(
                    feature_name='Price Fit',
                    value=f"${price:.0f}",
                    contribution_score=price_fit,
                    importance=self.feature_weights['price_fit'],
                    explanation=f"{'Within' if within_budget else 'Above'} your ${budget:.0f} budget"
                ))
        
        # 3. Use Case Alignment
        if 'use_case' in preferences and preferences['use_case']:
            use_case = preferences['use_case']
            spec_scores = self._evaluate_use_case_specs(specs, use_case)
            avg_score = np.mean(list(spec_scores.values())) if spec_scores else 0.5
            
            contributions.append(FeatureContribution(
                feature_name='Use Case Match',
                value=use_case.capitalize(),
                contribution_score=float(avg_score),
                importance=self.feature_weights['use_case_alignment'],
                explanation=f"{'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} fit for {use_case}"
            ))
            
            # Add individual spec contributions for use case
            for spec_name, spec_score in spec_scores.items():
                if spec_score > 0.3:  # Only show meaningful contributions
                    contributions.append(FeatureContribution(
                        feature_name=spec_name,
                        value=specs.get(spec_name, 'N/A'),
                        contribution_score=spec_score,
                        importance=self.use_case_specs.get(use_case, {}).get(spec_name, 0.1),
                        explanation=self._explain_spec(spec_name, specs.get(spec_name, ''), use_case)
                    ))
        
        # 4. Overall Specs Quality
        quality_score = self._evaluate_specs_quality(specs)
        contributions.append(FeatureContribution(
            feature_name='Specs Quality',
            value='Overall',
            contribution_score=quality_score,
            importance=self.feature_weights['specs_quality'],
            explanation=f"{'Premium' if quality_score > 0.7 else 'Mid-range' if quality_score > 0.4 else 'Budget'} specifications"
        ))
        
        return contributions
    
    def _evaluate_use_case_specs(self, specs: Dict[str, Any], use_case: str) -> Dict[str, float]:
        """Evaluate how well specs match a use case"""
        
        scores = {}
        
        if use_case == 'gaming':
            # Chipset evaluation
            chipset = specs.get('Chipset', '').lower()
            if any(kw in chipset for kw in ['snapdragon 8', 'dimensity 9', 'a17', 'a16']):
                scores['Chipset'] = 1.0
            elif any(kw in chipset for kw in ['snapdragon 7', 'dimensity 7', 'a15']):
                scores['Chipset'] = 0.7
            else:
                scores['Chipset'] = 0.4
            
            # RAM evaluation
            ram = self._extract_numeric_value(specs.get('Internal', ''))
            if ram >= 12:
                scores['RAM'] = 1.0
            elif ram >= 8:
                scores['RAM'] = 0.7
            else:
                scores['RAM'] = 0.4
            
            # Display refresh rate
            display = specs.get('Display', '').lower()
            if '120hz' in display or '144hz' in display:
                scores['Display'] = 1.0
            elif '90hz' in display:
                scores['Display'] = 0.7
            else:
                scores['Display'] = 0.4
        
        elif use_case == 'photography':
            # Camera MP evaluation
            camera = specs.get('Main Camera', '') or specs.get('Camera', '')
            mp = self._extract_numeric_value(camera)
            if mp >= 64:
                scores['Main Camera'] = 1.0
            elif mp >= 48:
                scores['Main Camera'] = 0.7
            else:
                scores['Main Camera'] = 0.5
        
        elif use_case == 'battery':
            # Battery capacity
            battery = specs.get('Battery', '') or specs.get('Type_1', '')
            mah = self._extract_numeric_value(battery)
            if mah >= 5000:
                scores['Battery'] = 1.0
            elif mah >= 4000:
                scores['Battery'] = 0.7
            else:
                scores['Battery'] = 0.4
            
            # Fast charging
            charging = specs.get('Charging', '').lower()
            watts = self._extract_numeric_value(charging)
            if watts >= 65:
                scores['Charging'] = 1.0
            elif watts >= 33:
                scores['Charging'] = 0.7
            else:
                scores['Charging'] = 0.4
        
        return scores
    
    def _evaluate_specs_quality(self, specs: Dict[str, Any]) -> float:
        """Evaluate overall quality of specifications"""
        
        quality_indicators = []
        
        # Check for premium features
        chipset = specs.get('Chipset', '').lower()
        if any(kw in chipset for kw in ['snapdragon 8', 'dimensity 9', 'a17', 'a16', 'a15']):
            quality_indicators.append(1.0)
        elif any(kw in chipset for kw in ['snapdragon 7', 'dimensity 7']):
            quality_indicators.append(0.7)
        else:
            quality_indicators.append(0.4)
        
        # RAM
        ram = self._extract_numeric_value(specs.get('Internal', ''))
        quality_indicators.append(min(1.0, ram / 12))
        
        # Camera
        camera_mp = self._extract_numeric_value(specs.get('Main Camera', '') or specs.get('Camera', ''))
        quality_indicators.append(min(1.0, camera_mp / 64))
        
        # Battery
        battery_mah = self._extract_numeric_value(specs.get('Battery', '') or specs.get('Type_1', ''))
        quality_indicators.append(min(1.0, battery_mah / 5000))
        
        return float(np.mean(quality_indicators)) if quality_indicators else 0.5
    
    def _extract_numeric_value(self, text: str) -> float:
        """Extract first numeric value from string"""
        if not text:
            return 0.0
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return float(match.group(1)) if match else 0.0
    
    def _extract_price(self, device: Dict[str, Any]) -> float:
        """Extract price from device variants"""
        variants = device.get('variants', [])
        if variants:
            prices = [v.get('price', 0) for v in variants if v.get('price', 0) > 0]
            return min(prices) if prices else 0
        return 0
    
    def _explain_spec(self, spec_name: str, spec_value: str, use_case: str) -> str:
        """Generate human-readable explanation for a spec"""
        
        if use_case == 'gaming':
            if spec_name == 'Chipset':
                if any(kw in spec_value.lower() for kw in ['snapdragon 8', 'dimensity 9', 'a17']):
                    return "Flagship processor delivers excellent gaming performance"
                return "Capable processor for gaming"
            elif spec_name == 'RAM':
                ram = self._extract_numeric_value(spec_value)
                if ram >= 12:
                    return "Ample RAM for smooth multitasking while gaming"
                return "Sufficient RAM for gaming"
            elif spec_name == 'Display':
                if '120hz' in spec_value.lower() or '144hz' in spec_value.lower():
                    return "High refresh rate for smooth gameplay"
                return "Good display for gaming"
        
        elif use_case == 'photography':
            if 'Camera' in spec_name:
                mp = self._extract_numeric_value(spec_value)
                if mp >= 64:
                    return "High-resolution camera for detailed photos"
                return "Capable camera system"
        
        elif use_case == 'battery':
            if spec_name == 'Battery':
                mah = self._extract_numeric_value(spec_value)
                if mah >= 5000:
                    return "Large battery for all-day usage"
                return "Decent battery capacity"
            elif spec_name == 'Charging':
                watts = self._extract_numeric_value(spec_value)
                if watts >= 65:
                    return "Ultra-fast charging support"
                return "Fast charging available"
        
        return f"{spec_name}: {spec_value}"
    
    def _generate_match_summary(
        self,
        device: Dict[str, Any],
        preferences: Dict[str, Any],
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate overall match summary"""
        
        # Calculate weighted score
        total_score = sum(
            c.contribution_score * c.importance 
            for c in contributions
        )
        
        brand = device.get('brand', 'This device')
        model = device.get('model_name', '')
        
        if total_score > 0.7:
            match_quality = "excellent"
        elif total_score > 0.5:
            match_quality = "good"
        else:
            match_quality = "moderate"
        
        use_case = preferences.get('use_case', '')
        use_case_text = f" for {use_case}" if use_case else ""
        
        return f"{brand} {model} is an {match_quality} match{use_case_text} based on your preferences."
    
    def _extract_top_reasons(
        self,
        contributions: List[FeatureContribution],
        top_n: int = 3
    ) -> List[str]:
        """Extract top reasons for recommendation"""
        
        # Sort by weighted contribution
        sorted_contributions = sorted(
            contributions,
            key=lambda c: c.contribution_score * c.importance,
            reverse=True
        )
        
        reasons = []
        for contrib in sorted_contributions[:top_n]:
            if contrib.contribution_score > 0.3:  # Only meaningful contributions
                reasons.append(contrib.explanation)
        
        return reasons if reasons else ["Matches your overall preferences"]
    
    def _find_alternatives(
        self,
        device: Dict[str, Any],
        all_devices: List[Dict[str, Any]],
        preferences: Dict[str, Any],
        n: int = 3
    ) -> List[Dict[str, Any]]:
        """Find comparable alternative devices"""
        
        device_id = device.get('id')
        device_price = self._extract_price(device)
        device_type = device.get('device_type', '')
        
        alternatives = []
        
        for other in all_devices:
            if other.get('id') == device_id:
                continue
            
            # Same device type
            if other.get('device_type', '') != device_type:
                continue
            
            other_price = self._extract_price(other)
            
            # Similar price range (±20%)
            if device_price > 0 and other_price > 0:
                price_diff = abs(other_price - device_price) / device_price
                if price_diff > 0.2:
                    continue
            
            alternatives.append({
                'id': other.get('id'),
                'brand': other.get('brand'),
                'model_name': other.get('model_name'),
                'price': other_price,
                'reason': self._compare_devices(device, other)
            })
        
        return alternatives[:n]
    
    def _compare_devices(self, device1: Dict[str, Any], device2: Dict[str, Any]) -> str:
        """Generate comparison reason between two devices"""
        
        brand2 = device2.get('brand', '')
        
        # Simple comparison
        price1 = self._extract_price(device1)
        price2 = self._extract_price(device2)
        
        if price2 < price1:
            return f"Similar specs at a lower price point"
        elif price2 > price1:
            return f"Premium alternative with enhanced features"
        else:
            return f"Comparable option from {brand2}"
    
    def _calculate_confidence(
        self,
        score: float,
        contributions: List[FeatureContribution]
    ) -> float:
        """Calculate confidence in the recommendation"""
        
        # Factors affecting confidence:
        # 1. Overall score
        # 2. Number of strong contributions
        # 3. Variance in contributions
        
        strong_contributions = sum(
            1 for c in contributions 
            if c.contribution_score * c.importance > 0.15
        )
        
        contribution_scores = [
            c.contribution_score * c.importance 
            for c in contributions
        ]
        variance = np.var(contribution_scores) if contribution_scores else 0
        
        # Normalize confidence
        confidence = (
            score * 0.5 +  # Base score
            min(strong_contributions / 5, 1.0) * 0.3 +  # Strong feature matches
            (1 - min(variance, 1.0)) * 0.2  # Low variance = more confident
        )
        
        return float(min(1.0, max(0.0, confidence)))
    
    def _generate_counterfactual(
        self,
        device: Dict[str, Any],
        preferences: Dict[str, Any],
        all_devices: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate counterfactual explanation"""
        
        # "If you increased your budget by $X, you could get Y"
        # "If you considered Z brand, you'd have more options"
        
        if 'budget' in preferences and preferences['budget']:
            budget = preferences['budget']
            device_price = self._extract_price(device)
            
            # Find devices just above budget
            higher_budget_devices = [
                d for d in all_devices
                if self._extract_price(d) > budget
                and self._extract_price(d) <= budget * 1.3
            ]
            
            if higher_budget_devices:
                avg_better_price = np.mean([self._extract_price(d) for d in higher_budget_devices])
                increase = avg_better_price - budget
                return f"Increasing your budget by ${increase:.0f} would give you access to {len(higher_budget_devices)} more premium options"
        
        return None


# Global explainer instance
xai_explainer = XAIExplainer()
