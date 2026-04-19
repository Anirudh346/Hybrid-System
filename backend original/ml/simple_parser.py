"""
Lightweight query parser without external ML dependencies.
Extracts preferences from natural language text using simple pattern matching.
"""

import re
from typing import Dict, Any


class SimpleParser:
    """Simple rule-based query parser"""
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured preferences.
        
        Args:
            query: Natural language query string
            
        Returns:
            dict with extracted preferences
        """
        if not query:
            return {}
            
        query_lower = query.lower()
        preferences = {'query': query}
        
        # Extract budget
        budget_patterns = [
            r'\$?([\d,]+)\s*(usd|eur|gbp)?',
            r'under\s+\$?([\d,]+)',
            r'budget\s+\$?([\d,]+)',
            r'max\s+\$?([\d,]+)',
            r'([\d,]+)\s*dollars',
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query_lower)
            if match:
                budget_str = match.group(1).replace(',', '')
                try:
                    preferences['budget'] = float(budget_str)
                    break
                except ValueError:
                    pass
        
        # Extract use case
        use_cases = {
            'gaming': ['gaming', 'game', 'gamer', 'fps', 'performance', 'powerful'],
            'camera': ['camera', 'photo', 'photography', 'picture', 'video'],
            'battery': ['battery', 'battery life', 'long battery'],
            'business': ['business', 'work', 'professional', 'productive'],
            'budget': ['budget', 'cheap', 'affordable', 'inexpensive'],
            'premium': ['premium', 'flagship', 'high-end', 'top-tier'],
        }
        
        for use_case, keywords in use_cases.items():
            for keyword in keywords:
                if keyword in query_lower:
                    preferences['use_case'] = use_case
                    break
        
        # Extract brand preferences
        brands = ['apple', 'samsung', 'google', 'pixel', 'iphone', 'xiaomi', 'oneplus', 
                  'motorola', 'nokia', 'sony', 'lg', 'huawei', 'realme', 'poco', 'oppo', 'vivo']
        
        brand_prefs = []
        for brand in brands:
            if brand in query_lower:
                brand_prefs.append(brand.capitalize())
        
        if brand_prefs:
            preferences['brand_preference'] = brand_prefs
        
        # Extract OS preference
        os_patterns = {
            'iOS': ['iphone', 'ios', 'apple'],
            'Android': ['android', 'samsung', 'google', 'xiaomi'],
        }
        
        for os, keywords in os_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    preferences['os'] = os
                    break
        
        # Extract device type
        device_types = {
            'phone': ['phone', 'smartphone', 'mobile'],
            'tablet': ['tablet', 'ipad'],
            'smartwatch': ['watch', 'smartwatch'],
        }
        
        for dev_type, keywords in device_types.items():
            for keyword in keywords:
                if keyword in query_lower:
                    preferences['device_type'] = dev_type
                    break
        
        return preferences


# Create global instance
simple_parser = SimpleParser()
