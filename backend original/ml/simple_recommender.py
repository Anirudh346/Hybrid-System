"""
Lightweight device recommender system without heavy dependencies.
Uses simple scoring based on preferences and device specifications.
"""

class SimpleRecommender:
    """Lightweight device recommendation engine"""
    
    def __init__(self):
        self.devices = []
        
    def fit(self, device_dicts):
        """Train/load devices for recommendation"""
        self.devices = device_dicts if device_dicts else []
        
    def recommend_by_preferences(self, preferences, top_n=10):
        """
        Recommend devices based on user preferences.
        
        Args:
            preferences: dict with keys like 'budget', 'use_case', 'brand_preference', etc.
            top_n: number of recommendations to return
            
        Returns:
            list of (device_id, score) tuples sorted by score descending
        """
        if not self.devices:
            return []
            
        scores = {}
        
        for device in self.devices:
            device_id = device.get('id')
            score = 0.0
            
            # Budget match
            if 'budget' in preferences and preferences['budget']:
                try:
                    price = float(device.get('price', 0))
                    budget = float(preferences['budget'])
                    if 0 < price <= budget:
                        score += 0.3  # Base score for being in budget
                        # Bonus for being closer to budget
                        discount = (budget - price) / budget
                        score += 0.1 * (1 - discount)
                except (ValueError, TypeError):
                    pass
            
            # Brand preference match
            if 'brand_preference' in preferences and preferences['brand_preference']:
                device_brand = (device.get('brand') or '').lower()
                for pref_brand in preferences['brand_preference']:
                    if device_brand == pref_brand.lower():
                        score += 0.2
                        
            # Use case match
            if 'use_case' in preferences and preferences['use_case']:
                use_case = (preferences['use_case'] or '').lower()
                
                if 'gaming' in use_case:
                    # Gaming devices should have good processors
                    chipset = (device.get('chipset') or '').lower()
                    if any(kw in chipset for kw in ['snapdragon', 'dimensity', 'a17', 'a18', 'a16', 'a15']):
                        score += 0.25
                    else:
                        score += 0.05
                        
                elif 'camera' in use_case or 'photo' in use_case:
                    # Photography devices should have good cameras
                    camera = device.get('camera') or ''
                    if camera and 'mp' in (camera or '').lower():
                        score += 0.25
                    else:
                        score += 0.05
                        
                elif 'battery' in use_case:
                    # Battery life preference
                    battery = device.get('battery') or ''
                    if battery:
                        score += 0.2
                    else:
                        score += 0.05
                        
                elif 'business' in use_case or 'work' in use_case:
                    # Business devices - consider OS and display
                    score += 0.1
                    
            # Device type preference
            if 'device_type' in preferences and preferences['device_type']:
                pref_type = (preferences['device_type'] or '').lower()
                dev_type = (device.get('device_type') or '').lower()
                if pref_type in dev_type or dev_type in pref_type:
                    score += 0.15
                    
            # OS preference
            if 'os' in preferences and preferences['os']:
                pref_os = (preferences['os'] or '').upper()
                dev_os = (device.get('os') or '').upper()
                if pref_os in dev_os or dev_os in pref_os:
                    score += 0.15
                    
            # Default base score if no preferences matched
            if score == 0:
                # All devices get a small base score
                score = 0.1
            
            # Store score
            if score > 0:
                scores[device_id] = score
        
        # Sort by score descending and return top N
        if not scores:
            # If no matches, return all devices with equal score
            return [(d['id'], 0.1) for d in self.devices[:top_n]]
            
        sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_n]


# Create global instance
simple_recommender = SimpleRecommender()
