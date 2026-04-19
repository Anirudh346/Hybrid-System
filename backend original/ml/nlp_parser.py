from transformers import pipeline
from typing import Dict, Any, List, Optional
import re


class NLPQueryParser:
    """Parse natural language queries for device recommendations using BERT"""
    
    def __init__(self):
        # Load BERT-based NER model from Hugging Face
        try:
            self.ner_pipeline = pipeline(
                "ner",  # type: ignore
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
        except Exception:
            # Fallback to None if model loading fails
            self.ner_pipeline = None
        
        # Keyword mappings
        self.use_case_keywords = {
            'flagship': ['flagship', 'premium', 'high-end', 'top-end', 'best available', 'upgrade', 'replace my'],
            'gaming': ['gaming', 'games', 'game', 'gamer', 'fps'],
            'photography': ['photo', 'camera', 'photography', 'picture', 'vlog'],
            'battery': ['battery', 'endurance', 'long-lasting', 'backup', 'charge'],
            'display': ['display', 'screen', 'amoled', 'oled', 'refresh rate', 'hdr'],
            'business': ['business', 'work', 'productivity', 'office'],
            'budget': ['cheap', 'affordable', 'budget', 'value', 'economical']
        }
        
        self.brand_keywords = [
            'apple', 'samsung', 'google', 'oneplus', 'xiaomi', 'oppo', 'vivo',
            'realme', 'motorola', 'nokia', 'sony', 'lg', 'huawei', 'honor',
            'asus', 'lenovo', 'alcatel', 'zte'
        ]
        
        self.device_type_keywords = {
            'mobile': ['phone', 'mobile', 'smartphone', 'iphone', 'android'],
            'tablet': ['tablet', 'ipad', 'tab'],
            'smartwatch': ['watch', 'smartwatch', 'band', 'wearable']
        }
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured preferences
        
        Args:
            query: Natural language query like "best gaming phone under $500"
        
        Returns:
            Dict with extracted preferences:
                - budget: float
                - device_type: str or List[str]
                - use_case: str
                - brand_preference: List[str]
                - query: original query
        """
        
        query_lower = query.lower()
        preferences = {
            'query': query,
            'device_type': [],
            'brand_preference': [],
            'use_case': '',
            'budget': None,
            'min_ram_gb': None,
            'min_refresh_hz': None
        }
        
        # Extract budget
        budget = self._extract_budget(query_lower)
        if budget:
            preferences['budget'] = budget
        
        # Extract explicit RAM requirement
        ram_match = re.search(r'(\\d+)\\s*gb\\s*(?:of\\s+)?ram|at\\s+least\\s+(\\d+)\\s*gb', query_lower)
        if ram_match:
            ram_val = int(ram_match.group(1) or ram_match.group(2))
            preferences['min_ram_gb'] = ram_val
        
        # Extract explicit refresh rate requirement
        refresh_match = re.search(r'(\\d{2,3})\\s*hz|refresh\\s+rate\\s+of\\s+(\\d{2,3})', query_lower)
        if refresh_match:
            refresh_val = int(refresh_match.group(1) or refresh_match.group(2))
            preferences['min_refresh_hz'] = refresh_val
        
        # Extract brands
        brands = self._extract_brands(query_lower)
        if brands:
            preferences['brand_preference'] = brands
        
        # Extract device type
        device_type = self._extract_device_type(query_lower)
        if device_type:
            preferences['device_type'] = device_type
        
        # Extract use case
        use_case = self._extract_use_case(query_lower)
        if use_case:
            preferences['use_case'] = use_case
        
        return preferences
    
    def _extract_budget(self, query: str) -> Optional[float]:
        """Extract budget from query"""
        
        # Patterns: "under $500", "below 800", "less than 1000", "max 600"
        patterns = [
            r'under\s+\$?(\d+(?:,\d+)?)',
            r'below\s+\$?(\d+(?:,\d+)?)',
            r'less\s+than\s+\$?(\d+(?:,\d+)?)',
            r'max\s+\$?(\d+(?:,\d+)?)',
            r'around\s+\$?(\d+(?:,\d+)?)',
            r'about\s+\$?(\d+(?:,\d+)?)',
            r'\$(\d+(?:,\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                budget_str = match.group(1).replace(',', '')
                return float(budget_str)
        
        return None
    
    def _extract_brands(self, query: str) -> List[str]:
        """Extract brand mentions from query"""
        
        found_brands = []
        for brand in self.brand_keywords:
            if brand in query:
                found_brands.append(brand.capitalize())
        
        return found_brands
    
    def _extract_device_type(self, query: str) -> List[str]:
        """Extract device type from query"""
        
        device_types = []
        
        # Explicit tablet mentions
        if any(kw in query for kw in ['tablet', 'ipad', 'tab']):
            device_types.append('tablet')
        # Explicit watch mentions
        elif any(kw in query for kw in ['watch', 'smartwatch', 'band', 'wearable']):
            device_types.append('smartwatch')
        # Phone/mobile mentions OR flagship/upgrade hints (assume phone)
        elif any(kw in query for kw in ['phone', 'mobile', 'smartphone', 'flagship', 'upgrade', 'replace my']):
            device_types.append('mobile')
        
        # Default to mobile if not specified
        if not device_types:
            device_types = ['mobile']
        
        return device_types
    
    def _extract_use_case(self, query: str) -> str:
        """Extract primary use case from query"""
        
        # Priority order: flagship first (most specific)
        for use_case, keywords in self.use_case_keywords.items():
            if any(keyword in query for keyword in keywords):
                return use_case
        
        return ''
    
    def enhance_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance preferences with BERT NER analysis"""
        
        if self.ner_pipeline is None or 'query' not in preferences:
            return preferences
        
        # Use BERT NER to extract entities
        entities = self.ner_pipeline(preferences['query'])
        
        # Process extracted entities
        for entity in entities:
            entity_text = entity['word'].lower().replace('#', '')
            entity_type = entity['entity_group']
            
            # Extract organizations (potential brands)
            if entity_type == 'ORG':
                for brand in self.brand_keywords:
                    if brand in entity_text:
                        if brand.capitalize() not in preferences['brand_preference']:
                            preferences['brand_preference'].append(brand.capitalize())
            
            # Extract monetary values
            elif entity_type in ['MONEY', 'CARDINAL']:
                amount = re.search(r'(\d+(?:,\d+)?)', entity_text)
                if amount and not preferences.get('budget'):
                    try:
                        preferences['budget'] = float(amount.group(1).replace(',', ''))
                    except:
                        pass
        
        return preferences


# Global NLP parser instance
nlp_parser = NLPQueryParser()
