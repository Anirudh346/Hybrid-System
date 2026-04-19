"""
Advanced NLP Query Parser for Complex, Inconsistent, and Context-Dependent Prompts

Handles:
- Conflicting requirements ("cheap flagship")
- Multi-intent queries ("gaming AND photography")
- Implicit preferences ("I travel" → battery, dual SIM)
- Context references ("better than iPhone 12")
- Trade-offs ("sacrifice camera for battery")
- Negations ("not Samsung", "without notch")
- Comparisons ("similar to", "cheaper than")
"""

from transformers import pipeline
from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class ParsedIntent:
    """Structured representation of user intent"""
    primary_use_case: str
    secondary_use_cases: List[str]
    must_have: List[str]
    nice_to_have: List[str]
    avoid: List[str]
    trade_offs: List[Tuple[str, str]]  # (sacrifice, for)
    context_references: List[str]
    confidence: float


class AdvancedNLPParser:
    """Advanced NLP parser for complex queries"""
    
    def __init__(self):
        # Load BERT NER
        try:
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")  # type: ignore
        except Exception:
            self.ner_pipeline = None
        
        # Enhanced keyword mappings with synonyms and variations
        self.use_case_keywords = {
            'gaming': {
                'keywords': ['gaming', 'games', 'game', 'gamer', 'performance', 'fps', 'pubg', 'cod', 'fortnite', 'apex'],
                'implicit': ['smooth', 'lag-free', 'fast', 'powerful'],
                'specs': ['processor', 'chipset', 'gpu', 'refresh rate', 'cooling']
            },
            'photography': {
                'keywords': ['photo', 'camera', 'photography', 'picture', 'video', 'vlog', 'youtube', 'instagram', 'tiktok'],
                'implicit': ['memories', 'capture', 'shoot', 'record'],
                'specs': ['megapixel', 'mp', 'lens', 'optical', 'zoom', 'stabilization', 'ois']
            },
            'battery': {
                'keywords': ['battery', 'endurance', 'long-lasting', 'backup', 'charge', 'all-day', 'heavy use'],
                'implicit': ['travel', 'outdoor', 'field work', 'on-the-go'],
                'specs': ['mah', 'fast charging', 'wireless charging', 'power']
            },
            'display': {
                'keywords': ['display', 'screen', 'amoled', 'oled', 'hdr', 'bright', 'outdoor visibility'],
                'implicit': ['media', 'movies', 'streaming', 'netflix', 'youtube'],
                'specs': ['refresh rate', 'hz', 'resolution', 'nits', 'brightness']
            },
            'business': {
                'keywords': ['business', 'work', 'productivity', 'office', 'professional', 'meetings'],
                'implicit': ['calls', 'emails', 'documents', 'multitask'],
                'specs': ['security', 'knox', 'enterprise']
            },
            'budget': {
                'keywords': ['cheap', 'affordable', 'budget', 'value', 'economical', 'bang for buck'],
                'implicit': ['student', 'first phone', 'basic'],
                'specs': []
            }
        }
        
        # Conflict resolution patterns
        self.conflicts = {
            ('budget', 'gaming'): 'mid_range_gaming',
            ('budget', 'photography'): 'mid_range_camera',
            ('cheap', 'flagship'): 'flagship_killer',
            ('cheap', 'premium'): 'value_flagship'
        }
        
        # Context patterns
        self.context_patterns = {
            'comparison': [
                r'better than (\w+)',
                r'upgrade from (\w+)',
                r'similar to (\w+)',
                r'like (\w+) but',
                r'compared to (\w+)'
            ],
            'negation': [
                r'not (\w+)',
                r'without (\w+)',
                r'except (\w+)',
                r'no (\w+)',
                r"don't want (\w+)"
            ],
            'trade_off': [
                r'sacrifice (\w+) for (\w+)',
                r'prefer (\w+) over (\w+)',
                r'care more about (\w+) than (\w+)',
                r'(\w+) more important than (\w+)'
            ]
        }
        
        # Implicit preference mappings
        self.implicit_preferences = {
            'travel': ['battery', 'dual sim', 'lightweight'],
            'outdoor': ['battery', 'durable', 'ip rating', 'bright display'],
            'student': ['budget', 'battery', 'value'],
            'content creator': ['camera', 'video', 'storage', 'display'],
            'professional': ['business', 'security', 'battery'],
            'elderly': ['simple', 'large display', 'loud speaker'],
            'parent': ['durable', 'parental controls', 'value']
        }
    
    def parse_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Parse complex, potentially conflicting queries
        
        Args:
            query: Natural language query
        
        Returns:
            Enhanced preferences dict with conflict resolution
        """
        
        query_lower = query.lower()
        
        # Basic extraction
        preferences = {
            'query': query,
            'device_type': [],
            'brand_preference': [],
            'brand_avoid': [],
            'use_case': '',
            'secondary_use_cases': [],
            'budget': None,
            'must_have_features': [],
            'nice_to_have_features': [],
            'avoid_features': [],
            'trade_offs': [],
            'context_references': [],
            'priority': 'balanced',  # balanced, performance, value, camera, battery
            'confidence': 1.0
        }
        
        # Extract all components
        budget = self._extract_budget(query_lower)
        if budget:
            preferences['budget'] = budget
        
        brands = self._extract_brands(query_lower)
        if brands:
            preferences['brand_preference'] = brands
        
        device_type = self._extract_device_type(query_lower)
        if device_type:
            preferences['device_type'] = device_type
        
        # Extract multiple use cases
        use_cases = self._extract_multiple_use_cases(query_lower)
        if use_cases:
            preferences['use_case'] = use_cases[0]
            preferences['secondary_use_cases'] = use_cases[1:]
        
        # Extract negations (brands/features to avoid)
        avoid = self._extract_negations(query_lower)
        preferences['brand_avoid'] = avoid.get('brands', [])
        preferences['avoid_features'] = avoid.get('features', [])
        
        # Extract trade-offs
        trade_offs = self._extract_trade_offs(query_lower)
        preferences['trade_offs'] = trade_offs
        
        # Extract context references
        context = self._extract_context_references(query_lower)
        preferences['context_references'] = context
        
        # Extract implicit preferences
        implicit = self._extract_implicit_preferences(query_lower)
        preferences['must_have_features'].extend(implicit.get('must_have', []))
        preferences['nice_to_have_features'].extend(implicit.get('nice_to_have', []))
        
        # Detect and resolve conflicts
        conflicts = self._detect_conflicts(preferences)
        if conflicts:
            preferences = self._resolve_conflicts(preferences, conflicts)
            preferences['confidence'] = 0.7  # Lower confidence due to conflicts
        
        # Determine priority
        priority = self._determine_priority(preferences, query_lower)
        preferences['priority'] = priority
        
        return preferences
    
    def _extract_multiple_use_cases(self, query: str) -> List[str]:
        """Extract multiple use cases from query"""
        
        found_cases = []
        
        for use_case, patterns in self.use_case_keywords.items():
            # Check keywords
            if any(kw in query for kw in patterns['keywords']):
                found_cases.append(use_case)
                continue
            
            # Check implicit keywords
            if any(kw in query for kw in patterns['implicit']):
                found_cases.append(use_case)
        
        return found_cases
    
    def _extract_negations(self, query: str) -> Dict[str, List[str]]:
        """Extract things user wants to avoid"""
        
        avoid = {'brands': [], 'features': []}
        
        # Check negation patterns
        for pattern in self.context_patterns['negation']:
            matches = re.finditer(pattern, query)
            for match in matches:
                avoided_item = match.group(1)
                
                # Check if it's a brand
                if avoided_item in ['samsung', 'apple', 'google', 'oneplus', 'xiaomi']:
                    avoid['brands'].append(avoided_item.capitalize())
                else:
                    avoid['features'].append(avoided_item)
        
        return avoid
    
    def _extract_trade_offs(self, query: str) -> List[Tuple[str, str]]:
        """Extract trade-off preferences"""
        
        trade_offs = []
        
        for pattern in self.context_patterns['trade_off']:
            matches = re.finditer(pattern, query)
            for match in matches:
                if len(match.groups()) >= 2:
                    sacrifice = match.group(1)
                    prefer = match.group(2)
                    trade_offs.append((sacrifice, prefer))
        
        return trade_offs
    
    def _extract_context_references(self, query: str) -> List[str]:
        """Extract references to other devices or comparisons"""
        
        references = []
        
        for pattern in self.context_patterns['comparison']:
            matches = re.finditer(pattern, query)
            for match in matches:
                reference = match.group(1)
                references.append(reference)
        
        return references
    
    def _extract_implicit_preferences(self, query: str) -> Dict[str, List[str]]:
        """Extract implicit preferences from lifestyle mentions"""
        
        implicit = {'must_have': [], 'nice_to_have': []}
        
        for lifestyle, prefs in self.implicit_preferences.items():
            if lifestyle in query:
                # First preference is must-have, rest are nice-to-have
                if prefs:
                    implicit['must_have'].append(prefs[0])
                    implicit['nice_to_have'].extend(prefs[1:])
        
        return implicit
    
    def _detect_conflicts(self, preferences: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Detect conflicting requirements"""
        
        conflicts = []
        
        # Budget conflicts
        if preferences.get('budget'):
            budget = preferences['budget']
            
            # Cheap but high-end use case
            if budget < 500:
                if preferences.get('use_case') == 'gaming':
                    conflicts.append(('budget', 'gaming'))
                if 'photography' in preferences.get('secondary_use_cases', []):
                    conflicts.append(('budget', 'photography'))
        
        # Check for explicit conflict keywords
        query_lower = preferences.get('query', '').lower()
        if 'cheap' in query_lower or 'budget' in query_lower:
            if 'flagship' in query_lower or 'premium' in query_lower:
                conflicts.append(('cheap', 'flagship'))
        
        return conflicts
    
    def _resolve_conflicts(self, preferences: Dict[str, Any], conflicts: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Resolve conflicting requirements"""
        
        for conflict in conflicts:
            resolution = self.conflicts.get(conflict)
            
            if resolution == 'mid_range_gaming':
                # Adjust expectations for budget gaming
                preferences['must_have_features'].append('mid_range_processor')
                preferences['nice_to_have_features'].append('90hz_display')
                preferences['priority'] = 'value'
            
            elif resolution == 'flagship_killer':
                # Look for flagship killers (OnePlus, Poco, etc.)
                preferences['brand_preference'].extend(['OnePlus', 'Poco', 'Realme'])
                preferences['priority'] = 'value'
        
        return preferences
    
    def _determine_priority(self, preferences: Dict[str, Any], query: str) -> str:
        """Determine user's priority from query"""
        
        # Check for explicit priority keywords
        if any(kw in query for kw in ['best camera', 'camera beast', 'photography']):
            return 'camera'
        
        if any(kw in query for kw in ['best battery', 'long battery', 'all-day']):
            return 'battery'
        
        if any(kw in query for kw in ['best performance', 'fastest', 'gaming beast']):
            return 'performance'
        
        if any(kw in query for kw in ['best value', 'bang for buck', 'affordable']):
            return 'value'
        
        # Check trade-offs
        if preferences.get('trade_offs'):
            _, preferred = preferences['trade_offs'][0]
            return preferred
        
        return 'balanced'
    
    def _extract_budget(self, query: str) -> Optional[float]:
        """Extract budget from query"""
        
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
        
        brand_keywords = [
            'apple', 'samsung', 'google', 'oneplus', 'xiaomi', 'oppo', 'vivo',
            'realme', 'motorola', 'nokia', 'sony', 'lg', 'huawei', 'honor',
            'asus', 'lenovo', 'alcatel', 'zte', 'poco', 'nothing'
        ]
        
        found_brands = []
        for brand in brand_keywords:
            if brand in query:
                found_brands.append(brand.capitalize())
        
        return found_brands
    
    def _extract_device_type(self, query: str) -> List[str]:
        """Extract device type from query"""
        
        device_type_keywords = {
            'mobile': ['phone', 'mobile', 'smartphone', 'iphone', 'android'],
            'tablet': ['tablet', 'ipad', 'tab'],
            'smartwatch': ['watch', 'smartwatch', 'band', 'wearable']
        }
        
        device_types = []
        for dtype, keywords in device_type_keywords.items():
            if any(keyword in query for keyword in keywords):
                device_types.append(dtype)
        
        if not device_types:
            device_types = ['mobile']
        
        return device_types


# Global advanced parser instance
advanced_parser = AdvancedNLPParser()
