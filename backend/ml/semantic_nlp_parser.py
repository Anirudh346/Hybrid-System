"""
Semantic NLP Parser using BERT embeddings for comprehensive query understanding.
Replaces regex-based parsing with semantic similarity and transformer models.
"""

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import Dict, Any, List, Tuple, Optional
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticNLPParser:
    """
    Advanced NLP parser using BERT embeddings for semantic understanding.
    
    Features:
    - Use case classification via semantic similarity
    - Multi-intent detection with confidence scores
    - Semantic synonym understanding (e.g., "fast" → "120Hz")
    - Implicit preference inference (e.g., "travel" → battery focus)
    - Spec extraction with natural language support
    """
    
    def __init__(self):
        """Initialize semantic models and reference data"""
        
        logger.info("Initializing Semantic NLP Parser...")
        
        # Load sentence transformer model (384-dim, fast, 80MB)
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Loaded sentence-transformers model")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            self.embedder = None
        
        # Load NER model for entity extraction fallback
        try:
            self.ner_pipeline = pipeline(
                task="token-classification",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            logger.info("✓ Loaded NER model")
        except Exception as e:
            logger.warning(f"NER model not available: {e}")
            self.ner_pipeline = None
        
        # Define use case reference queries (semantic examples)
        self.use_case_references = {
            'gaming': [
                "I want to play mobile games smoothly",
                "Need phone for PUBG and Call of Duty",
                "Gaming phone with high FPS and good cooling",
                "Best device for competitive mobile gaming",
                "Phone that handles graphics-intensive games",
                "Want smooth gameplay without lag",
                "Need phone for gaming tournaments",
                "Mobile gaming setup with fast processor",
                "Phone for Genshin Impact and demanding games",
                "Gaming device with high refresh rate screen"
            ],
            'photography': [
                "I need excellent camera quality",
                "Phone for professional photography",
                "Best camera phone for content creation",
                "Want to take amazing photos and videos",
                "Need good camera for Instagram and YouTube",
                "Camera quality is my top priority",
                "Phone for taking professional pictures",
                "Want great low-light photography",
                "Need phone for vlogging and video recording",
                "Best camera system for smartphone photography"
            ],
            'battery': [
                "Phone that lasts all day without charging",
                "Need long battery life for heavy usage",
                "Want device with excellent battery endurance",
                "Phone that won't die quickly",
                "Need phone with great battery backup",
                "Long-lasting battery for traveling",
                "Phone that can go two days without charging",
                "Battery life is most important to me",
                "Need phone with large battery capacity",
                "Want phone that charges fast and lasts long"
            ],
            'display': [
                "Want phone with amazing screen quality",
                "Need bright vibrant AMOLED display",
                "Phone with high refresh rate screen",
                "Best display quality for watching videos",
                "Want smooth scrolling and vivid colors",
                "Phone with HDR support and great brightness",
                "Need phone for media consumption",
                "Want large screen with high resolution",
                "Display quality matters most to me",
                "Phone with 120Hz smooth display"
            ],
            'business': [
                "Need phone for work and productivity",
                "Business phone with good security",
                "Professional device for office work",
                "Phone for emails and video calls",
                "Need reliable phone for business use",
                "Want productivity-focused smartphone",
                "Phone for work with good multitasking",
                "Business device with professional features",
                "Need phone for corporate environment",
                "Work phone with security features"
            ],
            'flagship': [
                "Best phone",
                "Want the best phone available",
                "Need premium flagship device",
                "Looking for top-tier smartphone",
                "Want latest technology and features",
                "Best overall phone",
                "Premium device with all features",
                "Flagship phone with cutting-edge specs",
                "Top-of-the-line smartphone",
                "Want the most advanced phone",
                "Best premium device available",
                "Top phone",
                "Best smartphone"
            ],
            'budget': [
                "Looking for affordable phone",
                "Need cheap but good smartphone",
                "Best value phone under budget",
                "Want budget-friendly device",
                "Affordable phone with good features",
                "Cheap phone that works well",
                "Budget device with decent specs",
                "Best phone for the money",
                "Economical smartphone option",
                "Value for money device"
            ],
            'video': [
                "Want phone for watching movies",
                "Need device for streaming Netflix",
                "Phone for video consumption",
                "Best phone for YouTube and videos",
                "Want great screen for watching content",
                "Phone for media and entertainment",
                "Need device for binge-watching shows",
                "Best phone for streaming services",
                "Want phone for video playback",
                "Device for watching videos and movies"
            ],
            'performance': [
                "Need fast and powerful phone",
                "Want phone with best processor",
                "Need speedy device for multitasking",
                "Phone with excellent performance",
                "Want fastest smartphone available",
                "Need powerful device for heavy apps",
                "Phone with top performance specs",
                "Want phone that never lags",
                "Need high-performance device",
                "Fast phone with smooth operation"
            ],
            'balanced': [
                "Want all-round good phone",
                "Need balanced device for everything",
                "Phone that does everything well",
                "Versatile smartphone for all needs",
                "Balanced phone with no weaknesses",
                "All-purpose device for daily use",
                "Want phone with good overall package",
                "Balanced specs across the board",
                "Jack of all trades smartphone",
                "Well-rounded device for general use"
            ]
        }
        
        # Compute embeddings for use case references
        if self.embedder:
            self.use_case_embeddings = {}
            for use_case, queries in self.use_case_references.items():
                embeddings = self.embedder.encode(queries, convert_to_tensor=True)
                self.use_case_embeddings[use_case] = embeddings
            logger.info(f"✓ Pre-computed embeddings for {len(self.use_case_embeddings)} use cases")
        
        # Semantic spec mappings (natural language → spec type)
        self.spec_synonyms = {
            'ram': [
                'memory', 'RAM', 'gb ram', 'gigs', 'gigabytes ram', 
                'memory capacity', 'ram size', 'system memory'
            ],
            'storage': [
                'storage', 'internal storage', 'space', 'gb storage',
                'memory space', 'storage capacity', 'rom', 'disk space'
            ],
            'battery': [
                'battery', 'mah', 'battery life', 'battery capacity',
                'power', 'battery size', 'endurance', 'battery backup'
            ],
            'camera': [
                'camera', 'megapixel', 'mp camera', 'photo quality',
                'camera quality', 'lens', 'sensor', 'photography'
            ],
            'refresh_rate': [
                'refresh rate', 'hz', 'refresh', 'smooth', 'fast display',
                'high refresh', 'screen smoothness', 'fluid display'
            ],
            'display_size': [
                'screen size', 'display size', 'inch display', 'screen',
                'large screen', 'big screen', 'display dimensions'
            ],
            'processor': [
                'processor', 'chipset', 'cpu', 'snapdragon', 'dimensity',
                'chip', 'soc', 'performance chip', 'processing power'
            ],
            '5g': [
                '5g', '5g network', '5g support', 'fifth generation',
                'next gen network', '5g connectivity'
            ]
        }
        
        # Compute embeddings for spec synonyms
        if self.embedder:
            self.spec_embeddings = {}
            for spec_type, synonyms in self.spec_synonyms.items():
                embeddings = self.embedder.encode(synonyms, convert_to_tensor=True)
                self.spec_embeddings[spec_type] = embeddings
            logger.info(f"✓ Pre-computed embeddings for {len(self.spec_embeddings)} spec types")
        
        # Implicit preference patterns (lifestyle → requirements)
        self.implicit_patterns = {
            'travel': {
                'keywords': ['travel', 'traveling', 'trip', 'vacation', 'journey', 'tourist', 'backpack'],
                'preferences': {
                    'min_battery': 4500,
                    'require_durability': True,
                    'priority': 'battery',
                    'reason': 'Travel lifestyle requires long battery life and durability'
                }
            },
            'creator': {
                'keywords': ['content creator', 'youtube', 'vlogger', 'influencer', 'creator', 
                            'streaming', 'tiktok', 'instagram'],
                'preferences': {
                    'min_camera_mp': 48,
                    'min_storage': 128,
                    'min_battery': 4000,
                    'priority': 'photography',
                    'reason': 'Content creation needs excellent camera, storage, and battery'
                }
            },
            'student': {
                'keywords': ['student', 'college', 'university', 'school', 'studying'],
                'preferences': {
                    'budget_conscious': True,
                    'min_battery': 4000,
                    'priority': 'balanced',
                    'reason': 'Student needs balanced device with good battery on budget'
                }
            },
            'professional': {
                'keywords': ['professional', 'business', 'work', 'office', 'corporate'],
                'preferences': {
                    'priority': 'business',
                    'require_security': True,
                    'reason': 'Professional use requires business features and security'
                }
            },
            'photographer': {
                'keywords': ['photographer', 'photos', 'photography enthusiast'],
                'preferences': {
                    'min_camera_mp': 50,
                    'min_storage': 256,
                    'priority': 'photography',
                    'reason': 'Photography focus requires top camera and storage'
                }
            }
        }
        
        # Brand keywords for fallback extraction
        self.brand_keywords = [
            'apple', 'samsung', 'google', 'oneplus', 'xiaomi', 'oppo', 'vivo',
            'realme', 'motorola', 'nokia', 'sony', 'lg', 'huawei', 'honor',
            'asus', 'lenovo', 'alcatel', 'zte', 'nothing', 'poco', 'redmi'
        ]
        
        logger.info("✓ Semantic NLP Parser initialized successfully")
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Main parsing method - comprehensive semantic analysis
        
        Args:
            query: Natural language query
        
        Returns:
            Dict with extracted preferences including semantic understanding
        """
        
        if not query or not query.strip():
            return {'query': query, 'error': 'Empty query'}
        
        query_lower = query.lower()
        
        # Initialize preferences dict
        preferences = {
            'query': query,
            'device_type': ['mobile'],  # Default
            'brand_preference': [],
            'brand_avoid': [],
            'use_case': '',
            'use_case_confidence': 0.0,
            'multi_intent': [],  # List of (use_case, confidence) tuples
            'budget': None,
            'budget_min': None,
            'budget_max': None,
            'exclusions': [],
            'query_confidence': 0.0,
            'semantic_matches': []
        }
        
        try:
            # 1. Semantic use case detection (with multi-intent support)
            use_cases = self.detect_use_cases_semantic(query)
            if use_cases:
                preferences['multi_intent'] = use_cases
                # Primary use case is highest confidence
                preferences['use_case'] = use_cases[0][0]
                preferences['use_case_confidence'] = use_cases[0][1]
            
            # 2. Extract budget (enhanced regex + NER fallback)
            budget_info = self._extract_budget_enhanced(query_lower)
            preferences.update(budget_info)
            
            # 3. Extract brands (semantic + keyword)
            brands = self._extract_brands_semantic(query_lower)
            preferences['brand_preference'] = brands.get('prefer', [])
            preferences['brand_avoid'] = brands.get('avoid', [])
            
            # 4. Extract exclusions (negations)
            exclusions = self._extract_exclusions(query_lower)
            preferences['exclusions'] = exclusions
            
            # 5. Semantic spec extraction
            specs = self.extract_specs_semantic(query)
            preferences.update(specs)
            
            # 6. Implicit preference inference
            implicit = self._infer_implicit_preferences(query_lower)
            if implicit:
                preferences.update(implicit)
            
            # 7. Extract device type
            device_type = self._extract_device_type(query_lower)
            if device_type:
                preferences['device_type'] = device_type
            
            # 8. Calculate overall query confidence
            preferences['query_confidence'] = self._calculate_query_confidence(preferences)
            
            logger.debug(f"Parsed query: {query[:50]}... → use_case: {preferences['use_case']}, "
                        f"confidence: {preferences['use_case_confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}", exc_info=True)
            preferences['error'] = str(e)
        
        return preferences
    
    def detect_use_cases_semantic(self, query: str, threshold: float = 0.35, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Detect use cases using semantic similarity to reference queries.
        Supports multi-intent detection.
        
        Args:
            query: User query
            threshold: Minimum similarity score to consider (0-1)
            top_k: Maximum number of use cases to return
        
        Returns:
            List of (use_case, confidence) tuples, sorted by confidence
        """
        
        if not self.embedder or not self.use_case_embeddings:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            
            # Calculate similarity to each use case
            similarities = {}
            for use_case, ref_embeddings in self.use_case_embeddings.items():
                # Max similarity across all reference queries for this use case
                sims = util.cos_sim(query_embedding, ref_embeddings)[0]
                max_sim = float(sims.max())
                similarities[use_case] = max_sim
            
            # Sort by similarity and filter by threshold
            sorted_cases = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            filtered_cases = [(uc, score) for uc, score in sorted_cases if score >= threshold]
            
            # Return top_k
            return filtered_cases[:top_k]
        
        except Exception as e:
            logger.error(f"Error in semantic use case detection: {e}")
            return []
    
    def extract_specs_semantic(self, query: str) -> Dict[str, Any]:
        """
        Extract spec requirements using semantic understanding.
        Maps natural language to specific requirements.
        
        Args:
            query: User query
        
        Returns:
            Dict with spec requirements
        """
        
        specs = {}
        query_lower = query.lower()
        
        try:
            # Semantic matching for qualitative descriptions
            if self.embedder and self.spec_embeddings:
                query_embedding = self.embedder.encode(query, convert_to_tensor=True)
                
                # Match to spec types
                for spec_type, ref_embeddings in self.spec_embeddings.items():
                    sims = util.cos_sim(query_embedding, ref_embeddings)[0]
                    max_sim = float(sims.max())
                    
                    # If strong semantic match, look for associated values
                    if max_sim > 0.5:
                        if spec_type not in specs:
                            specs[f'mention_{spec_type}'] = True
            
            # Extract explicit numeric values with enhanced patterns
            
            # RAM
            ram_patterns = [
                r'(\d+)\s*(?:gb|gigs?)\s+(?:of\s+)?(?:ram|memory)',
                r'at\s+least\s+(\d+)\s*gb',
                r'(\d+)\s*gb\s+ram',
                r'(\d+)gb'
            ]
            for pattern in ram_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    specs['min_ram_gb'] = int(match.group(1))
                    break
            
            # Storage
            storage_patterns = [
                r'(\d+)\s*(?:gb|tb)\s+(?:storage|space|rom)',
                r'storage\s+of\s+(\d+)\s*gb',
                r'(\d+)\s*gb\s+internal'
            ]
            for pattern in storage_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    storage_val = int(match.group(1))
                    # Convert TB to GB if needed
                    if 'tb' in match.group(0).lower():
                        storage_val *= 1000
                    specs['min_storage'] = storage_val
                    break
            
            # Battery
            battery_patterns = [
                r'(\d{4,5})\s*mah',
                r'battery\s+(?:of\s+)?(\d{4,5})',
                r'(\d{4,5})\s*ma?h?\s+battery'
            ]
            for pattern in battery_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    specs['min_battery'] = int(match.group(1))
                    break
            
            # Camera
            camera_patterns = [
                r'(\d+)\s*(?:mp|megapixel)',
                r'camera\s+of\s+(\d+)\s*mp',
                r'(\d+)\s*mp\s+camera'
            ]
            for pattern in camera_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    specs['min_camera_mp'] = float(match.group(1))
                    break
            
            # Display size
            display_patterns = [
                r'(\d+\.?\d*)\s*(?:inch|")',
                r'screen\s+size\s+of\s+(\d+\.?\d*)'
            ]
            for pattern in display_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    specs['min_display_size'] = float(match.group(1))
                    break
            
            # 5G requirement
            if re.search(r'\b5g\b', query_lower):
                specs['require_5g'] = True
            
            # Qualitative interpretations using semantic understanding
            # "good battery" → reasonable minimum
            if any(phrase in query_lower for phrase in ['good battery', 'decent battery', 'solid battery']):
                if 'min_battery' not in specs:
                    specs['min_battery'] = 4000
            
            # "excellent battery" / "great battery" → higher minimum
            if any(phrase in query_lower for phrase in ['excellent battery', 'great battery', 
                                                          'amazing battery', 'long battery']):
                if 'min_battery' not in specs:
                    specs['min_battery'] = 4500
            
            # "powerful" → RAM and processor
            if 'powerful' in query_lower or 'high performance' in query_lower:
                if 'min_ram_gb' not in specs:
                    specs['min_ram_gb'] = 8
        
        except Exception as e:
            logger.error(f"Error extracting specs: {e}")
        
        return specs
    
    def _extract_budget_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced budget extraction with range support"""
        
        budget_info = {}
        
        try:
            # Range patterns: "$500-$800", "between 500 and 800"
            range_patterns = [
                r'\$?(\d{3,5})\s*(?:-|to|and)\s*\$?(\d{3,5})',
                r'between\s+\$?(\d{3,5})\s+and\s+\$?(\d{3,5})',
                r'from\s+\$?(\d{3,5})\s+to\s+\$?(\d{3,5})'
            ]
            
            for pattern in range_patterns:
                match = re.search(pattern, query)
                if match:
                    min_budget = int(match.group(1))
                    max_budget = int(match.group(2))
                    budget_info['budget_min'] = min_budget
                    budget_info['budget_max'] = max_budget
                    budget_info['budget'] = max_budget  # Use max as primary budget
                    return budget_info
            
            # Single value patterns
            single_patterns = [
                r'under\s+\$?(\d{3,5})',
                r'below\s+\$?(\d{3,5})',
                r'less\s+than\s+\$?(\d{3,5})',
                r'max\s+\$?(\d{3,5})',
                r'around\s+\$?(\d{3,5})',
                r'about\s+\$?(\d{3,5})',
                r'\$(\d{3,5})\b',
            ]
            
            for pattern in single_patterns:
                match = re.search(pattern, query)
                if match:
                    budget = int(match.group(1))
                    budget_info['budget'] = budget
                    
                    # Set range based on keyword
                    if 'under' in pattern or 'below' in pattern or 'less' in pattern or 'max' in pattern:
                        budget_info['budget_max'] = budget
                    elif 'around' in pattern or 'about' in pattern:
                        budget_info['budget_min'] = int(budget * 0.85)
                        budget_info['budget_max'] = int(budget * 1.15)
                    
                    return budget_info
            
            # Try NER extraction if available
            if self.ner_pipeline:
                entities = self.ner_pipeline(query)
                for entity in entities:
                    if entity['entity_group'] in ['MONEY', 'CARDINAL']:
                        amount_match = re.search(r'(\d{3,5})', entity['word'])
                        if amount_match:
                            budget_info['budget'] = int(amount_match.group(1))
                            return budget_info
        
        except Exception as e:
            logger.error(f"Error extracting budget: {e}")
        
        return budget_info
    
    def _extract_brands_semantic(self, query: str) -> Dict[str, List[str]]:
        """Extract brand preferences and exclusions"""
        
        brands = {'prefer': [], 'avoid': []}
        
        try:
            # Check for negations/exclusions first
            avoid_patterns = [
                r'not\s+(\w+)',
                r'no\s+(\w+)',
                r'without\s+(\w+)',
                r'avoid\s+(\w+)',
                r'anything\s+but\s+(\w+)',
                r'except\s+(\w+)',
                r"don'?t\s+want\s+(\w+)"
            ]
            
            for pattern in avoid_patterns:
                matches = re.finditer(pattern, query)
                for match in matches:
                    brand_candidate = match.group(1).lower()
                    if brand_candidate in self.brand_keywords:
                        brands['avoid'].append(brand_candidate.capitalize())
            
            # Extract preferred brands (simple matching, excluding avoided)
            for brand in self.brand_keywords:
                if brand in query and brand.capitalize() not in brands['avoid']:
                    brands['prefer'].append(brand.capitalize())
            
            # Use NER as fallback for ORG entities
            if self.ner_pipeline:
                entities = self.ner_pipeline(query)
                for entity in entities:
                    if entity['entity_group'] == 'ORG':
                        entity_text = entity['word'].lower().replace('#', '').replace(' ', '')
                        for brand in self.brand_keywords:
                            if brand in entity_text:
                                brand_cap = brand.capitalize()
                                if brand_cap not in brands['prefer'] and brand_cap not in brands['avoid']:
                                    brands['prefer'].append(brand_cap)
        
        except Exception as e:
            logger.error(f"Error extracting brands: {e}")
        
        return brands
    
    def _extract_exclusions(self, query: str) -> List[str]:
        """Extract general exclusions beyond brands"""
        
        exclusions = []
        
        # Exclusion patterns for various features
        exclusion_patterns = {
            r'no\s+(\w+ ?\w*)': lambda m: m.group(1),
            r'without\s+(\w+ ?\w*)': lambda m: m.group(1),
            r'avoid\s+(\w+ ?\w*)': lambda m: m.group(1),
        }
        
        for pattern, extractor in exclusion_patterns.items():
            matches = re.finditer(pattern, query)
            for match in matches:
                exclusion = extractor(match).strip()
                if exclusion and len(exclusion) > 2:
                    exclusions.append(exclusion)
        
        return exclusions
    
    def _infer_implicit_preferences(self, query: str) -> Dict[str, Any]:
        """Infer preferences from lifestyle/usage hints"""
        
        implicit_prefs = {}
        
        try:
            for lifestyle, config in self.implicit_patterns.items():
                # Check if any lifestyle keywords match
                if any(keyword in query for keyword in config['keywords']):
                    # Add the preferences
                    prefs = config['preferences'].copy()
                    reason = prefs.pop('reason', '')
                    
                    # Add reasoning to implicit_prefs
                    if 'implicit_reasoning' not in implicit_prefs:
                        implicit_prefs['implicit_reasoning'] = []
                    implicit_prefs['implicit_reasoning'].append(reason)
                    
                    # Merge preferences (don't override explicit ones)
                    for key, value in prefs.items():
                        if key not in implicit_prefs or key == 'priority':
                            implicit_prefs[key] = value
                    
                    logger.debug(f"Detected implicit lifestyle: {lifestyle}")
        
        except Exception as e:
            logger.error(f"Error inferring implicit preferences: {e}")
        
        return implicit_prefs
    
    def _extract_device_type(self, query: str) -> List[str]:
        """Extract device type from query"""
        
        device_types = []
        
        if any(kw in query for kw in ['tablet', 'ipad', 'tab ']):
            device_types.append('tablet')
        elif any(kw in query for kw in ['watch', 'smartwatch', 'band', 'wearable']):
            device_types.append('smartwatch')
        elif any(kw in query for kw in ['phone', 'mobile', 'smartphone']):
            device_types.append('mobile')
        
        # Default to mobile if not specified
        if not device_types:
            device_types = ['mobile']
        
        return device_types
    
    def _calculate_query_confidence(self, preferences: Dict[str, Any]) -> float:
        """
        Calculate overall confidence in query understanding.
        Based on how much information was extracted.
        """
        
        confidence = 0.0
        total_weight = 0.0
        
        # Weights for different components
        weights = {
            'use_case_confidence': 0.3,
            'budget': 0.2,
            'brand_preference': 0.15,
            'specs': 0.25,
            'implicit': 0.1
        }
        
        # Use case confidence
        if preferences.get('use_case_confidence', 0) > 0:
            confidence += preferences['use_case_confidence'] * weights['use_case_confidence']
        total_weight += weights['use_case_confidence']
        
        # Budget
        if preferences.get('budget') is not None:
            confidence += 1.0 * weights['budget']
        total_weight += weights['budget']
        
        # Brand preference
        if preferences.get('brand_preference'):
            confidence += 1.0 * weights['brand_preference']
        total_weight += weights['brand_preference']
        
        # Specs (count how many were extracted)
        spec_keys = ['min_ram_gb', 'min_battery', 'min_camera_mp', 'min_refresh_hz', 
                     'min_storage', 'require_5g']
        specs_found = sum(1 for key in spec_keys if preferences.get(key) is not None)
        if specs_found > 0:
            confidence += min(1.0, specs_found / 3.0) * weights['specs']
        total_weight += weights['specs']
        
        # Implicit preferences
        if preferences.get('implicit_reasoning'):
            confidence += 1.0 * weights['implicit']
        total_weight += weights['implicit']
        
        # Normalize
        if total_weight > 0:
            confidence = confidence / total_weight
        
        return min(1.0, confidence)


# Singleton instance for import
_semantic_parser_instance = None

def get_semantic_parser() -> SemanticNLPParser:
    """Get or create singleton instance of semantic parser"""
    global _semantic_parser_instance
    if _semantic_parser_instance is None:
        _semantic_parser_instance = SemanticNLPParser()
    return _semantic_parser_instance
