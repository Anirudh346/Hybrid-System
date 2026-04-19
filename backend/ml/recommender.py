"""
ENHANCED RECOMMENDER ENGINE
Implements Priority 1-4 Optimizations:
- Priority 1: Zero-spec filtering, score normalization, gaming bias fix
- Priority 2: Hybrid scoring, semantic embeddings, LTR, MCDM
- Priority 3: Data imputation, popularity tracking, feature importance
- Priority 4: Improved NLP, negation support, confidence scoring
- NEW: Full semantic NLP with BERT embeddings
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix, spmatrix, issparse
from typing import List, Dict, Any, Tuple, Optional
import re
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Chipset tier mapping for performance-aware scoring
CHIPSET_TIER_PATTERNS = {
    'apple': {
        'top': [
            r'\ba18 pro\b', r'\ba18\b', r'\ba17 pro\b', r'\bm4\b', r'\bm3\b', r'\bm2\b', r'\bm1\b',
            r'\ba16 bionic\b', r'\ba15 bionic\b', r'\ba14 bionic\b', r'\ba13 bionic\b',
            r'\ba12x bionic\b', r'\ba12z bionic\b', r'\ba12 bionic\b'
        ],
        'mid': [
            r'\ba15\b', r'\ba14\b', r'\ba13\b', r'\ba12\b', r'\ba11 bionic\b'
        ],
        'budget': [
            r'\ba10 fusion\b', r'\ba10\b', r'\ba9\b', r'\ba8\b'
        ]
    },
    'qualcomm': {
        'top': [
            r'\b8 elite\b', r'\b8 gen 4\b', r'\b8 gen 3\b', r'\b8\+ gen 2\b', r'\b8 gen 2\b',
            r'\b8\+ gen 1\b', r'\b8 gen 1\b', r'\b888\+\b', r'\b888\b', r'\b870\b', r'\b865\+\b',
            r'\b865\b', r'\b855\+\b', r'\b855\b', r'\b845\b',
            r'\bsm8750\b', r'\bsm8650\b', r'\bsm8475\b', r'\bsm8450\b', r'\bsm8350\b',
            r'\bsm8250(?:-ac)?\b', r'\bsm8150(?:-ac)?\b', r'\bsdm845\b'
        ],
        'mid': [
            r'\b7s gen 3\b', r'\b7 gen 3\b', r'\b7\+ gen 2\b', r'\b7 gen 2\b', r'\b7\+ gen 1\b',
            r'\b7 gen 1\b', r'\b780g\b', r'\b778g\+\b', r'\b778g\b', r'\b765g\b', r'\b750g\b',
            r'\b732g\b', r'\b730g\b', r'\b720g\b', r'\b710\b',
            r'\bsm7635\b', r'\bsm7550\b', r'\bsm7475\b', r'\bsm7450\b', r'\bsm7350\b',
            r'\bsm7325(?:-ae)?\b', r'\bsm7250(?:-ab)?\b', r'\bsm7150(?:-ac|-aa)?\b', r'\bsm7125\b'
        ],
        'budget': [
            r'\b6 gen 3\b', r'\b6 gen 1\b', r'\b4 gen 2\b', r'\b4 gen 1\b',
            r'\b695\b', r'\b680\b', r'\b665\b', r'\b662\b', r'\b460\b', r'\b450\b',
            r'\bsm6475\b', r'\bsm6375\b', r'\bsm6370\b', r'\bsm6350\b', r'\bsm6225\b',
            r'\bsm6125\b', r'\bsm6115\b', r'\bsm4350\b', r'\bsdm450\b'
        ]
    },
    'samsung': {
        'top': [
            r'\bexynos 2500\b', r'\bexynos 2400\b', r'\bexynos 2300\b', r'\bexynos 2200\b',
            r'\bexynos 2100\b', r'\bexynos 990\b', r'\bexynos 9825\b', r'\bexynos 9820\b'
        ],
        'mid': [
            r'\bexynos 1580\b', r'\bexynos 1480\b', r'\bexynos 1380\b', r'\bexynos 1330\b',
            r'\bexynos 1280\b', r'\bexynos 1080\b', r'\bexynos 980\b'
        ],
        'budget': [
            r'\bexynos 9611\b', r'\bexynos 850\b', r'\bexynos 7904\b', r'\bexynos 7885\b'
        ]
    },
    'mediatek': {
        'top': [
            r'\bdimensity 9400\b', r'\bdimensity 9300\+?\b', r'\bdimensity 9200\+?\b',
            r'\bdimensity 9000\+?\b', r'\bdimensity 1300\b', r'\bdimensity 1200\b',
            r'\bmt6991t?\b', r'\bmt6989\b', r'\bmt6985t?\b', r'\bmt6895t?\b', r'\bmt6893\b'
        ],
        'mid': [
            r'\bdimensity 8300\b', r'\bdimensity 8200\b', r'\bdimensity 8100\b', r'\bdimensity 8050\b',
            r'\bdimensity 1100\b', r'\bdimensity 920\b', r'\bdimensity 810\b', r'\bdimensity 800u\b',
            r'\bdimensity 700\b', r'\bmt6897\b', r'\bmt6896\b', r'\bmt6895z\b', r'\bmt6891\b',
            r'\bmt6877v\b', r'\bmt6833\b', r'\bmt6853\b'
        ],
        'budget': [
            r'\bdimensity 6300\b', r'\bdimensity 6100\+\b', r'\bdimensity 6020\b',
            r'\bhelio g99\b', r'\bhelio g96\b', r'\bhelio g95\b', r'\bhelio g90t\b',
            r'\bhelio g88\b', r'\bhelio g85\b', r'\bhelio g80\b', r'\bhelio p70\b',
            r'\bhelio p65\b', r'\bhelio p60\b', r'\bhelio p35\b', r'\bhelio p22\b',
            r'\bhelio a22\b', r'\bmt6855\b', r'\bmt6853t\b', r'\bmt6789\b', r'\bmt6781\b',
            r'\bmt6769t\b', r'\bmt6769\b', r'\bmt6771\b', r'\bmt6765g\b', r'\bmt6765\b',
            r'\bmt6762\b', r'\bmt6761\b', r'\bmt8768\b', r'\bmt8766\b', r'\bmt8765\b',
            r'\bmt8762\b', r'\bmt8735\b'
        ]
    },
    'google': {
        'top': [
            r'\btensor g4\b', r'\btensor g3\b', r'\btensor g2\b', r'\btensor g1\b',
            r'\bgs301\b', r'\bgs201\b', r'\bgs101\b'
        ],
        'mid': [],
        'budget': []
    }
}

CHIPSET_TIER_SCORES = {'top': 1.0, 'mid': 0.6, 'budget': 0.3}

# Import semantic parser
try:
    from ml.semantic_nlp_parser import SemanticNLPParser
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("Semantic NLP not available - install sentence-transformers")

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ============================================================================
# PRIORITY 3.1: SMART DATA IMPUTATION
# ============================================================================

class DataImputer:
    """Smart imputation of missing device specs"""
    
    @staticmethod
    def impute_missing_specs(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligently fill missing critical specs"""
        
        imputed = [d.copy() for d in devices]
        
        # Group devices by brand for brand-specific imputation
        brand_groups = {}
        for device in imputed:
            brand = device.get('brand', 'Unknown')
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(device)
        
        for device in imputed:
            specs = device.get('specs', {})
            brand = device.get('brand', 'Unknown')
            display_size = specs.get('display_size_inches', 6.0)
            
            # Impute RAM
            if specs.get('ram_gb', 0) == 0:
                similar_devices = [d for d in brand_groups.get(brand, [])
                                  if d.get('specs', {}).get('ram_gb', 0) > 0]
                if similar_devices:
                    specs['ram_gb'] = int(np.median([d['specs']['ram_gb'] for d in similar_devices]))
                else:
                    specs['ram_gb'] = 4
            
            # Impute Battery
            if specs.get('battery_mah', 0) == 0:
                similar_devices = [d for d in imputed
                                  if abs(d.get('specs', {}).get('display_size_inches', 6.0) - display_size) < 0.5
                                  and d.get('specs', {}).get('battery_mah', 0) > 0]
                if similar_devices:
                    specs['battery_mah'] = int(np.median([d['specs']['battery_mah'] for d in similar_devices]))
                else:
                    specs['battery_mah'] = 4000
            
            # Impute Price
            if specs.get('price', 0) == 0:
                similar_devices = [d for d in brand_groups.get(brand, [])
                                  if d.get('specs', {}).get('price', 0) > 0]
                if similar_devices:
                    specs['price'] = int(np.median([d['specs']['price'] for d in similar_devices]))
                else:
                    specs['price'] = 300
            
            # Impute Camera
            if specs.get('main_camera_mp', 0) == 0:
                similar_devices = [d for d in brand_groups.get(brand, [])
                                  if d.get('specs', {}).get('main_camera_mp', 0) > 0]
                if similar_devices:
                    specs['main_camera_mp'] = np.median([d['specs']['main_camera_mp'] for d in similar_devices])
                else:
                    specs['main_camera_mp'] = 13
            
            device['specs'] = specs
        
        return imputed


# ============================================================================
# PRIORITY 4.1 & 4.3: ENHANCED NLP PARSER
# ============================================================================

class EnhancedNLPParser:
    """Improved NLP with use-case confidence and query understanding"""
    
    def __init__(self):
        self.use_case_keywords = {
            'flagship': ['flagship', 'premium', 'high-end', 'top-end', 'best available', 'upgrade', 'replace my'],
            'photography': ['camera', 'photo', 'picture', 'selfie', 'zoom', 'macro',
                           'portrait', 'night', 'professional', 'photographer', 'shoot',
                           'photography', 'megapixel', 'mp', 'lens', 'optical'],
            'gaming': ['gaming', 'game', 'fps', 'gpu', 'processor', 'snapdragon', 'powerful', 'play', 'esports'],
            'battery': ['battery', 'endurance', 'day', 'hours', 'charge', 'travel',
                       'long', 'last', 'power', 'lasting', 'juice', '5000mah', '6000mah'],
            'display': ['screen', 'display', 'oled', 'amoled', 'ips', 'brightness', 'color', 'hdr'],
            'budget': ['cheap', 'affordable', 'budget', 'low-cost', 'under', 'less than',
                      'inexpensive', 'bargain', 'limited budget'],
            'video': ['video', 'recording', 'content', '4k', '8k', 'stabilization', 'streaming'],
            'performance': ['performance', 'fast', 'speed', 'quick', 'responsive', 'processor'],
            'business': ['business', 'professional', 'work', 'productivity', 'security', 'privacy', 'encryption', 'updates'],
        }

        self.durability_keywords = ['durable', 'rugged', 'dust', 'water', 'ip68', 'ip67', 'mil-std', 'tough']
        self.network_keywords = ['poor network', 'weak signal', 'fallback', '3g', '4g', 'coverage', 'unreliable network']
        
        self.negation_patterns = [
            r'(?:not|no|exclude|avoid|without|skip)\s+([a-zA-Z]+)',
            r'(?:anything|any) (?:but|except)\s+([a-zA-Z]+)',
            r'(?:don\'t|doesnt?|won\'t|can\'t)\s+(?:want|use|have)\s+([a-zA-Z]+)',
        ]

    def parse_constraints(self, query: str) -> Dict[str, Any]:
        """Extract hard constraints like budget ranges, battery minimums, durability and network hints"""

        q = query.lower()
        constraints: Dict[str, Any] = {
            'budget_min': None,
            'budget_max': None,
            'min_battery': None,
            'min_storage': None,
            'min_ram_gb': None,
            'require_durability': False,
            'require_network_resilience': False,
        }

        # Budget range: "$300-$500", "between 300 and 500", "from 300 to 500"
        range_patterns = [
            r'\$?(\d+[,.]?\d*)\s*[-toand]{1,3}\s*\$?(\d+[,.]?\d*)',
            r'between\s+\$?(\d+[,.]?\d*)\s+and\s+\$?(\d+[,.]?\d*)',
            r'from\s+\$?(\d+[,.]?\d*)\s+to\s+\$?(\d+[,.]?\d*)',
        ]
        for pat in range_patterns:
            m = re.search(pat, q)
            if m:
                low = float(m.group(1).replace(',', ''))
                high = float(m.group(2).replace(',', ''))
                if low > high:
                    low, high = high, low
                constraints['budget_min'] = low
                constraints['budget_max'] = high
                break

        # Single budget upper bound
        single_budget = re.search(r'(?:under|below|less than|max)\s+\$?(\d+[,.]?\d*)', q)
        if single_budget and constraints['budget_max'] is None:
            constraints['budget_max'] = float(single_budget.group(1).replace(',', ''))

        # Battery minimums, e.g., "6000mah" or "6000 mAh"
        battery_match = re.search(r'(\d{4,5})\s*mah', q)
        if battery_match:
            constraints['min_battery'] = max(int(battery_match.group(1)), constraints['min_battery'] or 0)
        elif 'big battery' in q or 'long battery' in q or 'lasting battery' in q:
            constraints['min_battery'] = 5000

        # Storage minimums, e.g., "128GB" or "256 gb"
        storage_match = re.search(r'(\d{2,4})\s*gb\s*storage', q)
        if storage_match:
            constraints['min_storage'] = int(storage_match.group(1))
        else:
            storage_match = re.search(r'(\d{2,4})\s*gb', q)
            if storage_match and int(storage_match.group(1)) >= 64:
                constraints['min_storage'] = int(storage_match.group(1))

        # RAM minimums - check for explicit "RAM" keyword first, e.g., "8GB RAM", "12 gb ram", "at least 8GB"
        # Try RAM-specific patterns first
        ram_match = re.search(r'(?:at least|minimum|min|need|require)?\s*(\d{1,2})\s*(?:gb|g)?\s*(?:ram|memory|of ram)', q, re.IGNORECASE)
        if not ram_match:
            # Fallback to generic pattern with "GB RAM"
            ram_match = re.search(r'(\d{1,2})\s*gb\s*ram', q, re.IGNORECASE)
        if ram_match and int(ram_match.group(1)) >= 4:
            constraints['min_ram_gb'] = int(ram_match.group(1))

        # Durability / rugged hints
        if any(kw in q for kw in self.durability_keywords):
            constraints['require_durability'] = True

        # Network reliability / fallback hints
        if any(kw in q for kw in self.network_keywords):
            constraints['require_network_resilience'] = True

        return constraints
    
    def detect_use_case(self, query: str) -> Tuple[str, float]:
        """Detect use case with confidence score"""
        
        query_lower = query.lower()
        scores = {}
        
        # Count keyword matches with weighted scoring
        for use_case, keywords in self.use_case_keywords.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            
            # Apply priority boost for specific use cases with strong indicators
            if use_case == 'photography' and ('photographer' in query_lower or 'photo editing' in query_lower):
                matches += 3
            elif use_case == 'flagship' and ('flagship' in query_lower or 'premium' in query_lower):
                matches += 3
            elif use_case == 'gaming' and 'gam' in query_lower:
                # Downgrade casual/basic gaming to balanced
                if 'basic gaming' in query_lower or 'casual gaming' in query_lower:
                    matches = 0
                else:
                    matches += 2
            elif use_case == 'business' and ('privacy' in query_lower or 'security' in query_lower):
                matches += 3  # Strong boost for privacy/security/business keywords
            
            scores[use_case] = matches
        
        max_score = max(scores.values()) if scores else 0
        
        if max_score >= 2:
            use_case = max(scores, key=lambda x: scores[x])
            # More lenient confidence calculation
            confidence = min(max_score / 4.0, 1.0)
            return use_case, confidence
        else:
            return 'balanced', 0.3
    
    def parse_exclusions(self, query: str) -> List[str]:
        """Extract brand/model exclusions (e.g., 'not Samsung')"""
        
        exclusions = []
        
        for pattern in self.negation_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            exclusions.extend(matches)
        
        # Drop very short/stop words (e.g., 'too', 'no', 'not', 'any')
        stop_words = {'no', 'not', 'any', 'too', 'issues', 'issue', 'problem', 'problems', 'lag'}
        cleaned = [e.strip() for e in exclusions if e.strip() and e.strip().lower() not in stop_words and len(e.strip()) > 2]
        return cleaned
    
    def calculate_query_confidence(self, parsed_query: Dict[str, Any]) -> float:
        """Calculate overall query parsing confidence (0-1)"""
        
        confidence = 0.0
        
        if parsed_query.get('budget') or parsed_query.get('budget_max'):
            confidence += 0.25
        
        # Lowered threshold from > 0.6 to >= 0.5
        if parsed_query.get('use_case_confidence', 0) >= 0.5:
            confidence += 0.25
        
        if parsed_query.get('brand_preference'):
            confidence += 0.15
        
        if parsed_query.get('device_type'):
            confidence += 0.15
        
        # Count explicit requirements as "required features"
        feature_count = 0
        if parsed_query.get('min_ram_gb'):
            feature_count += 1
        if parsed_query.get('min_battery'):
            feature_count += 1
        if parsed_query.get('require_5g'):
            feature_count += 1
        if feature_count > 0:
            confidence += min(0.2, feature_count * 0.1)
        
        return min(confidence, 1.0)


# ============================================================================
# PRIORITY 2.1: HYBRID RECOMMENDER
# ============================================================================

class HybridRecommender:
    """Hybrid scoring: Content (TF-IDF) + Popularity + Collaborative"""
    
    def __init__(self):
        self.content_scores = {}
        self.popularity_scores = {}
        self.collaborative_scores = {}
        self.weights = {
            'content': 0.6,
            'popularity': 0.2,
            'collaborative': 0.2
        }
    
    def calculate_hybrid_score(self, device_id: str, 
                              content_score: float,
                              popularity_score: float = 0.5,
                              collaborative_score: float = 0.5) -> float:
        """Combine scores using weighted hybrid approach"""
        
        hybrid = (
            self.weights['content'] * content_score +
            self.weights['popularity'] * popularity_score +
            self.weights['collaborative'] * collaborative_score
        )
        
        return min(hybrid, 1.0)


# ============================================================================
# PRIORITY 2.4: MULTI-CRITERIA DECISION MAKING (TOPSIS)
# ============================================================================

class MCDMRecommender:
    """Multi-Criteria Decision Making using TOPSIS"""
    
    @staticmethod
    def normalize_criteria(criteria_matrix: np.ndarray) -> np.ndarray:
        """Normalize all criteria to 0-1 scale"""
        
        normalized = np.zeros_like(criteria_matrix, dtype=float)
        
        for col in range(criteria_matrix.shape[1]):
            col_data = criteria_matrix[:, col].astype(float)
            col_min = np.min(col_data)
            col_max = np.max(col_data)
            
            if col_max - col_min > 0:
                normalized[:, col] = (col_data - col_min) / (col_max - col_min)
            else:
                normalized[:, col] = 0.5
        
        return normalized
    
    @staticmethod
    def calculate_topsis_scores(criteria_matrix: np.ndarray,
                               weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate TOPSIS scores for multi-criteria decision making"""
        
        normalized = MCDMRecommender.normalize_criteria(criteria_matrix)
        
        if weights is not None:
            normalized = normalized * weights
        
        ideal = np.max(normalized, axis=0)
        anti_ideal = np.min(normalized, axis=0)
        
        scores = []
        for i in range(len(normalized)):
            d_pos = euclidean(normalized[i], ideal)
            d_neg = euclidean(normalized[i], anti_ideal)
            
            topsis_score = d_neg / (d_pos + d_neg) if (d_pos + d_neg) > 0 else 0.5
            scores.append(topsis_score)
        
        return np.array(scores)


# ============================================================================
# PRIORITY 2.2: LEARNING-TO-RANK (LTR) SYSTEM
# ============================================================================

class LTRRanker:
    """Learning-to-Rank using Gradient Boosting"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, device: Dict[str, Any], 
                        preferences: Dict[str, Any]) -> List[float]:
        """Extract features for LTR model"""
        
        specs = device.get('specs', {})
        
        features = [
            specs.get('ram_gb', 0),
            specs.get('storage_gb', 0),
            specs.get('main_camera_mp', 0),
            specs.get('battery_mah', 0),
            specs.get('price', 0) or 300,
            float(specs.get('has_5g', False)),
            float(specs.get('has_nfc', False)),
            float(specs.get('has_wireless_charging', False)),
            float(specs.get('has_fast_charging', False)),
            float(device.get('brand', '').lower() in str(preferences.get('brand_preference', '')).lower()),
            float('gaming' in preferences.get('use_case', '').lower()),
            float('photography' in preferences.get('use_case', '').lower()),
            float('battery' in preferences.get('use_case', '').lower()),
        ]
        
        return features
    
    def train(self, devices: List[Dict[str, Any]], 
              preferences_list: List[Dict[str, Any]],
              ratings: List[float]):
        """Train LTR model (requires user ratings)"""
        
        X = []
        y = ratings
        
        for device, prefs in zip(devices, preferences_list):
            features = self.extract_features(device, prefs)
            X.append(features)
        
        if X and len(X) >= 10:
            X_array = np.array(X)
            self.model.fit(X_array, np.array(y))
            self.is_trained = True
    
    def rank_candidates(self, candidates: List[Dict[str, Any]],
                       preferences: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank candidates using trained model"""
        
        if not self.is_trained:
            return []
        
        X = []
        for device in candidates:
            features = self.extract_features(device, preferences)
            X.append(features)
        
        X_array = np.array(X)
        scores = self.model.predict(X_array)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        ranked = [(candidates[i].get('id', ''), float(scores[i])) 
                 for i in range(len(candidates))]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)


# ============================================================================
# PRIORITY 1 & 2: ENHANCED DEVICE RECOMMENDER
# ============================================================================

class DeviceRecommender:
    """
    Enhanced recommendation engine with Priority 1-4 optimizations
    
    Improvements:
    - Priority 1.1: Filter devices with >2 missing critical specs
    - Priority 1.2: Normalize scores to 0-1 range
    - Priority 1.3: Reduce gaming boost default based on confidence
    - Priority 2.1: Hybrid scoring (content + popularity)
    - Priority 2.4: TOPSIS for multi-criteria
    - Priority 3.1: Smart data imputation
    - Priority 4.1: Improved NLP with confidence
    - Priority 4.2: Negation/exclusion support
    - Priority 4.3: Query confidence scoring
    """
    
    def __init__(self, use_semantic: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.device_features = None
        self.feature_matrix = None
        self.device_ids = []
        self.raw_devices = []
        self.nlp_parser = EnhancedNLPParser()
        self.hybrid_scorer = HybridRecommender()
        self.ltr_ranker = LTRRanker()
        self.imputer = DataImputer()
        
        # Feature importance weights (XAI-style)
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
        
        # Semantic components
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.semantic_parser = None
        self.semantic_embedder = None
        self.device_embeddings = None
        
        if self.use_semantic:
            try:
                self.semantic_parser = SemanticNLPParser()
                self.semantic_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Semantic NLP enabled")
            except Exception as e:
                logger.warning(f"Semantic NLP initialization failed: {e}. Falling back to legacy NLP.")
                self.use_semantic = False
    
    def _count_missing_specs(self, device: Dict[str, Any]) -> int:
        """Count how many critical specs are missing"""
        
        specs = device.get('specs', {})
        missing = 0
        
        critical_specs = ['ram_gb', 'battery_mah', 'main_camera_mp', 'price']
        
        for spec in critical_specs:
            if specs.get(spec, 0) == 0:
                missing += 1
        
        return missing

    def _get_chipset_text(self, specs: Dict[str, Any]) -> str:
        """Get chipset text from normalized specs"""

        if not isinstance(specs, dict):
            return ""
        return str(specs.get('chipset') or specs.get('Chipset') or "")

    def _get_chipset_tier(self, chipset_text: str) -> Optional[str]:
        """Return chipset tier label based on known patterns"""

        if not chipset_text:
            return None

        text = chipset_text.lower()
        for company, tiers in CHIPSET_TIER_PATTERNS.items():
            for tier, patterns in tiers.items():
                for pat in patterns:
                    if re.search(pat, text):
                        return tier
        return None

    def _chipset_tier_score(self, chipset_text: str) -> float:
        """Return normalized chipset tier score"""

        tier = self._get_chipset_tier(chipset_text)
        return CHIPSET_TIER_SCORES.get(tier or '', 0.0)
    
    def _calculate_gaming_score(self, specs: Dict[str, Any]) -> float:
        """Calculate gaming capability score: 60% chipset tier + 40% RAM
        
        Returns a 0-1 score indicating gaming performance capability.
        High-end chipsets with 8GB+ RAM get scores >= 0.8
        Mid-tier chipsets with 8GB+ RAM get scores ~0.5-0.6
        """
        if not isinstance(specs, dict):
            return 0.0
        
        # Get chipset tier score (0-1, where top=1.0, mid=0.6, budget=0.3)
        chipset_text = self._get_chipset_text(specs)
        chipset_score = self._chipset_tier_score(chipset_text)
        
        # Get RAM normalized score (0-1, where 16GB = 1.0)
        ram_gb = float(specs.get('ram_gb', 0) or 0)
        ram_score = min(ram_gb / 16.0, 1.0)
        
        # Weighted combination: 60% chipset + 40% RAM
        gaming_score = (chipset_score * 0.6) + (ram_score * 0.4)
        
        return float(gaming_score)
    
    def _create_feature_text(self, device: Dict[str, Any]) -> str:
        """Create searchable feature text from device specs"""
        
        specs = device.get('specs', {})
        features = []
        
        # Brand and model (high weight)
        features.extend([device.get('brand', '')] * 3)
        features.extend([device.get('model_name', '')] * 2)
        
        # Device type
        features.extend([device.get('device_type', '')] * 2)
        
        # Key specifications
        important_specs = [
            'OS', 'Chipset', 'chipset', 'CPU', 'GPU',
            'Display', 'Type', 'Size',
            'Internal', 'Card slot',
            'Main Camera', 'Selfie camera',
            'Battery', 'Charging',
            'WLAN', 'Bluetooth', 'NFC', 'USB'
        ]
        
        for spec in important_specs:
            if spec in specs and specs[spec]:
                features.append(str(specs[spec]))
        
        return ' '.join(features)
    
    def _create_device_description(self, device: Dict[str, Any]) -> str:
        """Create natural language description for semantic embedding"""
        
        specs = device.get('specs', {})
        brand = device.get('brand', '')
        model = device.get('model_name', '')
        
        # Extract key specs
        ram = specs.get('ram_gb', 0)
        storage = specs.get('storage_gb', 0)
        battery = specs.get('battery_mah', 0)
        camera = specs.get('main_camera_mp', 0)
        processor = specs.get('Chipset', '')
        display_size = specs.get('display_size_inches', 0)
        os = specs.get('OS', '')
        
        # Build natural description
        parts = [f"{brand} {model}"]
        
        if ram:
            parts.append(f"{ram}GB RAM")
        if storage:
            parts.append(f"{storage}GB storage")
        if battery:
            parts.append(f"{battery}mAh battery")
        if camera:
            parts.append(f"{camera}MP camera")
        if display_size:
            parts.append(f"{display_size} inch screen")
        if processor:
            parts.append(processor)
        if os:
            parts.append(os)
        
        # Add use case hints based on specs
        if ram >= 12:
            parts.append("excellent for gaming")
        if camera >= 50:
            parts.append("great for photography")
        if battery >= 5000:
            parts.append("long battery life")
        
        return " ".join(parts)
    
    def _extract_price(self, device: Dict[str, Any]) -> float:
        """Extract numeric price"""
        
        specs = device.get('specs', {})
        price = specs.get('price', 0)
        
        if isinstance(price, (int, float)):
            return float(price)
        
        price_str = str(price)
        price_match = re.search(r'(\d+(?:,\d+)?(?:\.\d+)?)', price_str)
        if price_match:
            price_val = price_match.group(1).replace(',', '')
            return float(price_val)
        
        return 0.0
    
    def _extract_numeric_value(self, text: str) -> float:
        """Extract first numeric value from string"""
        if not text:
            return 0.0
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return float(match.group(1)) if match else 0.0

    def _infer_brand_preferences_from_query(self, query: str) -> List[str]:
        """Infer brand preference from free-text query when parser misses it."""
        if not query:
            return []

        query_lower = str(query).lower()
        aliases = {
            'apple': ['apple', 'iphone'],
            'samsung': ['samsung', 'galaxy'],
            'google': ['google', 'pixel'],
            'oneplus': ['oneplus', 'one plus'],
            'xiaomi': ['xiaomi', ' mi '],
            'motorola': ['motorola', ' moto '],
            'oppo': ['oppo'],
            'vivo': ['vivo'],
            'realme': ['realme'],
            'nothing': ['nothing'],
            'nokia': ['nokia']
        }

        inferred = []
        for canonical, tokens in aliases.items():
            if any(token in query_lower for token in tokens):
                inferred.append(canonical)

        return inferred
    
    def _evaluate_use_case_specs(self, specs: Dict[str, Any], use_case: str) -> Dict[str, float]:
        """Evaluate how well specs match a use case (XAI-style)"""
        
        scores = {}
        
        if use_case == 'gaming':
            # Chipset evaluation (40% weight for gaming)
            chipset = str(specs.get('chipset', '') or specs.get('Chipset', '')).lower()
            if any(kw in chipset for kw in ['snapdragon 8', 'dimensity 9', 'a17', 'a16']):
                scores['Chipset'] = 1.0
            elif any(kw in chipset for kw in ['snapdragon 7', 'dimensity 7', 'a15']):
                scores['Chipset'] = 0.7
            elif any(kw in chipset for kw in ['snapdragon']):
                scores['Chipset'] = 0.5
            else:
                scores['Chipset'] = 0.2  # Penalize unknown chipsets more for gaming
            
            # RAM evaluation (25% weight for gaming)
            ram = float(specs.get('ram_gb', 0) or 0)
            if ram <= 0:
                ram = self._extract_numeric_value(specs.get('Internal', ''))
            if ram >= 12:
                scores['RAM'] = 1.0
            elif ram >= 8:
                scores['RAM'] = 0.7
            elif ram >= 6:
                scores['RAM'] = 0.4
            else:
                scores['RAM'] = 0.2
            
            # Display type (weight reduced since refresh rate not available)
            display = str(specs.get('display', '') or specs.get('Display', '')).lower()
            if 'amoled' in display or 'oled' in display:
                scores['Display'] = 0.8
            elif 'ips' in display or 'lcd' in display:
                scores['Display'] = 0.5
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
        """Evaluate overall quality of specifications (XAI-style)"""
        
        quality_indicators = []
        
        # Check for premium features
        chipset = str(specs.get('chipset', '') or specs.get('Chipset', '')).lower()
        if any(kw in chipset for kw in ['snapdragon 8', 'dimensity 9', 'a17', 'a16', 'a15']):
            quality_indicators.append(1.0)
        elif any(kw in chipset for kw in ['snapdragon 7', 'dimensity 7']):
            quality_indicators.append(0.7)
        else:
            quality_indicators.append(0.4)
        
        # RAM
        ram = float(specs.get('ram_gb', 0) or 0)
        if ram <= 0:
            ram = self._extract_numeric_value(specs.get('Internal', ''))
        quality_indicators.append(min(1.0, ram / 12))
        
        # Camera
        camera_mp = float(specs.get('main_camera_mp', 0) or 0)
        if camera_mp <= 0:
            camera_mp = self._extract_numeric_value(specs.get('Main Camera', '') or specs.get('Camera', ''))
        quality_indicators.append(min(1.0, camera_mp / 64))
        
        # Battery
        battery_mah = float(specs.get('battery_mah', 0) or 0)
        if battery_mah <= 0:
            battery_mah = self._extract_numeric_value(specs.get('Battery', '') or specs.get('Type_1', ''))
        quality_indicators.append(min(1.0, battery_mah / 5000))
        
        return float(np.mean(quality_indicators)) if quality_indicators else 0.5
    
    def _calculate_feature_contributions(self, device: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted feature contributions for scoring (XAI-inspired)"""
        
        contributions = {}
        specs = device.get('specs', {})
        device_brand = str(device.get('brand', '')).lower().strip()
        device_model = str(device.get('model_name', '')).lower().strip()

        # Prefer explicit brand field; only fall back to model text when brand is missing/unknown
        is_brand_missing = (not device_brand) or device_brand in ['unknown', 'n/a', 'nan']
        device_brand_text = device_brand if not is_brand_missing else f"{device_brand} {device_model}".strip()
        
        # 1. Brand Match - Check explicit preference AND implicit brand in query
        brand_score = 0.0
        
        # Check explicit brand preference
        if 'brand_preference' in preferences and preferences['brand_preference']:
            raw_brand_prefs = [b.lower().strip() for b in (preferences['brand_preference'] if isinstance(preferences['brand_preference'], list) else [preferences['brand_preference']])]
            brand_tokens = []
            for pref in raw_brand_prefs:
                brand_tokens.append(pref)
                if pref == 'samsung':
                    brand_tokens.append('galaxy')
                elif pref == 'apple':
                    brand_tokens.append('iphone')
                elif pref == 'google':
                    brand_tokens.append('pixel')
            brand_score = 1.0 if any(device_brand_text.startswith(token) for token in brand_tokens) else 0.0
        
        # Check implicit brand in query if not explicitly set
        if brand_score == 0 and 'query' in preferences:
            query_lower = preferences['query'].lower()
            for brand_keyword in ['samsung', 'iphone', 'apple', 'oneplus', 'google', 'pixel', 'xiaomi', 'motorola', 'oppo', 'vivo']:
                if brand_keyword in query_lower and brand_keyword in device_brand_text:
                    brand_score = 1.0
                    break
        
        contributions['brand_match'] = brand_score
        
        # 2. Price Fit
        price_score = 0.5  # Default neutral
        if 'budget' in preferences and preferences['budget']:
            budget = preferences['budget']
            price = self._extract_price(device)
            if price > 0 and budget > 0:
                price_score = max(0, 1 - abs(price - budget) / budget)
        contributions['price_fit'] = price_score
        
        # 3. Use Case Alignment
        use_case_score = 0.5  # Default
        if 'use_case' in preferences and preferences['use_case']:
            use_case = preferences['use_case']
            spec_scores = self._evaluate_use_case_specs(specs, use_case)
            if spec_scores:
                if 'gaming' in str(use_case):
                    # Gaming should depend mostly on processor + RAM
                    chipset_score = spec_scores.get('Chipset', 0.2)
                    ram_score = spec_scores.get('RAM', 0.2)
                    display_score = spec_scores.get('Display', 0.2)
                    use_case_score = float(0.55 * chipset_score + 0.35 * ram_score + 0.10 * display_score)
                else:
                    use_case_score = float(np.mean(list(spec_scores.values())))
            else:
                use_case_score = 0.5
        contributions['use_case_alignment'] = use_case_score
        
        # 4. Overall Specs Quality
        specs_quality = self._evaluate_specs_quality(specs)
        contributions['specs_quality'] = specs_quality
        
        return contributions
    
    def _calculate_weighted_score(self, contributions: Dict[str, float]) -> float:
        """Calculate weighted score from feature contributions"""
        
        weighted_score = 0.0
        for feature, value in contributions.items():
            weight = self.feature_weights.get(feature, 0.1)
            weighted_score += value * weight
        
        return float(min(1.0, max(0.0, weighted_score)))
    
    def _calculate_confidence(self, score: float, contributions: Dict[str, float]) -> float:
        """Calculate confidence in the recommendation (XAI-style)"""
        
        # Strong contribution if weighted value > 0.15
        strong_contributions = sum(
            1 for feature, value in contributions.items()
            if value * self.feature_weights.get(feature, 0.1) > 0.15
        )
        
        # Calculate variance
        weighted_values = [value * self.feature_weights.get(feature, 0.1) for feature, value in contributions.items()]
        variance = np.var(weighted_values) if weighted_values else 0
        
        # Combine factors
        confidence = (
            score * 0.5 +
            min(strong_contributions / 5, 1.0) * 0.3 +
            (1 - min(variance, 1.0)) * 0.2
        )
        
        return float(min(1.0, max(0.0, confidence)))
    
    def fit(self, devices: List[Dict[str, Any]]):
        """Train the recommender with device data"""
        
        if not devices:
            return
        
        # PRIORITY 3.1: Impute missing specs
        logger.info("Imputing missing specs...")
        devices = self.imputer.impute_missing_specs(devices)
        
        # PRIORITY 1.1: Filter invalid devices
        logger.info("Filtering invalid devices...")
        valid_devices = [d for d in devices if self._count_missing_specs(d) <= 2]
        logger.info(f"Kept {len(valid_devices)}/{len(devices)} devices (filtered {len(devices)-len(valid_devices)} with >2 missing specs)")
        
        self.raw_devices = valid_devices
        self.device_ids = [str(d.get('id', '')) for d in valid_devices]
        
        # Create feature text
        feature_texts = [self._create_feature_text(d) for d in valid_devices]
        
        # Train TF-IDF
        self.feature_matrix = self.vectorizer.fit_transform(feature_texts)
        
        # Store device features
        self.device_features = pd.DataFrame([
            {
                'id': str(d.get('id', '')),
                'brand': d.get('brand', ''),
                'model_name': d.get('model_name', ''),
                'device_type': d.get('device_type', ''),
                'price': self._extract_price(d),
                'specs': d.get('specs', {}),
            }
            for d in valid_devices
        ])
        
        # Generate semantic embeddings for devices
        if self.use_semantic and self.semantic_embedder:
            logger.info("Generating semantic embeddings for devices...")
            device_descriptions = [self._create_device_description(d) for d in valid_devices]
            self.device_embeddings = self.semantic_embedder.encode(
                device_descriptions,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            logger.info(f"✓ Generated embeddings for {len(self.device_embeddings)} devices")
        
        logger.info(f"Recommender trained on {len(self.device_ids)} devices")
    
    def recommend_by_preferences(self,
                                preferences: Dict[str, Any],
                                top_n: int = 3,
                                use_mcdm: bool = False) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Recommend devices with explanations (with semantic NLP support)
        
        Returns: List of (device_id, score, explanation) tuples
        """
        
        if self.feature_matrix is None:
            return []
        
        # Use semantic parser if available, otherwise fall back to legacy
        if self.use_semantic and self.semantic_parser:
            logger.info("Using semantic NLP parser")
            semantic_prefs = self.semantic_parser.parse(preferences.get('query', ''))
            
            # Merge semantic preferences with user preferences (user prefs take priority)
            for key, value in semantic_prefs.items():
                if key not in preferences or preferences.get(key) in [None, '', []]:
                    preferences[key] = value
            
            # Log semantic understanding
            if semantic_prefs.get('multi_intent'):
                intents_str = ', '.join([f"{uc}:{conf:.2f}" for uc, conf in semantic_prefs['multi_intent'][:3]])
                logger.info(f"Semantic multi-intent: {intents_str}")
            
            if semantic_prefs.get('implicit_reasoning'):
                logger.debug(f"Implicit: {semantic_prefs['implicit_reasoning'][0]}")
        else:
            # Fall back to legacy NLP
            logger.info("Using legacy NLP parser")
            use_case, confidence = self.nlp_parser.detect_use_case(preferences.get('query', ''))
            exclusions = self.nlp_parser.parse_exclusions(preferences.get('query', ''))
            constraints = self.nlp_parser.parse_constraints(preferences.get('query', ''))
            
            preferences['use_case'] = use_case
            preferences['use_case_confidence'] = confidence
            preferences['exclusions'] = exclusions

            # Merge extracted constraints if user didn't pre-set them
            for key, val in constraints.items():
                if val and not preferences.get(key):
                    preferences[key] = val

        # If user mentions "phone" and no device_type specified, prefer mobile to avoid tablets
        if not preferences.get('device_type') and 'query' in preferences:
            ql = str(preferences['query']).lower()
            if 'phone' in ql:
                preferences['device_type'] = ['mobile']

        # Infer brand preference from raw query when parser misses it
        if not preferences.get('brand_preference') and preferences.get('query'):
            inferred_brands = self._infer_brand_preferences_from_query(preferences.get('query', ''))
            if inferred_brands:
                preferences['brand_preference'] = inferred_brands
                logger.info(f"Inferred brand preference from query: {inferred_brands}")
        
        # Calculate confidence
        if not preferences.get('query_confidence'):
            query_confidence = self.nlp_parser.calculate_query_confidence(preferences)
            preferences['query_confidence'] = query_confidence

        logger.debug(f"Final prefs: use_case={preferences.get('use_case')} "
                    f"(conf={preferences.get('use_case_confidence', 0):.2f}), "
                    f"query_conf={preferences.get('query_confidence', 0):.2f}")
        
        # Filter devices
        filtered_indices = self._apply_filters(preferences)
        
        logger.info(f"After filters: {len(filtered_indices)} devices remaining")

        # Fallbacks: progressively relax strict numeric filters to avoid empty results
        if len(filtered_indices) == 0:
            relaxed_prefs = preferences.copy()
            relaxed_order = ['min_storage', 'min_ram_gb', 'min_battery']
            for key in relaxed_order:
                if relaxed_prefs.get(key):
                    relaxed_prefs.pop(key, None)
                    relaxed_indices = self._apply_filters(relaxed_prefs)
                    logger.warning(f"No devices after filters; relaxing {key} -> {len(relaxed_indices)} devices")
                    if len(relaxed_indices) > 0:
                        filtered_indices = relaxed_indices
                        preferences[f'relaxed_{key}'] = True
                        break
            
            # Special relaxation for gaming: lower RAM requirement from 6GB to 4GB
            if len(filtered_indices) == 0 and 'gaming' in str(preferences.get('use_case', '')):
                logger.warning("No gaming devices found; relaxing gaming RAM filter from 6GB to 4GB")
                relaxed_prefs['relaxed_gaming_ram'] = True
                filtered_indices = self._apply_filters(relaxed_prefs)
                logger.info(f"After gaming RAM relaxation: {len(filtered_indices)} devices")
            
            if len(filtered_indices) == 0:
                logger.warning("No devices passed filters even after relaxation")
                return []
        
        # Create query vector
        query_text = self._create_query_text(preferences)
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate TF-IDF scores
        if issparse(self.feature_matrix):
            feature_matrix_csr = csr_matrix(self.feature_matrix)
        else:
            feature_matrix_csr = csr_matrix(self.feature_matrix)
        filtered_matrix = feature_matrix_csr[filtered_indices]
        tfidf_scores = cosine_similarity(query_vector, filtered_matrix).flatten()
        
        # Calculate semantic scores if available
        semantic_scores = None
        if self.use_semantic and self.semantic_embedder and self.device_embeddings is not None:
            try:
                query_embedding = self.semantic_embedder.encode(
                    preferences.get('query', ''), 
                    convert_to_tensor=True
                )
                filtered_embeddings = self.device_embeddings[filtered_indices.tolist()]
                semantic_scores = util.cos_sim(query_embedding, filtered_embeddings)[0].cpu().numpy()
                logger.info(f"Semantic scores: min={semantic_scores.min():.3f}, max={semantic_scores.max():.3f}")
            except Exception as e:
                logger.warning(f"Semantic scoring failed: {e}")
                semantic_scores = None
        
        # Hybrid scoring: combine TF-IDF and semantic scores
        if semantic_scores is not None:
            # Normalize both scores to 0-1
            tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-10)
            semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
            
            # Weighted combination: 70% semantic, 30% TF-IDF
            similarity_scores = 0.7 * semantic_norm + 0.3 * tfidf_norm
            logger.info("Using hybrid scoring (70% semantic + 30% TF-IDF)")
        else:
            similarity_scores = tfidf_scores
            logger.info("Using TF-IDF scoring only")
        
        # NEW: Blend in feature contribution scores for better XAI alignment
        feature_contribution_scores = self._calculate_feature_contribution_scores(
            filtered_indices, preferences
        )
        if feature_contribution_scores is not None:
            # Normalize feature scores
            fc_norm = (feature_contribution_scores - feature_contribution_scores.min()) / (
                feature_contribution_scores.max() - feature_contribution_scores.min() + 1e-10
            )
            # Blend with similarity scores: 60% textual/semantic, 40% feature contribution
            similarity_scores = 0.6 * similarity_scores + 0.4 * fc_norm
            logger.info("Blended scoring: 60% textual/semantic + 40% feature contribution")
        
        # Adjust scores
        if use_mcdm:
            adjusted_scores = self._adjust_scores_mcdm(similarity_scores, filtered_indices, preferences)
        else:
            adjusted_scores = self._adjust_scores(similarity_scores, filtered_indices, preferences)
        
        # Calibrate scores to 0-1 without forcing weak top result to 100%
        if len(adjusted_scores) > 0:
            adjusted_scores = np.nan_to_num(adjusted_scores, nan=0.0, posinf=0.0, neginf=0.0)
            adjusted_scores = np.clip(adjusted_scores, 0, None)
            adjusted_scores = np.tanh(adjusted_scores)
        
        # Get top N
        top_indices = np.argsort(adjusted_scores)[-top_n:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            device_idx = filtered_indices[idx]
            device_id = self.device_ids[device_idx]
            score = float(adjusted_scores[idx])
            device = self.raw_devices[device_idx]
            
            explanation = self._generate_explanation(device, preferences, score)
            results.append((device_id, score, explanation))
        
        return results
    
    def _apply_filters(self, preferences: Dict[str, Any]) -> np.ndarray:
        """Apply hard filters"""
        
        if self.device_features is None:
            return np.array([], dtype=int)
        
        mask = np.ones(len(self.device_features), dtype=bool)
        logger.debug(f"Initial candidates: {np.sum(mask)} devices")

        # Exclude devices missing basic data
        price_series = self.device_features['price'].fillna(0).astype(float)
        battery_series = self.device_features['specs'].apply(lambda s: (s.get('battery_mah', 0) if isinstance(s, dict) else 0) or 0)
        storage_series = self.device_features['specs'].apply(lambda s: (s.get('storage_gb', 0) if isinstance(s, dict) else 0) or 0)
        
        # For brand-constrained queries, allow partial specs/price metadata to avoid dropping entire brand catalogs
        if not preferences.get('brand_preference'):
            mask &= (np.asarray(price_series.values) > 0)
            mask &= (np.asarray(battery_series.values) > 0)
            mask &= (np.asarray(storage_series.values) > 0)
            logger.debug(f"After basic data filter: {np.sum(mask)} devices")
        else:
            logger.debug(f"Skipping basic data filter (brand-constrained query)")
        
        # Budget filter - only apply to devices WITH price data
        if 'budget' in preferences and preferences['budget']:
            budget = float(preferences['budget'])
            # Only include devices within budget
            price_mask = (price_series > 0) & (price_series <= budget)
            mask &= np.asarray(price_mask.values)

        # Budget range filter if provided
        if preferences.get('budget_min') or preferences.get('budget_max'):
            low = float(preferences.get('budget_min') or 0)
            high = float(preferences.get('budget_max') or np.inf)
            # Only include devices in price range
            range_mask = (price_series >= low) & (price_series <= high)
            mask &= np.asarray(range_mask.values)
        
        # Device type filter
        if 'device_type' in preferences and preferences['device_type']:
            device_types = preferences['device_type']
            if not isinstance(device_types, list):
                device_types = [device_types]
            if any(dt == 'mobile' for dt in device_types):
                type_series = self.device_features['device_type'].fillna('').astype(str).str.lower().str.strip()
                # Keep explicit mobile plus unknown labels; tablets are filtered below by model name
                type_mask = (type_series == 'mobile') | (type_series == '')
            else:
                type_mask = self.device_features['device_type'].isin(device_types)
            mask &= np.asarray(type_mask.values)

            # If user wants a phone/mobile, explicitly drop tablets/iPads that may lack type labels
            if any(dt == 'mobile' for dt in device_types):
                model_series = self.device_features['model_name'].fillna('').str.lower()
                tablet_mask = ~(model_series.str.contains('ipad') | model_series.str.contains('tablet') | model_series.str.contains('tab '))
                mask &= np.asarray(tablet_mask.values)

        # Brand preference filter
        if preferences.get('brand_preference'):
            brands = preferences['brand_preference']
            if not isinstance(brands, list):
                brands = [brands]
            brand_targets = [str(b).strip().lower() for b in brands if str(b).strip()]
            if brand_targets:
                expanded_targets = list(brand_targets)
                if 'samsung' in expanded_targets:
                    expanded_targets.append('galaxy')
                if 'apple' in expanded_targets:
                    expanded_targets.append('iphone')
                if 'google' in expanded_targets:
                    expanded_targets.append('pixel')

                brand_series = self.device_features['brand'].fillna('').astype(str).str.strip().str.lower()
                model_series = self.device_features['model_name'].fillna('').astype(str).str.strip().str.lower()

                def _brand_match_row(row_idx: int) -> bool:
                    brand_text = str(brand_series.iloc[row_idx]).strip().lower()
                    model_text = str(model_series.iloc[row_idx]).strip().lower()
                    is_brand_missing = (not brand_text) or brand_text in ['unknown', 'n/a', 'nan']
                    text = brand_text if not is_brand_missing else f"{brand_text} {model_text}".strip()
                    return any(text.startswith(target) for target in expanded_targets)

                brand_mask = np.array([_brand_match_row(i) for i in range(len(brand_series))], dtype=bool)
                matched_count = np.sum(brand_mask)
                logger.info(f"Brand filter '{brand_targets}' matched {matched_count} devices")
                mask &= brand_mask

        # Battery minimum filter
        if preferences.get('min_battery'):
            battery_min = int(preferences['min_battery'])
            battery_mask = battery_series.apply(lambda x: (x or 0) >= battery_min if isinstance(x, (int, float)) or x is None else False)
            mask &= np.asarray(battery_mask.values)
        
        # RAM minimum filter
        if preferences.get('min_ram_gb'):
            ram_series = self.device_features['specs'].apply(lambda s: s.get('ram_gb', 0) if isinstance(s, dict) else 0)
            ram_min = int(preferences['min_ram_gb'])
            ram_mask = ram_series.apply(lambda x: (x or 0) >= ram_min if isinstance(x, (int, float)) or x is None else False)
            mask &= np.asarray(ram_mask.values)
        
        # Storage minimum filter
        if preferences.get('min_storage'):
            storage_min = int(preferences['min_storage'])
            storage_mask = storage_series.apply(lambda x: (x or 0) >= storage_min if isinstance(x, (int, float)) or x is None else False)
            mask &= np.asarray(storage_mask.values)

        # Gaming baseline requirements: avoid clearly non-gaming devices
        # Lower threshold to 6GB (flagship chips with 8GB RAM still score highly via scoring function)
        if 'gaming' in str(preferences.get('use_case', '')):
            ram_series = self.device_features['specs'].apply(lambda s: (s.get('ram_gb', 0) if isinstance(s, dict) else 0) or 0)
            # Check if gaming RAM was relaxed (6GB -> 4GB)
            min_gaming_ram = 4 if preferences.get('relaxed_gaming_ram') else 6
            mask &= np.asarray((ram_series >= min_gaming_ram).values)
            logger.info(f"Gaming RAM filter: requiring {min_gaming_ram}GB+, {np.sum(mask)} devices remain")

        # Flagship minimum requirements
        if preferences.get('use_case') == 'flagship':
            ram_series = self.device_features['specs'].apply(lambda s: (s.get('ram_gb', 0) if isinstance(s, dict) else 0) or 0)
            camera_series = self.device_features['specs'].apply(lambda s: (s.get('main_camera_mp', 0) if isinstance(s, dict) else 0) or 0)
            
            # Flagship minimums: 8GB+ RAM, 48MP+ camera
            mask &= np.asarray((ram_series >= 8).values)
            mask &= np.asarray((camera_series >= 48).values)
        
        # PRIORITY 4.2: Exclusion filter
        if 'exclusions' in preferences and preferences['exclusions']:
            for exclusion in preferences['exclusions']:
                brand_mask = ~self.device_features['brand'].str.contains(exclusion, case=False, na=False)
                mask &= np.asarray(brand_mask.values)
        
        return np.where(mask)[0]
    
    def _create_query_text(self, preferences: Dict[str, Any]) -> str:
        """Create search query from preferences"""
        
        query_parts = []
        
        # Query text (reduced repetition to prevent dominance)
        if 'query' in preferences and preferences['query']:
            query_parts.append(preferences['query'])
        
        # Use-case keywords (only if confidence > 0.6)
        use_case_confidence = preferences.get('use_case_confidence', 0)
        use_case = preferences.get('use_case', 'balanced')
        
        if use_case_confidence > 0.6:
            if 'gaming' in use_case:
                query_parts.extend(['snapdragon', 'adreno', 'gpu', 'processor'] * 2)
            elif 'photography' in use_case:
                query_parts.extend(['camera', 'megapixel', 'lens', 'zoom'] * 2)
            elif 'battery' in use_case:
                query_parts.extend(['battery', 'mah', 'charging', 'endurance'] * 2)
            elif 'display' in use_case:
                query_parts.extend(['display', 'oled', 'amoled', 'hdr', 'screen'] * 2)
        
        # Brand preference
        if 'brand_preference' in preferences and preferences['brand_preference']:
            brands = preferences['brand_preference']
            if isinstance(brands, list):
                query_parts.extend(brands * 2)
            else:
                query_parts.extend([brands] * 2)
        
        # Device type
        if 'device_type' in preferences:
            device_type = preferences['device_type']
            if isinstance(device_type, list):
                query_parts.extend(device_type)
            else:
                query_parts.append(device_type)

        # Hard constraints
        if preferences.get('min_battery'):
            query_parts.extend(['battery', 'mah', 'long lasting'])
        if preferences.get('require_durability'):
            query_parts.extend(['durable', 'ip68', 'rugged'])
        if preferences.get('require_network_resilience'):
            query_parts.extend(['coverage', 'network', 'dual sim'])
        
        return ' '.join(query_parts)
    
    def _calculate_feature_contribution_scores(self, indices, preferences):
        """Calculate feature contribution scores for filtered devices"""
        
        if self.raw_devices is None or len(indices) == 0:
            return None
        
        scores = np.zeros(len(indices))
        
        for i, device_idx in enumerate(indices):
            if device_idx >= len(self.raw_devices):
                continue
            
            device = self.raw_devices[device_idx]
            contributions = self._calculate_feature_contributions(device, preferences)
            weighted_score = self._calculate_weighted_score(contributions)
            scores[i] = weighted_score
        
        return scores
    
    def _adjust_scores(self, scores: np.ndarray, indices: np.ndarray,
                      preferences: Dict[str, Any]) -> np.ndarray:
        """Adjust scores with Priority optimizations"""
        
        # Start with lower base scores to give more weight to spec boosts
        adjusted = scores.copy() * 0.4
        
        if self.device_features is None:
            return adjusted
        
        # Brand preference
        if 'brand_preference' in preferences and preferences['brand_preference']:
            brands = preferences['brand_preference']
            if not isinstance(brands, list):
                brands = [brands]
            brand_tokens = [str(b).lower() for b in brands]
            if 'samsung' in brand_tokens:
                brand_tokens.append('galaxy')
            if 'apple' in brand_tokens:
                brand_tokens.append('iphone')
            if 'google' in brand_tokens:
                brand_tokens.append('pixel')
            
            for i, idx in enumerate(indices):
                brand_val = str(self.device_features.iloc[idx].get('brand', '')).lower().strip()
                model_val = str(self.device_features.iloc[idx].get('model_name', '')).lower().strip()
                is_brand_missing = (not brand_val) or brand_val in ['unknown', 'n/a', 'nan']
                device_brand_text = brand_val if not is_brand_missing else f"{brand_val} {model_val}".strip()
                if any(device_brand_text.startswith(token) for token in brand_tokens):
                    adjusted[i] *= 1.5
        
        # Price preference
        if 'budget' in preferences and preferences['budget']:
            budget = float(preferences['budget'])
            for i, idx in enumerate(indices):
                device_price = self.device_features.iloc[idx]['price']
                if device_price > 0:
                    price_ratio = device_price / budget
                    if 0.7 <= price_ratio <= 0.95:
                        adjusted[i] *= 1.4
        
        # PRIORITY 1.3: Use confidence-based gaming boost
        use_case_confidence = preferences.get('use_case_confidence', 0)
        use_case = preferences.get('use_case', 'balanced')
        
        # Reduce boost if confidence is low
        if use_case_confidence < 0.6:
            boost_factor = 0.5
        else:
            boost_factor = 1.0
        
        adjusted = self._adjust_scores_by_use_case(adjusted, indices, use_case, boost_factor)
        adjusted = self._adjust_scores_by_specs(adjusted, indices, preferences)
        
        return adjusted
    
    def _adjust_scores_by_use_case(self, scores: np.ndarray, indices: np.ndarray,
                                  use_case: str, boost_factor: float = 1.0) -> np.ndarray:
        """Adjust scores by use case"""
        
        adjusted = scores.copy()
        
        if self.device_features is None or len(indices) == 0:
            return adjusted
        
        for i, idx in enumerate(indices):
            if idx >= len(self.device_features):
                continue
                
            specs = self.device_features.iloc[idx]['specs']
            
            # Defensive check: specs should be a dict
            if not isinstance(specs, dict):
                specs = {}
            
            if 'gaming' in use_case:
                # Calculate gaming score using smart chipset+RAM scoring (60%+40%)
                gaming_score = self._calculate_gaming_score(specs)
                
                # Apply strong boosts based on gaming score
                # High gaming scores (0.8+) = top-tier gaming phones (flagship chipset + 12GB+ RAM)
                # Mid gaming scores (0.5-0.8) = capable gaming phones (mid/top chipset + 8GB+ RAM)
                # Low gaming scores (<0.5) = budget gaming phones (mid chipset + 6GB RAM)
                
                if gaming_score >= 0.8:
                    # Elite gaming phones: 3x boost
                    gaming_multiplier = 3.0 * boost_factor
                elif gaming_score >= 0.6:
                    # Strong gaming phones: 2.5x boost
                    gaming_multiplier = 2.5 * boost_factor
                elif gaming_score >= 0.4:
                    # Capable gaming phones: 2x boost
                    gaming_multiplier = 2.0 * boost_factor
                else:
                    # Basic gaming capability: 1.3x boost
                    gaming_multiplier = 1.3 * boost_factor
                
                adjusted[i] *= gaming_multiplier
            
            elif 'photography' in use_case:
                main_cam_boost = min((specs.get('main_camera_mp', 0) or 0) / 200.0, 1.0) * 0.25 * boost_factor
                selfie_cam_boost = min((specs.get('selfie_camera_mp', 0) or 0) / 100.0, 1.0) * 0.10 * boost_factor
                
                adjusted[i] *= (1.0 + main_cam_boost + selfie_cam_boost)
            
            elif 'battery' in use_case:
                battery_mah = specs.get('battery_mah', 0) or 0
                
                # Tier-based battery scoring (similar to gaming)
                if battery_mah >= 6000:
                    # Exceptional battery: 3x boost
                    battery_multiplier = 3.0 * boost_factor
                elif battery_mah >= 5500:
                    # Excellent battery: 2.5x boost
                    battery_multiplier = 2.5 * boost_factor
                elif battery_mah >= 5000:
                    # Very good battery: 2x boost
                    battery_multiplier = 2.0 * boost_factor
                elif battery_mah >= 4500:
                    # Good battery: 1.5x boost
                    battery_multiplier = 1.5 * boost_factor
                elif battery_mah >= 4000:
                    # Decent battery: 1.2x boost
                    battery_multiplier = 1.2 * boost_factor
                else:
                    # Below average battery: minimal boost
                    battery_multiplier = 1.0 * boost_factor
                
                adjusted[i] *= battery_multiplier
            
            elif 'display' in use_case:
                size_boost = min((specs.get('display_size_inches', 6.0) or 6.0) / 6.8, 1.0) * 0.4 * boost_factor
                
                adjusted[i] *= (1.0 + size_boost)
            
            elif 'flagship' in use_case or 'performance' in use_case:
                # Flagship: strong boosts for premium specs
                ram_boost = min((specs.get('ram_gb', 0) or 0) / 12.0, 1.0) * 0.5 * boost_factor
                camera_boost = min((specs.get('main_camera_mp', 0) or 0) / 200.0, 1.0) * 0.4 * boost_factor
                
                chipset = self._get_chipset_text(specs)
                tier = self._get_chipset_tier(chipset)
                if tier == 'top':
                    chipset_boost = 0.5 * boost_factor
                elif tier == 'mid':
                    chipset_boost = 0.25 * boost_factor
                elif tier == 'budget':
                    chipset_boost = 0.1 * boost_factor
                else:
                    chipset_boost = 0.0
                
                # 5G bonus for flagship
                fiveg_boost = 0.2 * boost_factor if specs.get('has_5g', False) else -0.3
                
                adjusted[i] *= (1.0 + ram_boost + camera_boost + chipset_boost + fiveg_boost)
        
        return adjusted
    
    def _adjust_scores_by_specs(self, scores: np.ndarray, indices: np.ndarray,
                               preferences: Dict[str, Any]) -> np.ndarray:
        """Adjust scores based on spec requirements"""
        
        adjusted = scores.copy()
        
        if self.device_features is None or len(indices) == 0:
            return adjusted
        
        for i, idx in enumerate(indices):
            if idx >= len(self.device_features):
                continue
                
            specs = self.device_features.iloc[idx]['specs']
            
            # Defensive check: specs should be a dict
            if not isinstance(specs, dict):
                specs = {}
            
            spec_boost = 0.0
            
            if 'min_ram_gb' in preferences:
                min_ram = preferences['min_ram_gb']
                device_ram = (specs.get('ram_gb', 0) or 0)
                if device_ram >= min_ram:
                    spec_boost += min((device_ram - min_ram) / 8.0, 0.2)
            
            if 'min_camera_mp' in preferences:
                min_cam = preferences['min_camera_mp']
                device_cam = (specs.get('main_camera_mp', 0) or 0)
                if device_cam >= min_cam:
                    spec_boost += min((device_cam - min_cam) / 100.0, 0.2)
            
            if 'min_battery' in preferences:
                min_battery = preferences['min_battery']
                device_battery = (specs.get('battery_mah', 0) or 0)
                if device_battery >= min_battery:
                    spec_boost += min((device_battery - min_battery) / 3000.0, 0.25)
                else:
                    adjusted[i] *= 0.3  # hard penalty if battery requirement not met
            
            if preferences.get('require_5g') and not specs.get('has_5g', False):
                adjusted[i] *= 0.5
            
            if preferences.get('require_nfc') and not specs.get('has_nfc', False):
                adjusted[i] *= 0.5
            
            if preferences.get('require_wireless_charging') and not specs.get('has_wireless_charging', False):
                adjusted[i] *= 0.5
            
            if preferences.get('prefer_fast_charging') and specs.get('has_fast_charging', False):
                spec_boost += 0.15

            if preferences.get('require_durability'):
                build_text = str(specs.get('design_material', '') or '').lower()
                if any(tag in build_text for tag in ['ip68', 'ip67', 'mil-std', 'gorilla']):
                    spec_boost += 0.1
                else:
                    adjusted[i] *= 0.7

            if preferences.get('require_network_resilience'):
                # Prefer devices with dual SIM or broader tech coverage
                if specs.get('has_dual_sim'):
                    spec_boost += 0.05
                if specs.get('has_5g', False):
                    spec_boost += 0.05
            
            adjusted[i] *= (1.0 + spec_boost)
        
        return adjusted
    
    def _adjust_scores_mcdm(self, scores: np.ndarray, indices: np.ndarray,
                           preferences: Dict[str, Any]) -> np.ndarray:
        """Adjust scores using TOPSIS multi-criteria method"""
        
        if self.device_features is None:
            return scores
        
        criteria_list = []
        for idx in indices:
            specs = self.device_features.iloc[idx]['specs']
            price = self.device_features.iloc[idx]['price'] or 1
            chipset_score = self._chipset_tier_score(self._get_chipset_text(specs))
            
            criteria = [
                specs.get('ram_gb', 0),
                specs.get('main_camera_mp', 0),
                specs.get('battery_mah', 0),
                chipset_score,
                1 / (price + 1),
                scores[len(criteria_list)],
            ]
            criteria_list.append(criteria)
        
        if not criteria_list:
            return scores
        
        criteria_matrix = np.array(criteria_list)
        topsis_scores = MCDMRecommender.calculate_topsis_scores(criteria_matrix)
        
        return topsis_scores
    
    def _generate_explanation(self, device: Dict[str, Any],
                             preferences: Dict[str, Any],
                             score: float) -> Dict[str, Any]:
        """Generate enhanced XAI-style explanation for recommendation"""
        
        specs = device.get('specs', {})
        
        # Calculate feature contributions for better explanations
        contributions = self._calculate_feature_contributions(device, preferences)
        confidence = self._calculate_confidence(score, contributions)
        
        # Build reasons based on feature contributions
        reasons = []
        
        # Match quality summary
        if score > 0.7:
            match_quality = "excellent"
        elif score > 0.5:
            match_quality = "good"
        else:
            match_quality = "moderate"
        reasons.append(f"✓ {match_quality.capitalize()} overall match for your needs")
        
        # Brand Match
        if contributions['brand_match'] > 0.5:
            reasons.append(f"✓ Matches your preferred brand: {device.get('brand')}")
        
        # Price Fit
        price = self._extract_price(device)
        if 'budget' in preferences and preferences['budget']:
            budget = preferences['budget']
            if price > 0 and price <= budget:
                pct = (price / budget) * 100
                reasons.append(f"✓ Price within budget: ${price:.0f} ({pct:.0f}% of ${budget:.0f})")
            elif price > budget:
                diff = price - budget
                reasons.append(f"⚠ Price slightly above budget by ${diff:.0f}")
        
        # Use Case Alignment
        use_case = preferences.get('use_case', '')
        if use_case and contributions['use_case_alignment'] > 0.5:
            quality_desc = "excellent" if contributions['use_case_alignment'] > 0.7 else "good"
            reasons.append(f"✓ {quality_desc.capitalize()} fit for {use_case}")
            
            # Specific use case details
            if 'gaming' in use_case:
                chipset_text = str(specs.get('chipset', '')).lower()
                chipset_tier = self._get_chipset_tier(chipset_text)
                ram = specs.get('ram_gb', 0)
                if chipset_tier == 'top':
                    reasons.append("✓ Gaming: Flagship-tier processor for sustained performance")
                elif chipset_tier == 'mid':
                    reasons.append("✓ Gaming: Mid-tier processor suitable for most games")
                else:
                    reasons.append("⚠ Gaming: Processor may limit high-end gaming performance")
                if ram >= 12:
                    reasons.append(f"✓ Gaming: {ram}GB RAM for smooth multitasking")
                elif ram >= 8:
                    reasons.append(f"✓ Gaming: {ram}GB RAM is acceptable for gaming")
                else:
                    reasons.append(f"⚠ Gaming: Low RAM ({ram}GB) for modern games")
            
            elif 'photography' in use_case:
                camera = specs.get('main_camera_mp', 0)
                if camera >= 64:
                    reasons.append(f"✓ Photography: {camera:.0f}MP camera for detailed shots")
                elif camera >= 48:
                    reasons.append(f"✓ Photography: Capable {camera:.0f}MP camera system")
            
            elif 'battery' in use_case:
                battery = specs.get('battery_mah', 0)
                if battery >= 5000:
                    reasons.append(f"✓ Battery: {battery:.0f}mAh for all-day usage")
                elif battery >= 4000:
                    reasons.append(f"✓ Battery: {battery:.0f}mAh capacity")
        
        # Specs Quality
        if contributions['specs_quality'] > 0.6:
            tier = "premium" if contributions['specs_quality'] > 0.7 else "solid"
            reasons.append(f"✓ {tier.capitalize()} specification tier")
        
        # Minimum spec satisfaction
        if preferences.get('min_battery') and specs.get('battery_mah', 0) >= preferences['min_battery']:
            reasons.append(f"✓ Meets minimum battery requirement ({preferences['min_battery']}mAh+)")
        
        if preferences.get('min_ram_gb') and specs.get('ram_gb', 0) >= preferences['min_ram_gb']:
            reasons.append(f"✓ Has required RAM ({preferences['min_ram_gb']}GB+)")
        
        if preferences.get('require_5g') and specs.get('has_5g', False):
            reasons.append("✓ Supports 5G connectivity")
        
        return {
            'score': score,
            'confidence': confidence,
            'reasons': reasons if reasons else ['✓ High relevance match'],
            'specs': {
                'processor': specs.get('chipset', 'N/A'),
                'ram': specs.get('ram_gb', 'N/A'),
                'storage': specs.get('storage_gb', 'N/A'),
                'camera': specs.get('main_camera_mp', 'N/A'),
                'battery': specs.get('battery_mah', 'N/A'),
                'price': specs.get('price', 'N/A'),
            },
            'feature_contributions': contributions,
        }
    
    def recommend_by_features(self,
                              min_ram_gb: Optional[int] = None,
                              max_price: Optional[float] = None,
                              min_camera_mp: Optional[float] = None,
                              min_battery: Optional[int] = None,
                              device_type: Optional[str] = None,
                              use_case: Optional[str] = None,
                              top_n: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Recommend devices based on specific feature requirements"""
        
        preferences = {}
        if min_ram_gb:
            preferences['min_ram_gb'] = min_ram_gb
        if max_price:
            preferences['budget'] = max_price
        if min_camera_mp:
            preferences['min_camera_mp'] = min_camera_mp
        if min_battery:
            preferences['min_battery'] = min_battery
        if device_type:
            preferences['device_type'] = device_type
        if use_case:
            preferences['use_case'] = use_case
        
        return self.recommend_by_preferences(preferences, top_n=top_n)


# Global recommender instance
recommender = DeviceRecommender()
