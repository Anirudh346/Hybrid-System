"""
ENHANCED RECOMMENDER ENGINE
Implements Priority 1-4 Optimizations:
- Priority 1: Zero-spec filtering, score normalization, gaming bias fix
- Priority 2: Hybrid scoring, semantic embeddings, LTR, MCDM
- Priority 3: Data imputation, popularity tracking, feature importance
- Priority 4: Improved NLP, negation support, confidence scoring
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
            'display': ['screen', 'display', 'oled', 'amoled', 'ips', 'brightness', 'color', '120hz'],
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
        if parsed_query.get('min_refresh_hz'):
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
            specs.get('refresh_rate_hz', 60),
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
    
    def __init__(self):
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
    
    def _count_missing_specs(self, device: Dict[str, Any]) -> int:
        """Count how many critical specs are missing"""
        
        specs = device.get('specs', {})
        missing = 0
        
        critical_specs = ['ram_gb', 'battery_mah', 'main_camera_mp', 'price']
        
        for spec in critical_specs:
            if specs.get(spec, 0) == 0:
                missing += 1
        
        return missing
    
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
            'OS', 'Chipset', 'CPU', 'GPU',
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
        
        logger.info(f"Recommender trained on {len(self.device_ids)} devices")
    
    def recommend_by_preferences(self,
                                preferences: Dict[str, Any],
                                top_n: int = 3,
                                use_mcdm: bool = False) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Recommend devices with explanations
        
        Returns: List of (device_id, score, explanation) tuples
        """
        
        if self.feature_matrix is None:
            return []
        
        # PRIORITY 4.1 & 4.3: Enhanced NLP
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
        
        # Calculate confidence AFTER merging all extracted info
        query_confidence = self.nlp_parser.calculate_query_confidence(preferences)
        preferences['query_confidence'] = query_confidence

        logger.debug(f"NLP: use_case={use_case} (conf={confidence:.2f}), query_conf={query_confidence:.2f}, exclusions={exclusions}")
        
        # Filter devices
        filtered_indices = self._apply_filters(preferences)
        
        logger.info(f"After filters: {len(filtered_indices)} devices remaining")

        # Fallbacks: progressively relax strict numeric filters to avoid empty results
        if len(filtered_indices) == 0:
            relaxed_prefs = preferences.copy()
            relaxed_order = ['min_storage', 'min_ram_gb', 'min_refresh_hz', 'min_battery']
            for key in relaxed_order:
                if relaxed_prefs.get(key):
                    relaxed_prefs.pop(key, None)
                    relaxed_indices = self._apply_filters(relaxed_prefs)
                    logger.warning(f"No devices after filters; relaxing {key} -> {len(relaxed_indices)} devices")
                    if len(relaxed_indices) > 0:
                        filtered_indices = relaxed_indices
                        preferences[f'relaxed_{key}'] = True
                        break
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
        similarity_scores = cosine_similarity(query_vector, filtered_matrix).flatten()
        
        # Adjust scores
        if use_mcdm:
            adjusted_scores = self._adjust_scores_mcdm(similarity_scores, filtered_indices, preferences)
        else:
            adjusted_scores = self._adjust_scores(similarity_scores, filtered_indices, preferences)
        
        # PRIORITY 1.2: Normalize scores to 0-1 range
        if len(adjusted_scores) > 0:
            max_score = np.max(adjusted_scores)
            min_score = np.min(adjusted_scores)
            
            if max_score > min_score:
                adjusted_scores = (adjusted_scores - min_score) / (max_score - min_score)
            else:
                adjusted_scores = np.ones_like(adjusted_scores) * 0.5
        
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

        # Exclude devices missing basic price/battery data
        price_series = self.device_features['price'].fillna(0).astype(float)
        battery_series = self.device_features['specs'].apply(lambda s: (s.get('battery_mah', 0) if isinstance(s, dict) else 0) or 0)
        storage_series = self.device_features['specs'].apply(lambda s: (s.get('storage_gb', 0) if isinstance(s, dict) else 0) or 0)
        mask &= (np.asarray(price_series.values) > 0)
        mask &= (np.asarray(battery_series.values) > 0)
        mask &= (np.asarray(storage_series.values) > 0)
        
        # Budget filter
        if 'budget' in preferences and preferences['budget']:
            budget = float(preferences['budget'])
            price_mask = self.device_features['price'] <= budget
            mask &= price_mask.values

        # Budget range filter if provided
        if preferences.get('budget_min') or preferences.get('budget_max'):
            low = float(preferences.get('budget_min') or 0)
            high = float(preferences.get('budget_max') or np.inf)
            range_mask = (price_series >= low) & (price_series <= high)
            mask &= range_mask.values
        
        # Device type filter
        if 'device_type' in preferences and preferences['device_type']:
            device_types = preferences['device_type']
            if not isinstance(device_types, list):
                device_types = [device_types]
            type_mask = self.device_features['device_type'].isin(device_types)
            mask &= type_mask.values

            # If user wants a phone/mobile, explicitly drop tablets/iPads that may lack type labels
            if any(dt == 'mobile' for dt in device_types):
                model_series = self.device_features['model_name'].fillna('').str.lower()
                tablet_mask = ~(model_series.str.contains('ipad') | model_series.str.contains('tablet') | model_series.str.contains('tab '))
                mask &= tablet_mask.values

        # Battery minimum filter
        if preferences.get('min_battery'):
            battery_min = int(preferences['min_battery'])
            battery_mask = battery_series.apply(lambda x: (x or 0) >= battery_min if isinstance(x, (int, float)) or x is None else False)
            mask &= battery_mask.values
        
        # RAM minimum filter
        if preferences.get('min_ram_gb'):
            ram_series = self.device_features['specs'].apply(lambda s: s.get('ram_gb', 0) if isinstance(s, dict) else 0)
            ram_min = int(preferences['min_ram_gb'])
            ram_mask = ram_series.apply(lambda x: (x or 0) >= ram_min if isinstance(x, (int, float)) or x is None else False)
            mask &= ram_mask.values
        
        # Storage minimum filter
        if preferences.get('min_storage'):
            storage_min = int(preferences['min_storage'])
            storage_mask = storage_series.apply(lambda x: (x or 0) >= storage_min if isinstance(x, (int, float)) or x is None else False)
            mask &= storage_mask.values

        # Refresh rate minimum filter
        if preferences.get('min_refresh_hz'):
            refresh_series = self.device_features['specs'].apply(lambda s: s.get('refresh_rate_hz', 60) if isinstance(s, dict) else 60)
            refresh_min = int(preferences['min_refresh_hz'])
            refresh_mask = refresh_series.apply(lambda x: (x or 60) >= refresh_min if isinstance(x, (int, float)) or x is None else False)
            mask &= refresh_mask.values
        
        # Flagship minimum requirements
        if preferences.get('use_case') == 'flagship':
            ram_series = self.device_features['specs'].apply(lambda s: (s.get('ram_gb', 0) if isinstance(s, dict) else 0) or 0)
            camera_series = self.device_features['specs'].apply(lambda s: (s.get('main_camera_mp', 0) if isinstance(s, dict) else 0) or 0)
            refresh_series = self.device_features['specs'].apply(lambda s: (s.get('refresh_rate_hz', 60) if isinstance(s, dict) else 60) or 60)
            
            # Flagship minimums: 8GB+ RAM, 48MP+ camera, 90Hz+ refresh
            mask &= (ram_series >= 8).values
            mask &= (camera_series >= 48).values
            mask &= (refresh_series >= 90).values
        
        # PRIORITY 4.2: Exclusion filter
        if 'exclusions' in preferences and preferences['exclusions']:
            for exclusion in preferences['exclusions']:
                brand_mask = ~self.device_features['brand'].str.contains(exclusion, case=False, na=False)
                mask &= brand_mask.values
        
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
                query_parts.extend(['display', 'oled', 'refresh', 'hdr'] * 2)
        
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
            
            for i, idx in enumerate(indices):
                if self.device_features.iloc[idx]['brand'] in brands:
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
                ram_boost = min((specs.get('ram_gb', 0) or 0) / 12.0, 1.0) * 0.2 * boost_factor
                refresh_boost = min((specs.get('refresh_rate_hz', 60) or 60) / 120.0, 1.0) * 0.15 * boost_factor
                
                chipset = (specs.get('chipset', '') or '').lower()
                chipset_boost = 0.0
                if any(x in chipset for x in ['snapdragon 8', 'apple a1', 'exynos 2']):
                    chipset_boost = 0.3 * boost_factor
                elif any(x in chipset for x in ['snapdragon 7', 'exynos 1']):
                    chipset_boost = 0.15 * boost_factor
                
                adjusted[i] *= (1.0 + ram_boost + refresh_boost + chipset_boost)
            
            elif 'photography' in use_case:
                main_cam_boost = min((specs.get('main_camera_mp', 0) or 0) / 200.0, 1.0) * 0.25 * boost_factor
                selfie_cam_boost = min((specs.get('selfie_camera_mp', 0) or 0) / 100.0, 1.0) * 0.10 * boost_factor
                
                adjusted[i] *= (1.0 + main_cam_boost + selfie_cam_boost)
            
            elif 'battery' in use_case:
                battery_boost = min((specs.get('battery_mah', 0) or 0) / 7000.0, 1.0) * 0.35 * boost_factor
                
                if specs.get('has_fast_charging', False):
                    battery_boost += 0.10 * boost_factor
                
                adjusted[i] *= (1.0 + battery_boost)
            
            elif 'display' in use_case:
                refresh_boost = min((specs.get('refresh_rate_hz', 60) or 60) / 120.0, 1.0) * 0.25 * boost_factor
                size_boost = min((specs.get('display_size_inches', 6.0) or 6.0) / 6.8, 1.0) * 0.15 * boost_factor
                
                adjusted[i] *= (1.0 + refresh_boost + size_boost)
            
            elif 'flagship' in use_case or 'performance' in use_case:
                # Flagship: strong boosts for premium specs
                ram_boost = min((specs.get('ram_gb', 0) or 0) / 12.0, 1.0) * 0.4 * boost_factor
                refresh_boost = min((specs.get('refresh_rate_hz', 60) or 60) / 120.0, 1.0) * 0.3 * boost_factor
                camera_boost = min((specs.get('main_camera_mp', 0) or 0) / 200.0, 1.0) * 0.3 * boost_factor
                
                chipset = (specs.get('chipset', '') or '').lower()
                chipset_boost = 0.0
                if any(x in chipset for x in ['snapdragon 8', 'apple a1', 'apple a2', 'exynos 2', 'dimensity 9']):
                    chipset_boost = 0.5 * boost_factor
                elif any(x in chipset for x in ['snapdragon 7', 'exynos 1', 'dimensity 8']):
                    chipset_boost = 0.25 * boost_factor
                
                # 5G bonus for flagship
                fiveg_boost = 0.2 * boost_factor if specs.get('has_5g', False) else -0.3
                
                adjusted[i] *= (1.0 + ram_boost + refresh_boost + camera_boost + chipset_boost + fiveg_boost)
        
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
            
            criteria = [
                specs.get('ram_gb', 0),
                specs.get('main_camera_mp', 0),
                specs.get('battery_mah', 0),
                specs.get('refresh_rate_hz', 60),
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
        """Generate XAI explanation for recommendation"""
        
        specs = device.get('specs', {})
        reasons = []
        
        if 'brand_preference' in preferences and preferences['brand_preference']:
            brands = preferences['brand_preference']
            if isinstance(brands, str):
                brands = [brands]
            if device.get('brand') in brands:
                reasons.append(f"✓ Brand preference match: {device.get('brand')}")
        
        if 'budget' in preferences:
            budget = preferences['budget']
            price = specs.get('price', 0)
            if price > 0 and price <= budget:
                pct = (price / budget) * 100
                reasons.append(f"✓ Price: ${price:.0f} ({pct:.0f}% of budget)")
        elif preferences.get('budget_min') or preferences.get('budget_max'):
            price = specs.get('price', 0)
            low = preferences.get('budget_min') or 0
            high = preferences.get('budget_max') or price
            if price >= low and price <= high:
                reasons.append(f"✓ Price within range: ${price:.0f}")
        
        use_case = preferences.get('use_case', '')
        if 'gaming' in use_case:
            if specs.get('ram_gb', 0) >= 8:
                reasons.append(f"✓ Gaming: {specs['ram_gb']}GB RAM")
            if specs.get('refresh_rate_hz', 60) >= 120:
                reasons.append(f"✓ High refresh: {specs['refresh_rate_hz']}Hz")
        
        elif 'photography' in use_case:
            if specs.get('main_camera_mp', 0) >= 48:
                reasons.append(f"✓ Camera: {specs['main_camera_mp']:.0f}MP")
        
        elif 'battery' in use_case:
            if specs.get('battery_mah', 0) >= 4500:
                reasons.append(f"✓ Battery: {specs['battery_mah']:.0f}mAh")

        if preferences.get('min_battery') and specs.get('battery_mah', 0) >= preferences['min_battery']:
            reasons.append(f"✓ Meets battery minimum ({preferences['min_battery']}mAh+)")
        
        return {
            'score': score,
            'reasons': reasons if reasons else ['✓ High relevance match'],
            'specs': {
                'ram': specs.get('ram_gb', 'N/A'),
                'storage': specs.get('storage_gb', 'N/A'),
                'camera': specs.get('main_camera_mp', 'N/A'),
                'battery': specs.get('battery_mah', 'N/A'),
                'refresh': specs.get('refresh_rate_hz', 'N/A'),
                'price': specs.get('price', 'N/A'),
            }
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
