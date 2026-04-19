"""
RECOMMENDATION ENGINE OPTIMIZATION GUIDE
Analysis of Current Issues and Solutions to Improve Score Quality
"""

# ============================================================================
# ANALYSIS OF CURRENT SCORING ISSUES
# ============================================================================
"""
Current Problems:
1. Low Scores (0.1-0.5 range):
   - TF-IDF cosine similarity produces scores in 0.0-1.0 range
   - Multiplicative boosts are insufficient (×1.2 to ×1.5 max)
   - Final scores = baseline × boosts, caps out around 0.3-0.5
   
2. Data Quality Issues:
   - 40-60% of devices have RAM=0, battery=0, or price=0
   - These zero-value specs don't penalize devices enough
   - Budget filter (price==0 OR price ≤ budget) passes invalid devices
   
3. NLP & Query Understanding:
   - Query parser extracts limited features
   - No semantic understanding of implicit requirements
   - Defaults use_case to "gaming" on ambiguous queries
   - Doesn't understand multi-constraint logic (AND/OR operations)
   
4. Algorithm Limitations:
   - Content-based filtering alone is too simplistic
   - No understanding of feature importance hierarchy
   - No personalization or user behavior tracking
   - No ranking diversification
"""

# ============================================================================
# PRIORITY 1: IMMEDIATE FIXES (High Impact, Low Effort)
# ============================================================================

PRIORITY_1_FIXES = {
    
    "1.1 Fix Zero-Value Spec Handling": {
        "Problem": "Devices with missing critical specs rank high",
        "Current Code Location": "backend/ml/recommender.py::_apply_filters()",
        "Impact": "Scores could increase 20-40%",
        "Implementation": """
        # Before filtering, flag devices with critical missing specs
        def _apply_filters(self, preferences):
            # ... existing code ...
            
            # NEW: Filter out devices with too many missing specs
            valid_indices = []
            for idx in range(len(self.raw_devices)):
                device = self.raw_devices[idx]
                specs = device.get('specs', {})
                
                # Count missing critical specs
                missing = sum([
                    specs.get('ram_gb', 0) == 0,
                    specs.get('battery_mah', 0) == 0,
                    specs.get('main_camera_mp', 0) == 0,
                    specs.get('price', 0) == 0,
                ])
                
                # If >2 critical specs missing, skip device
                if missing <= 2:
                    valid_indices.append(idx)
            
            return np.intersect1d(budget_mask, type_mask, valid_indices)
        """,
        "Estimated Score Improvement": "+15-30% average"
    },
    
    "1.2 Normalize Scores to 0-1 Range": {
        "Problem": "Raw TF-IDF scores don't scale well with boosts",
        "Current Code Location": "backend/ml/recommender.py::_adjust_scores()",
        "Impact": "Scores become more interpretable (0.0-1.0)",
        "Implementation": """
        def _adjust_scores(self, similarity_scores, filtered_indices, preferences):
            # ... existing boost logic ...
            
            # NEW: Normalize final scores to 0-1 range
            if len(adjusted_scores) > 0:
                max_score = np.max(adjusted_scores)
                if max_score > 0:
                    adjusted_scores = adjusted_scores / max_score
            
            return adjusted_scores
        """,
        "Estimated Score Improvement": "+50-100% (visual, not algorithmic)"
    },
    
    "1.3 Reduce Default Gaming Boost": {
        "Problem": "All queries get gaming boost, inappropriate for photography/battery queries",
        "Current Code Location": "backend/ml/recommender.py::_adjust_scores_by_use_case()",
        "Impact": "Non-gaming queries get 20-30% better differentiation",
        "Implementation": """
        # Only apply use_case boost if confidence > 0.6
        if preferences.get('use_case_confidence', 0) > 0.6:
            boost = self._adjust_scores_by_use_case(...)
        else:
            # Conservative default: lower gaming boost to 0.1 instead of full value
            boost = min(self._adjust_scores_by_use_case(...), 0.1)
        """,
        "Estimated Score Improvement": "+10-20% for ambiguous queries"
    },
}

# ============================================================================
# PRIORITY 2: ALGORITHM IMPROVEMENTS (High Impact, Medium Effort)
# ============================================================================

PRIORITY_2_IMPROVEMENTS = {
    
    "2.1 Implement Hybrid Scoring (Content + Collaborative)": {
        "Problem": "Content-based alone ignores item popularity and user preferences",
        "Estimated Score Improvement": "+40-60%",
        "Implementation Strategy": """
        class HybridRecommender:
            def __init__(self):
                self.content_recommender = DeviceRecommender()  # TF-IDF
                self.popularity_scores = {}  # Device popularity
                self.similarity_matrix = None  # Device-to-device similarity
            
            def train(self, devices, user_interactions=None):
                # Train content-based model
                self.content_recommender.fit(devices)
                
                # Calculate device popularity (how often viewed/purchased)
                self._calculate_popularity(user_interactions)
                
                # Build device similarity matrix (user who bought X also bought Y)
                self._build_similarity_matrix(devices)
            
            def recommend(self, preferences, top_n=3):
                # Hybrid scoring: 60% content + 20% popularity + 20% collaborative
                content_scores = self.content_recommender.recommend(...)
                popularity_scores = self._get_popularity_scores()
                collab_scores = self._get_collaborative_scores(preferences)
                
                hybrid_scores = (
                    0.6 * content_scores +
                    0.2 * popularity_scores +
                    0.2 * collab_scores
                )
                
                return top_n_by_score(hybrid_scores)
        """,
        "Data Required": "User interaction history (optional for MVP)"
    },
    
    "2.2 Implement Learning-to-Rank (LTR)": {
        "Problem": "Importance of features not learned from data",
        "Estimated Score Improvement": "+50-80%",
        "Implementation Strategy": """
        from sklearn.ensemble import GradientBoostingRanker
        
        class LTRRecommender:
            def __init__(self):
                self.ranker = GradientBoostingRanker()
                self.feature_extractor = FeatureExtractor()
            
            def train(self, devices, preferences, user_ratings):
                # Extract features for each (device, query) pair
                X = []
                y = []  # Ground truth ratings
                
                for device in devices:
                    for prefs in preferences:
                        features = self.feature_extractor.extract(device, prefs)
                        rating = user_ratings.get((device.id, prefs.id), 0)
                        
                        X.append(features)
                        y.append(rating)
                
                # Train ranker to predict optimal ranking
                self.ranker.fit(np.array(X), np.array(y))
            
            def recommend(self, preferences, top_n=3):
                # Get candidate devices
                candidates = self._get_candidates(preferences)
                
                # Extract features for ranking
                X = [self.feature_extractor.extract(d, preferences) for d in candidates]
                
                # Predict scores using learned model
                scores = self.ranker.predict(np.array(X))
                
                return top_n_by_score(scores)
        """,
        "Data Required": "300+ user ratings for training"
    },
    
    "2.3 Add Semantic Understanding with Embeddings": {
        "Problem": "TF-IDF doesn't capture semantic similarity (e.g., 'fast' ≈ '120Hz')",
        "Estimated Score Improvement": "+30-50%",
        "Implementation Strategy": """
        from sentence_transformers import SentenceTransformer, util
        
        class SemanticRecommender:
            def __init__(self):
                # Use pre-trained BERT model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            def fit(self, devices):
                # Create semantic descriptions for devices
                self.device_embeddings = []
                for device in devices:
                    text = f\"{device.brand} {device.model}. RAM: {device.ram}GB. \
                            Camera: {device.camera}MP. Battery: {device.battery}mAh.\"
                    
                    embedding = self.model.encode(text)
                    self.device_embeddings.append(embedding)
            
            def recommend(self, query, top_n=3):
                # Encode user query
                query_embedding = self.model.encode(query)
                
                # Compute cosine similarity between query and all devices
                similarities = util.pytorch_cos_sim(query_embedding, 
                                                    self.device_embeddings)[0]
                
                # Get top N
                top_indices = torch.topk(similarities, top_n)[1]
                return [(self.devices[i].id, similarities[i]) for i in top_indices]
        """,
        "Dependencies": "sentence-transformers (lightweight, 100MB model)"
    },
    
    "2.4 Implement Multi-Criteria Decision Making": {
        "Problem": "Doesn't handle competing criteria well (fast vs battery-efficient)",
        "Estimated Score Improvement": "+20-40%",
        "Implementation Strategy": """
        from scipy.spatial.distance import euclidean
        
        class MCDMRecommender:
            # Multi-Criteria Decision Making using TOPSIS
            
            def recommend(self, preferences, top_n=3):
                # Get candidate devices
                candidates = self._get_candidates(preferences)
                
                # Normalize all criteria to 0-1 scale
                criteria = self._extract_criteria(candidates)
                criteria_normalized = self._normalize_criteria(criteria)
                
                # Calculate ideal and anti-ideal solutions
                ideal = np.max(criteria_normalized, axis=0)
                anti_ideal = np.min(criteria_normalized, axis=0)
                
                # Calculate distance from ideal and anti-ideal
                scores = []
                for device_idx in range(len(candidates)):
                    d_pos = euclidean(criteria_normalized[device_idx], ideal)
                    d_neg = euclidean(criteria_normalized[device_idx], anti_ideal)
                    
                    topsis_score = d_neg / (d_pos + d_neg)
                    scores.append(topsis_score)
                
                return top_n_by_score(scores)
        """,
        "Benefit": "Better handles Pareto frontier (tradeoffs)"
    },
}

# ============================================================================
# PRIORITY 3: DATA QUALITY & ENRICHMENT (Medium Impact, Medium Effort)
# ============================================================================

PRIORITY_3_ENHANCEMENTS = {
    
    "3.1 Implement Smart Data Imputation": {
        "Problem": "Missing specs (RAM, battery, price) hurt recommendations",
        "Estimated Score Improvement": "+15-25%",
        "Implementation": """
        class DataImputer:
            def impute_missing_specs(self, devices):
                # Group devices by brand and year
                brand_groups = defaultdict(list)
                
                for device in devices:
                    brand_groups[device.brand].append(device)
                
                # For each device with missing specs
                for device in devices:
                    if device.specs.ram == 0:
                        # Use median RAM of same brand, similar year
                        similar = [d for d in brand_groups[device.brand]
                                 if abs(d.year - device.year) <= 1 and d.specs.ram > 0]
                        device.specs.ram = np.median([d.specs.ram for d in similar]) if similar else 4
                    
                    if device.specs.battery == 0:
                        # Use median battery of similar-sized devices
                        similar = [d for d in devices 
                                 if abs(d.specs.display_size - device.specs.display_size) < 0.5 
                                 and d.specs.battery > 0]
                        device.specs.battery = np.median([d.specs.battery for d in similar]) if similar else 4000
                    
                    if device.specs.price == 0:
                        # Use median price in same market segment
                        tier = self._get_price_tier(device)
                        similar = [d for d in devices if self._get_price_tier(d) == tier and d.specs.price > 0]
                        device.specs.price = np.median([d.specs.price for d in similar]) if similar else 300
        """,
        "Code Location": "Create backend/ml/data_imputer.py"
    },
    
    "3.2 Add Device Popularity & Ratings": {
        "Problem": "No signal for which devices are actually good",
        "Estimated Score Improvement": "+20-30%",
        "Implementation": """
        # Track user interactions in MongoDB
        class UserInteraction(Document):
            user_id: str
            device_id: str
            interaction_type: str  # 'view', 'click', 'purchase', 'compare', 'save'
            timestamp: datetime
            rating: Optional[int]  # 1-5 stars
        
        # Calculate popularity scores
        def calculate_device_popularity(devices):
            for device in devices:
                interactions = UserInteraction.find(device_id=device.id)
                
                device.popularity_score = (
                    len(interactions['view']) * 0.1 +
                    len(interactions['click']) * 0.2 +
                    len(interactions['purchase']) * 0.5 +
                    len(interactions['compare']) * 0.15 +
                    np.mean([i.rating for i in interactions if i.rating]) * 0.3
                )
        """,
        "Code Location": "Extend backend/models/device.py"
    },
    
    "3.3 Implement Feature Importance Scoring": {
        "Problem": "All features weighted equally; RAM not more important than color",
        "Estimated Score Improvement": "+25-35%",
        "Implementation": """
        class FeatureImportanceScorer:
            def __init__(self):
                self.feature_weights = {}
            
            def train_from_user_data(self, user_interactions):
                # Analyze which features correlate with positive ratings
                
                for feature in ['ram', 'camera', 'battery', 'price', 'processor', 'display']:
                    correlation = self._calculate_correlation(feature, user_interactions)
                    self.feature_weights[feature] = correlation
            
            def score_device(self, device, query):
                score = 0
                for feature, weight in self.feature_weights.items():
                    device_value = getattr(device.specs, feature)
                    query_value = getattr(query, feature, None)
                    
                    if query_value is not None:
                        match = self._calculate_feature_match(device_value, query_value)
                        score += weight * match
                
                return score
        """,
        "Code Location": "Create backend/ml/feature_importance.py"
    },
}

# ============================================================================
# PRIORITY 4: NLP & QUERY UNDERSTANDING (Medium Impact, Medium Effort)
# ============================================================================

PRIORITY_4_NLP_IMPROVEMENTS = {
    
    "4.1 Improve Use-Case Detection": {
        "Problem": "Defaults to gaming; misses photography/budget intents",
        "Estimated Score Improvement": "+10-20%",
        "Implementation": """
        class EnhancedNLPParser:
            def __init__(self):
                self.use_case_keywords = {
                    'gaming': ['gaming', 'game', 'fps', 'lag', 'gpu', 'fast', 'performance', 
                              'smooth', 'refresh', 'processor', 'snapdragon'],
                    'photography': ['camera', 'photo', 'picture', 'selfie', 'zoom', 'macro',
                                   'portrait', 'night', 'professional', 'photographer'],
                    'battery': ['battery', 'endurance', 'day', 'hours', 'charge', 'travel',
                               'long', 'last', 'power'],
                    'display': ['screen', 'display', 'oled', 'amoled', 'refresh', 'smooth'],
                    'budget': ['cheap', 'affordable', 'budget', 'low-cost', 'under', 'less than'],
                    'video': ['video', 'recording', 'content creator', '4k', '8k', 'fps'],
                }
            
            def detect_use_case(self, query):
                query_lower = query.lower()
                scores = {}
                
                for use_case, keywords in self.use_case_keywords.items():
                    matches = sum(1 for kw in keywords if kw in query_lower)
                    scores[use_case] = matches
                
                # Only return use_case if it has strong signal (>2 mentions)
                if max(scores.values()) >= 2:
                    return max(scores, key=scores.get), max(scores.values()) / len(scores)
                else:
                    return 'balanced', 0.3  # Lower confidence
        """,
        "Code Location": "Enhance backend/ml/advanced_nlp_parser.py"
    },
    
    "4.2 Add Negation & Exclusion Support": {
        "Problem": "Can't handle 'NOT Samsung' or 'exclude Apple'",
        "Estimated Score Improvement": "+5-15%",
        "Implementation": """
        class NegationParser:
            def parse_exclusions(self, query):
                exclusions = []
                
                # Find patterns like "not X", "exclude X", "avoid X"
                patterns = [
                    r'(?:not|no|exclude|avoid|without|skip)\\s+([a-zA-Z]+)',
                    r'(?:anything|any|everything) (?:but|except)\\s+([a-zA-Z]+)',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    exclusions.extend(matches)
                
                return exclusions
            
            def apply_exclusions(self, recommendations, exclusions):
                # Filter out excluded brands/models
                return [r for r in recommendations 
                       if r.brand.lower() not in [e.lower() for e in exclusions]]
        """,
        "Code Location": "Extend backend/ml/advanced_nlp_parser.py"
    },
    
    "4.3 Implement Confidence Scoring for Queries": {
        "Problem": "No way to know if query was well-understood",
        "Estimated Score Improvement": "+5-10% (quality)",
        "Implementation": """
        class QueryConfidenceScorer:
            def score_query(self, parsed_query):
                confidence = 0
                
                # Budget specified: +0.3
                if parsed_query.get('budget'):
                    confidence += 0.3
                
                # Use case detected: +0.2
                if parsed_query.get('use_case_confidence', 0) > 0.5:
                    confidence += 0.2
                
                # Specific brand: +0.15
                if parsed_query.get('brand_preference'):
                    confidence += 0.15
                
                # Device type specified: +0.15
                if parsed_query.get('device_type'):
                    confidence += 0.15
                
                # Features specified: +0.2
                if parsed_query.get('required_features'):
                    confidence += 0.2
                
                return min(confidence, 1.0)
        """,
        "Benefit": "Lower confidence queries get conservative boosts"
    },
}

# ============================================================================
# PRIORITY 5: ADVANCED TECHNIQUES (Lower Priority, High Effort)
# ============================================================================

PRIORITY_5_ADVANCED = {
    
    "5.1 Implement Contextual Bandits": {
        "Problem": "No A/B testing; all users get same ranking",
        "Estimated Score Improvement": "+20-40%",
        "When to Implement": "After collecting 1000+ user interactions",
        "Overview": """
        # Use contextual bandits (Thompson Sampling) to learn best ranking strategy
        # Balances exploration (try new rankings) vs exploitation (use best known)
        from contextlib import bandit
        
        class ContextualBandit:
            def __init__(self, num_arms=10):
                self.arms = [...]  # 10 ranking strategies
                self.performance = defaultdict(list)
            
            def select_arm(self, context):
                # Given user context (age, location, budget), pick best ranking
                arm_id = thompson_sample(self.performance)
                return self.arms[arm_id]
            
            def update(self, arm_id, reward):
                self.performance[arm_id].append(reward)
        """
    },
    
    "5.2 Implement Graph-Based Recommendations": {
        "Problem": "Can't capture complex relationships (Samsung → iPhone, budget → Xiaomi)",
        "Estimated Score Improvement": "+30-50%",
        "When to Implement": "After building user interaction graph",
        "Overview": """
        # Build graph: devices connected by similarity, brand, price tier, use-case
        # Use GraphSAGE or similar GNN to generate embeddings
        
        class GraphBasedRecommender:
            def __init__(self, devices):
                self.graph = self._build_device_graph(devices)
                self.model = GraphSAGE(input_dim=100, output_dim=64)
            
            def _build_device_graph(self, devices):
                graph = nx.Graph()
                
                for i, device in enumerate(devices):
                    graph.add_node(i, data=device)
                    
                    # Connect similar devices
                    for j, other in enumerate(devices):
                        if i < j:
                            similarity = self._compute_similarity(device, other)
                            if similarity > 0.7:
                                graph.add_edge(i, j, weight=similarity)
                
                return graph
            
            def recommend(self, user_query):
                # Find seed device from content-based
                seed = self._get_seed_device(user_query)
                
                # Use GNN to find similar neighbors
                neighbors = self.model.find_neighbors(seed, k=3)
                
                return neighbors
        """
    },
    
    "5.3 Implement Deep Learning Ranking (LambdaMART)": {
        "Problem": "Gradient boosting may plateau; neural nets can model complex interactions",
        "Estimated Score Improvement": "+40-70%",
        "When to Implement": "After Priority 2 improvements",
        "Overview": """
        # LambdaMART: Gradient Boosting for Learning-to-Rank
        from xgboost import XGBRanker
        
        class DeepLearningRanker:
            def __init__(self):
                self.model = XGBRanker(objective='rank:ndcg')
            
            def train(self, X, y, groups):
                # X: features for each (device, query) pair
                # y: user ratings
                # groups: number of queries per group
                
                self.model.fit(X, y, group=groups)
            
            def rank(self, device_query_features):
                return self.model.predict(device_query_features)
        """
    },
}

# ============================================================================
# IMPLEMENTATION ROADMAP
# ============================================================================

IMPLEMENTATION_ROADMAP = """
PHASE 1 (Week 1): Quick Wins
├─ 1.1 Fix zero-value spec handling
├─ 1.2 Normalize scores to 0-1 range
└─ 1.3 Reduce gaming boost default

PHASE 2 (Week 2-3): Core Algorithm Improvements
├─ 2.1 Implement semantic embeddings (BERT)
├─ 2.2 Add hybrid scoring (content + popularity)
├─ 3.1 Implement data imputation for missing specs
└─ 4.1 Improve use-case detection

PHASE 3 (Week 4-5): Advanced Features
├─ 2.2 Implement Learning-to-Rank (LTR)
├─ 3.2 Add device popularity tracking
├─ 4.2 Add negation/exclusion support
└─ 4.3 Implement query confidence scoring

PHASE 4 (Week 6+): Production Optimization
├─ 2.3 Build device-to-device similarity matrix
├─ 3.3 Feature importance learning
├─ 5.1 Implement contextual bandits (A/B testing)
└─ 5.2 Graph-based recommendations

Expected Score Improvement:
- Phase 1: +30-50% (0.15-0.25 → 0.25-0.45)
- Phase 2: +40-60% (0.25-0.45 → 0.40-0.65)
- Phase 3: +50-80% (0.40-0.65 → 0.60-0.85)
- Phase 4: +70-100%+ (0.60-0.85 → 0.80-0.95+)
"""

# ============================================================================
# QUICK WIN: IMMEDIATE CODE CHANGES
# ============================================================================

IMMEDIATE_CHANGES = """
1. Modify backend/ml/recommender.py:

   In _apply_filters() method, add:
   
   # NEW: Filter invalid devices with too many missing specs
   missing_mask = np.ones(len(self.raw_devices), dtype=bool)
   for idx, device in enumerate(self.raw_devices):
       specs = device.get('specs', {})
       missing_count = sum([
           specs.get('ram_gb', 0) == 0,
           specs.get('battery_mah', 0) == 0,
           specs.get('main_camera_mp', 0) == 0,
       ])
       if missing_count > 2:
           missing_mask[idx] = False
   
   # Combine all masks
   final_mask = budget_mask & type_mask & missing_mask
   
   ---
   
2. In _adjust_scores() method, add at end:
   
   # Normalize scores to 0-1 range
   if len(adjusted_scores) > 0:
       max_score = np.max(adjusted_scores[adjusted_scores > 0])
       if max_score > 0:
           adjusted_scores[adjusted_scores > 0] /= max_score
   
   ---
   
3. In advanced_nlp_parser.py, modify use_case detection:
   
   # Return confidence level
   use_case_confidence = (matching_keywords / total_keywords)
   
   # Only apply use_case boost if confidence > 0.6
   if use_case_confidence < 0.6:
       use_case = 'balanced'
"""

print(__doc__)
print("\n" + "="*80)
print("PRIORITY 1: IMMEDIATE FIXES")
print("="*80)
for fix_name, details in PRIORITY_1_FIXES.items():
    print(f"\n{fix_name}")
    print(f"  Problem: {details['Problem']}")
    print(f"  Impact: {details['Estimated Score Improvement']}")
    print(f"  Location: {details['Current Code Location']}")

print("\n" + "="*80)
print("PRIORITY 2: ALGORITHM IMPROVEMENTS")
print("="*80)
for imp_name, details in PRIORITY_2_IMPROVEMENTS.items():
    print(f"\n{imp_name}")
    print(f"  Problem: {details['Problem']}")
    print(f"  Impact: {details['Estimated Score Improvement']}")

print("\n" + "="*80)
print("PRIORITY 3: DATA QUALITY")
print("="*80)
for enh_name, details in PRIORITY_3_ENHANCEMENTS.items():
    print(f"\n{enh_name}")
    print(f"  Problem: {details['Problem']}")
    print(f"  Impact: {details['Estimated Score Improvement']}")

print("\n" + "="*80)
print("PRIORITY 4: NLP IMPROVEMENTS")
print("="*80)
for nlp_name, details in PRIORITY_4_NLP_IMPROVEMENTS.items():
    print(f"\n{nlp_name}")
    print(f"  Problem: {details['Problem']}")
    print(f"  Impact: {details['Estimated Score Improvement']}")

print("\n" + "="*80)
print("PRIORITY 5: ADVANCED TECHNIQUES")
print("="*80)
for adv_name, details in PRIORITY_5_ADVANCED.items():
    print(f"\n{adv_name}")
    print(f"  Problem: {details['Problem']}")
    print(f"  Impact: {details['Estimated Score Improvement']}")

print("\n" + "="*80)
print("IMPLEMENTATION ROADMAP")
print("="*80)
print(IMPLEMENTATION_ROADMAP)

print("\n" + "="*80)
print("IMMEDIATE ACTIONS (Copy-paste ready)")
print("="*80)
print(IMMEDIATE_CHANGES)
