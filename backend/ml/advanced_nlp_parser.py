"""
Advanced NLP Query Parser for Complex, Inconsistent, and Context-Dependent Prompts

Pure BERT-based approach using transformer models:
- Named Entity Recognition (NER) for extracting brands, features, specs
- Zero-shot classification for intent detection and use cases
- Question-Answering (QA) for extracting specific information (budget, trade-offs)
- Sentence embeddings for semantic similarity and implicit preferences

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
from sentence_transformers import SentenceTransformer  # type: ignore
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
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
    """Pure BERT-based NLP parser for complex queries"""
    
    def __init__(self):
        print("Initializing Pure BERT-based NLP Parser...")
        
        # Load BERT NER for entity extraction
        try:
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")  # type: ignore
            print("[OK] NER model loaded")
        except Exception as e:
            print(f"[ERROR] NER model failed: {e}")
            self.ner_pipeline = None
        
        # Load Zero-Shot Classifier for intent detection
        try:
            self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # type: ignore
            print("[OK] Zero-shot classifier loaded")
        except Exception as e:
            print(f"[ERROR] Zero-shot classifier failed: {e}")
            self.zero_shot_classifier = None
        
        # Load Sentence Transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[OK] Sentence transformer loaded")
        except Exception as e:
            print(f"[ERROR] Sentence transformer failed: {e}")
            self.sentence_model = None
        
        # Load QA model for extracting specific information
        try:
            self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")  # type: ignore
            print("[OK] QA model loaded")
        except Exception as e:
            print(f"[ERROR] QA model failed: {e}")
            self.qa_pipeline = None
        
        # Conflict resolution patterns (domain-specific business rules)
        self.conflicts = {
            ('budget', 'gaming'): 'mid_range_gaming',
            ('budget', 'photography'): 'mid_range_camera',
            ('cheap', 'flagship'): 'flagship_killer',
            ('cheap', 'premium'): 'value_flagship'
        }
        
        # Pre-compute embeddings for use cases and lifestyles
        self._precompute_semantic_mappings()
        
        print("[OK] NLP Parser initialized successfully\n")
    
    def _precompute_semantic_mappings(self):
        """Pre-compute embeddings for common use cases and lifestyles"""
        
        if not self.sentence_model:
            self.use_case_embeddings = {}
            self.lifestyle_embeddings = {}
            return
        
        # Use case descriptions
        self.use_case_descriptions = {
            'gaming': 'playing mobile games, high performance gaming, smooth gameplay, fast processor, high refresh rate display',
            'photography': 'taking photos, camera quality, video recording, capturing memories, high megapixel camera',
            'battery': 'long battery life, all-day battery, extended usage, heavy use, power backup',
            'display': 'screen quality, bright display, AMOLED screen, watching movies, media consumption',
            'business': 'professional use, work productivity, office tasks, business meetings, enterprise features',
            'budget': 'affordable phone, cheap device, value for money, economical, budget-friendly'
        }
        
        # Lifestyle implications
        self.lifestyle_scenarios = {
            'travel frequently': {'must_have': ['battery'], 'nice_to_have': ['dual sim', 'lightweight']},
            'outdoor activities': {'must_have': ['battery'], 'nice_to_have': ['durable', 'ip rating', 'bright display']},
            'student': {'must_have': ['budget'], 'nice_to_have': ['battery', 'value']},
            'content creator': {'must_have': ['camera'], 'nice_to_have': ['video', 'storage', 'display']},
            'professional work': {'must_have': ['business'], 'nice_to_have': ['security', 'battery']},
            'elderly parent': {'must_have': ['simple'], 'nice_to_have': ['large display', 'loud speaker']},
            'parent with children': {'must_have': ['durable'], 'nice_to_have': ['parental controls', 'value']}
        }
        
        # Compute embeddings
        self.use_case_embeddings = {
            use_case: self.sentence_model.encode(desc, convert_to_tensor=True)
            for use_case, desc in self.use_case_descriptions.items()
        }
        
        self.lifestyle_embeddings = {
            lifestyle: self.sentence_model.encode(lifestyle, convert_to_tensor=True)
            for lifestyle in self.lifestyle_scenarios.keys()
        }
        
        # Known phone brands for filtering
        self.known_brands = [
            'Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Oppo', 'Vivo',
            'Realme', 'Motorola', 'Nokia', 'Sony', 'LG', 'Huawei', 'Honor',
            'Asus', 'Lenovo', 'Alcatel', 'ZTE', 'Poco', 'Nothing'
        ]
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using sentence embeddings"""
        
        if not self.sentence_model:
            return 0.0
        
        try:
            emb1 = self.sentence_model.encode(text1, convert_to_tensor=True)
            emb2 = self.sentence_model.encode(text2, convert_to_tensor=True)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception:
            return 0.0
    
    def _zero_shot_classify(self, text: str, candidate_labels: List[str], 
                           multi_label: bool = False) -> Dict[str, float]:
        """Perform zero-shot classification and return label scores"""
        
        if not self.zero_shot_classifier:
            return {label: 0.0 for label in candidate_labels}
        
        try:
            result = self.zero_shot_classifier(
                text,
                candidate_labels=candidate_labels,
                multi_label=multi_label
            )
            
            # Create label -> score mapping
            scores = {}
            if isinstance(result, dict) and 'labels' in result and 'scores' in result:
                for label, score in zip(result['labels'], result['scores']):
                    scores[label] = float(score)
            else:
                # Fallback for different result format
                return {label: 0.0 for label in candidate_labels}
            
            return scores
        except Exception:
            return {label: 0.0 for label in candidate_labels}
    
    def _extract_entities_with_ner(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract named entities using BERT NER"""
        
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            # Ensure entities is a list
            if not isinstance(entities, list):
                return []
            
            # Filter by entity type if specified and ensure all items are dicts
            result: List[Dict[str, Any]] = []
            for e in entities:
                if isinstance(e, dict):
                    if entity_types is None or e.get('entity_group') in entity_types:
                        result.append(e)
            
            return result
        except Exception:
            return []
    
    def _ask_qa_model(self, question: str, context: str) -> Optional[str]:
        """Ask QA model a question about the context"""
        
        if not self.qa_pipeline:
            return None
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            # Return answer if confidence is reasonable
            if isinstance(result, dict) and result.get('score', 0) > 0.1:
                return result.get('answer')
            return None
        except Exception:
            return None
    
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
        """Extract multiple use cases from query using zero-shot classification and semantic similarity"""
        
        found_cases = []
        
        # Use zero-shot classification for explicit use cases
        use_case_labels = [
            'gaming and mobile games',
            'photography and camera quality',
            'long battery life',
            'display quality and screen',
            'business and productivity',
            'budget and affordability'
        ]
        
        scores = self._zero_shot_classify(query, use_case_labels, multi_label=True)
        
        # Map labels back to use case keys
        label_to_key = {
            'gaming and mobile games': 'gaming',
            'photography and camera quality': 'photography',
            'long battery life': 'battery',
            'display quality and screen': 'display',
            'business and productivity': 'business',
            'budget and affordability': 'budget'
        }
        
        # Add use cases above confidence threshold
        for label, score in scores.items():
            if score > 0.4:
                use_case_key = label_to_key.get(label)
                if use_case_key:
                    found_cases.append(use_case_key)
        
        # Also check semantic similarity with use case descriptions
        if self.sentence_model and not found_cases:
            query_emb = self.sentence_model.encode(query, convert_to_tensor=True)
            
            for use_case, use_case_emb in self.use_case_embeddings.items():
                similarity = np.dot(query_emb, use_case_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(use_case_emb))
                
                if similarity > 0.5:
                    if use_case not in found_cases:
                        found_cases.append(use_case)
        
        return found_cases
    
    def _extract_negations(self, query: str) -> Dict[str, List[str]]:
        """Extract things user wants to avoid using NER and QA"""
        
        avoid = {'brands': [], 'features': []}
        
        # Ask QA model what to avoid
        questions = [
            "What brands should be avoided?",
            "What brand is not wanted?",
            "What features are not wanted?"
        ]
        
        for question in questions:
            answer = self._ask_qa_model(question, query)
            if answer:
                answer_lower = answer.lower()
                
                # Check if it's a known brand
                for brand in self.known_brands:
                    if brand.lower() in answer_lower:
                        if brand not in avoid['brands']:
                            avoid['brands'].append(brand)
                
                # If not a brand, it's a feature
                if not any(brand.lower() in answer_lower for brand in self.known_brands):
                    avoid['features'].append(answer)
        
        # Also extract entities with negative sentiment context
        # Look for negation words followed by entities
        negation_patterns = [
            'not', 'without', 'except', 'no', "don't want", "don't like",
            'avoid', 'excluding', 'besides'
        ]
        
        entities = self._extract_entities_with_ner(query, entity_types=['ORG', 'MISC'])
        
        for entity in entities:
            entity_word = entity['word']
            entity_start = entity.get('start', 0)
            
            # Check if there's a negation word nearby (within 20 characters before)
            context_start = max(0, entity_start - 20)
            context = query[context_start:entity_start].lower()
            
            if any(neg in context for neg in negation_patterns):
                # Check if it's a brand
                if any(brand.lower() == entity_word.lower() for brand in self.known_brands):
                    brand_match = [b for b in self.known_brands if b.lower() == entity_word.lower()][0]
                    if brand_match not in avoid['brands']:
                        avoid['brands'].append(brand_match)
                else:
                    if entity_word not in avoid['features']:
                        avoid['features'].append(entity_word)
        
        return avoid
    
    def _extract_trade_offs(self, query: str) -> List[Tuple[str, str]]:
        """Extract trade-off preferences using QA and semantic matching"""
        
        trade_offs = []
        
        # Ask QA model about trade-offs
        questions = [
            "What feature can be sacrificed?",
            "What is more important?",
            "What feature is preferred over others?"
        ]
        
        answers = []
        for question in questions:
            answer = self._ask_qa_model(question, query)
            if answer:
                answers.append(answer)
        
        # Try to extract sacrifice/prefer pairs
        if len(answers) >= 2:
            sacrifice = answers[0]
            prefer = answers[1]
            trade_offs.append((sacrifice, prefer))
        
        # Also look for comparative patterns in entities
        # Check for words like "over", "than", "instead of"
        comparative_words = ['over', 'than', 'instead of', 'rather than']
        
        for comp_word in comparative_words:
            if comp_word in query.lower():
                # Split by comparative word
                parts = query.lower().split(comp_word)
                if len(parts) >= 2:
                    # Extract entities from both parts
                    before_entities = self._extract_entities_with_ner(parts[0], entity_types=['MISC'])
                    after_entities = self._extract_entities_with_ner(parts[1], entity_types=['MISC'])
                    
                    if before_entities and after_entities:
                        prefer_feature = before_entities[-1]['word']
                        sacrifice_feature = after_entities[0]['word']
                        trade_offs.append((sacrifice_feature, prefer_feature))
        
        return trade_offs
    
    def _extract_context_references(self, query: str) -> List[str]:
        """Extract references to other devices or comparisons using NER and QA"""
        
        references = []
        
        # Ask QA model about device comparisons
        questions = [
            "What device is being compared?",
            "What phone is mentioned for comparison?",
            "Which device should this be similar to?"
        ]
        
        for question in questions:
            answer = self._ask_qa_model(question, query)
            if answer and answer not in references:
                references.append(answer)
        
        # Extract PRODUCT entities that might be devices
        entities = self._extract_entities_with_ner(query, entity_types=['MISC', 'ORG'])
        
        # Look for entities near comparison words
        comparison_words = ['better than', 'upgrade from', 'similar to', 'like', 'compared to', 'than']
        
        for entity in entities:
            entity_word = entity['word']
            entity_start = entity.get('start', 0)
            
            # Check if there's a comparison word nearby
            context_start = max(0, entity_start - 30)
            context_end = min(len(query), entity_start + len(entity_word) + 30)
            context = query[context_start:context_end].lower()
            
            if any(comp in context for comp in comparison_words):
                if entity_word not in references:
                    references.append(entity_word)
        
        return references
    
    def _extract_implicit_preferences(self, query: str) -> Dict[str, List[str]]:
        """Extract implicit preferences from lifestyle mentions using semantic similarity"""
        
        implicit = {'must_have': [], 'nice_to_have': []}
        
        if not self.sentence_model:
            return implicit
        
        # Compute query embedding
        query_emb = self.sentence_model.encode(query, convert_to_tensor=True)
        
        # Check similarity with lifestyle scenarios
        for lifestyle, lifestyle_emb in self.lifestyle_embeddings.items():
            similarity = np.dot(query_emb, lifestyle_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(lifestyle_emb))
            
            # If high similarity, add preferences
            if similarity > 0.65:
                preferences = self.lifestyle_scenarios[lifestyle]
                implicit['must_have'].extend(preferences.get('must_have', []))
                implicit['nice_to_have'].extend(preferences.get('nice_to_have', []))
        
        return implicit
    
    def _detect_conflicts(self, preferences: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Detect conflicting requirements using semantic similarity and zero-shot classification"""
        
        conflicts = []
        query = preferences.get('query', '')
        query_lower = query.lower()
        
        # Budget conflicts - check if budget is low but use case is high-end
        if preferences.get('budget'):
            budget = preferences['budget']
            
            if budget < 500:
                use_case = preferences.get('use_case', '')
                if use_case in ['gaming', 'photography']:
                    conflicts.append(('budget', use_case))
        
        # Semantic conflict detection using embeddings
        if self.sentence_model:
            # Check for "cheap" vs "flagship" semantic conflict
            cheap_phrases = ['cheap', 'affordable', 'budget', 'economical', 'low cost']
            premium_phrases = ['flagship', 'premium', 'high-end', 'top-tier', 'best']
            
            # Check if both concepts appear with high confidence
            cheap_scores = self._zero_shot_classify(
                query_lower,
                ['affordable and budget-friendly', 'expensive and premium'],
                multi_label=True
            )
            
            if cheap_scores.get('affordable and budget-friendly', 0) > 0.6 and \
               cheap_scores.get('expensive and premium', 0) > 0.6:
                conflicts.append(('cheap', 'flagship'))
        
        # Check explicit keyword conflicts as fallback
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable']):
            if any(word in query_lower for word in ['flagship', 'premium', 'high-end']):
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
        """Determine user's priority from query using zero-shot classification"""
        
        # Use zero-shot classification to determine priority
        priority_labels = [
            'camera quality and photography',
            'battery life and endurance',
            'gaming performance and speed',
            'best value for money',
            'balanced features'
        ]
        
        scores = self._zero_shot_classify(query, priority_labels, multi_label=False)
        
        # Map labels to priority keys
        label_to_priority = {
            'camera quality and photography': 'camera',
            'battery life and endurance': 'battery',
            'gaming performance and speed': 'performance',
            'best value for money': 'value',
            'balanced features': 'balanced'
        }
        
        # Get highest scoring priority
        if scores:
            top_label = max(scores.items(), key=lambda x: x[1])[0]
            priority = label_to_priority.get(top_label, 'balanced')
            
            # Override based on trade-offs if present
            if preferences.get('trade_offs') and len(preferences['trade_offs']) > 0:
                _, preferred = preferences['trade_offs'][0]
                # Map preferred feature to priority if it matches
                feature_priority_map = {
                    'camera': 'camera',
                    'battery': 'battery',
                    'performance': 'performance',
                    'gaming': 'performance',
                    'value': 'value',
                    'price': 'value'
                }
                for key, prio in feature_priority_map.items():
                    if key in preferred.lower():
                        return prio
            
            return priority
        
        return 'balanced'
    
    def _extract_budget(self, query: str) -> Optional[float]:
        """Extract budget from query using QA model and NER"""
        
        # Try QA model first
        questions = [
            "What is the maximum budget in dollars?",
            "What is the price limit?",
            "How much money can be spent?"
        ]
        
        for question in questions:
            answer = self._ask_qa_model(question, query)
            if answer:
                # Try to extract number from answer
                numbers = re.findall(r'\d+(?:,\d+)?', answer)
                if numbers:
                    budget_str = numbers[0].replace(',', '')
                    try:
                        return float(budget_str)
                    except ValueError:
                        pass
        
        # Fallback: Use NER to find numbers and pattern matching
        entities = self._extract_entities_with_ner(query, entity_types=['CARDINAL'])
        
        # Look for numbers near budget keywords
        budget_keywords = ['under', 'below', 'less than', 'max', 'maximum', 'around', 'about', 'budget', 'price']
        
        for entity in entities:
            entity_word = entity['word']
            entity_start = entity.get('start', 0)
            
            # Check context around number
            context_start = max(0, entity_start - 20)
            context_end = min(len(query), entity_start + len(entity_word) + 10)
            context = query[context_start:context_end].lower()
            
            if any(kw in context for kw in budget_keywords):
                # Try to parse the number
                numbers = re.findall(r'\d+(?:,\d+)?', entity_word)
                if numbers:
                    budget_str = numbers[0].replace(',', '')
                    try:
                        return float(budget_str)
                    except ValueError:
                        pass
        
        # Final fallback: regex patterns
        patterns = [
            r'under\s+\$?(\d+(?:,\d+)?)',
            r'below\s+\$?(\d+(?:,\d+)?)',
            r'less\s+than\s+\$?(\d+(?:,\d+)?)',
            r'max\s+\$?(\d+(?:,\d+)?)',
            r'around\s+\$?(\d+(?:,\d+)?)',
            r'\$(\d+(?:,\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                budget_str = match.group(1).replace(',', '')
                try:
                    return float(budget_str)
                except ValueError:
                    pass
        
        return None
    
    def _extract_brands(self, query: str) -> List[str]:
        """Extract brand mentions from query using NER and zero-shot classification"""
        
        found_brands = []
        
        # Extract ORG entities using NER
        entities = self._extract_entities_with_ner(query, entity_types=['ORG'])
        
        for entity in entities:
            entity_word = entity['word']
            entity_start = entity.get('start', 0)
            
            # Check if it matches a known brand
            for brand in self.known_brands:
                if brand.lower() == entity_word.lower():
                    # Check for negation context before the brand mention
                    context_start = max(0, entity_start - 20)
                    context = query[context_start:entity_start].lower()
                    negation_words = ['not', 'except', 'without', 'no', "don't", 'avoid', 'excluding']
                    
                    # Skip if brand is negated
                    if any(neg in context for neg in negation_words):
                        continue
                    
                    if brand not in found_brands:
                        found_brands.append(brand)
                    break
        
        # Also check for brand names directly in query (only if not negated)
        query_lower = query.lower()
        for brand in self.known_brands:
            if brand.lower() in query_lower:
                # Find position of brand in query
                brand_pos = query_lower.find(brand.lower())
                context_start = max(0, brand_pos - 20)
                context = query_lower[context_start:brand_pos]
                
                # Check for negation
                negation_words = ['not', 'except', 'without', 'no', "don't", 'avoid', 'excluding']
                if any(neg in context for neg in negation_words):
                    continue
                
                if brand not in found_brands:
                    found_brands.append(brand)
        
        return found_brands
    
    def _extract_device_type(self, query: str) -> List[str]:
        """Extract device type from query using zero-shot classification"""
        
        # Use zero-shot classification
        device_type_labels = [
            'mobile phone or smartphone',
            'tablet or iPad',
            'smartwatch or fitness band'
        ]
        
        scores = self._zero_shot_classify(query, device_type_labels, multi_label=True)
        
        # Map labels to device type keys
        label_to_type = {
            'mobile phone or smartphone': 'mobile',
            'tablet or iPad': 'tablet',
            'smartwatch or fitness band': 'smartwatch'
        }
        
        device_types = []
        for label, score in scores.items():
            if score > 0.3:  # Threshold for device type detection
                device_type = label_to_type.get(label)
                if device_type:
                    device_types.append(device_type)
        
        # Default to mobile if nothing detected
        if not device_types:
            device_types = ['mobile']
        
        return device_types


# Global advanced parser instance
advanced_parser = AdvancedNLPParser()
