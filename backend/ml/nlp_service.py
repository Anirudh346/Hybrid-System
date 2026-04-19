import logging
import json
import re
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class NLPService:
    """
    Centralized service for NLP parsing with error handling and intelligent fallback
    Handles BERT model loading, parsing, and fallback to keyword-based search
    """
    
    def __init__(self):
        self.parser = None
        self.recommender = None
        self.is_loaded = False
        self.load_error = None
        self.initialization_attempted = False
    
    def initialize(self) -> bool:
        """Initialize BERT models with comprehensive error handling"""
        if self.initialization_attempted:
            return self.is_loaded
        
        self.initialization_attempted = True
        
        try:
            logger.info("🔄 Initializing NLP parser (BERT models)...")
            from ml.advanced_nlp_parser import advanced_parser
            self.parser = advanced_parser
            logger.info("✅ NLP parser loaded successfully")
            self.is_loaded = True
            return True
        except ImportError as e:
            logger.error(f"❌ Import error loading NLP parser: {str(e)}")
            self.load_error = f"Import error: {str(e)}"
            self.is_loaded = False
            return False
        except RuntimeError as e:
            logger.error(f"❌ Runtime error loading NLP parser (likely CUDA/torch issue): {str(e)}")
            self.load_error = f"Runtime error: {str(e)}"
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error loading NLP parser: {str(e)}")
            self.load_error = f"Unexpected error: {str(e)}"
            self.is_loaded = False
            return False
    
    def parse_query(self, query: str, use_fallback: bool = True) -> Dict[str, Any]:
        """
        Parse natural language query with intelligent fallback
        Tries BERT first, falls back to keyword matching if needed
        """
        try:
            # Try BERT parsing
            if not self.parser:
                if not self.is_loaded:
                    logger.warning("⚠️  Parser not loaded, attempting initialization...")
                    self.initialize()
                
                if not self.parser:
                    logger.warning("⚠️  Parser still not available, using fallback parsing")
                    return self._fallback_parse(query)
            
            logger.debug(f"📝 Parsing query with BERT: '{query}'")
            preferences = self.parser.parse_complex_query(query)
            
            # Validate parsing
            if not preferences or (isinstance(preferences, dict) and len(preferences) == 0):
                logger.warning(f"⚠️  BERT returned empty preferences for query: {query}")
                if use_fallback:
                    logger.info("↩️  Falling back to keyword-based parsing")
                    return self._fallback_parse(query)
                return self._empty_preferences()
            
            logger.debug(f"✅ BERT parsing successful: {json.dumps(preferences, default=str)[:200]}...")
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Error during BERT parsing: {str(e)}")
            if use_fallback:
                logger.info("↩️  Falling back to keyword-based parsing")
                return self._fallback_parse(query)
            return self._empty_preferences()
    
    def _empty_preferences(self) -> Dict[str, Any]:
        """Return empty but valid preferences structure"""
        return {
            'brand_preference': [],
            'use_case': '',
            'budget': None,
            'specs': {}
        }
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """
        Fallback keyword-based query parsing when BERT fails
        Uses simple regex and keyword matching
        """
        logger.info(f"📍 Using fallback keyword parsing for: {query}")
        query_lower = query.lower()
        preferences = self._empty_preferences()
        
        # Extract brand preferences
        brands = ['samsung', 'apple', 'google', 'oneplus', 'xiaomi', 'poco', 'motorola', 
                  'nokia', 'sony', 'asus', 'oppo', 'vivo', 'realme', 'huawei', 'honor']
        for brand in brands:
            if brand in query_lower:
                preferences['brand_preference'].append(brand)
        
        # Detect use case from keywords
        use_cases = {
            'gaming': ['game', 'gaming', 'fps', 'performance', 'fast', 'powerful'],
            'camera': ['camera', 'photo', 'photography', 'pictures', 'photos'],
            'battery': ['battery', 'endurance', 'long battery', 'all-day'],
            'budget': ['cheap', 'affordable', 'budget', 'under', 'inexpensive'],
            'flagship': ['flagship', 'premium', 'best', 'latest', 'top'],
            '5g': ['5g', '5g enabled', '5g ready'],
            'battery_life': ['battery life', 'long lasting'],
        }
        
        for use_case, keywords in use_cases.items():
            if any(kw in query_lower for kw in keywords):
                preferences['use_case'] = use_case
                break
        
        # Extract budget/price
        price_patterns = [
            r'\$?(\d+)(?:\s*(?:usd|dollars))?(?:\s*-\s*\$?(\d+))?',  # $500 or $500-$700
            r'under\s*\$?(\d+)',  # under $500
            r'below\s*\$?(\d+)',  # below $500
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                preferences['budget'] = int(match.group(1))
                logger.debug(f"💰 Extracted budget: ${preferences['budget']}")
                break
        
        logger.debug(f"🔍 Fallback parsed: {json.dumps(preferences, default=str)}")
        return preferences
    
    def validate_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Validate preferences structure"""
        required_keys = ['brand_preference', 'use_case', 'budget']
        for key in required_keys:
            if key not in preferences:
                return False
        return True

# Global NLP service instance
nlp_service = NLPService()

def get_nlp_service() -> NLPService:
    """Get the global NLP service instance"""
    return nlp_service
