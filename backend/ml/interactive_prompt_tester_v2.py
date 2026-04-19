"""
INTERACTIVE PROMPT TESTER V2 - With Enhanced Recommendations & XAI
Tests the optimized recommendation system (Priority 1-4 implementations + Semantic BERT NLP)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractivePromptTesterV2:
    """Interactive testing tool for optimized recommender"""
    
    def __init__(self, device_limit: int = 5000):
        """Initialize the tester"""
        print("\n" + "="*90)
        print("📱 ENHANCED PHONE RECOMMENDATION SYSTEM V2 - TESTING MODE")
        print("With Priority 1-4 Optimizations + Semantic BERT NLP")
        print("="*90)
        print("\n⏳ Loading dataset... (this may take a moment)")
        
        # Load dataset
        self.loader = PhoneDatasetLoader()
        self.devices = self.loader.load_csv_files(limit=device_limit)
        print(f"✅ Loaded {len(self.devices)} devices\n")
        
        # Train recommender V2 (now with semantic embeddings)
        print("🤖 Training enhanced recommendation engine with semantic embeddings...")
        print("   This includes generating BERT embeddings for all devices...")
        self.recommender = DeviceRecommender(use_semantic=True)
        self.recommender.fit(self.devices)
        
        if self.recommender.use_semantic:
            print("✅ Recommender ready with SEMANTIC NLP enabled")
        else:
            print("⚠️  Recommender ready (semantic NLP not available, using legacy)\n")
        
        # Create device lookup
        self.device_lookup = {str(d.get('id', '')): d for d in self.devices}
    
    def _display_recommendation(self, device_id: str, explanation: dict, rank: int):
        """Display a single recommendation with full explanation"""
        
        device = self.device_lookup.get(device_id)
        if not device:
            return
        
        brand = device.get('brand', 'Unknown')
        model = device.get('model_name', 'Unknown')
        score = explanation.get('score', 0)
        confidence = explanation.get('confidence', 0)
        
        # Header
        print(f"\n{'─'*90}")
        print(f"#{rank} 📱 {brand.upper()} {model}")
        print(f"{'─'*90}")
        
        # Score bar visualization
        bar_length = 30
        filled = int(bar_length * score)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\n   ⭐ Recommendation Score: [{bar}] {score:.1%}")
        print(f"   💪 Confidence Level:    {confidence:.1%}")
        
        # Feature contributions breakdown (if available)
        if 'feature_contributions' in explanation:
            contribs = explanation['feature_contributions']
            print("\n   📈 Feature Contribution Breakdown:")
            print(f"      • Brand Match:           {contribs.get('brand_match', 0):.2f}/1.0")
            print(f"      • Price Fit:             {contribs.get('price_fit', 0):.2f}/1.0")
            print(f"      • Use Case Alignment:    {contribs.get('use_case_alignment', 0):.2f}/1.0")
            print(f"      • Specs Quality:         {contribs.get('specs_quality', 0):.2f}/1.0")
        
        # Specifications
        specs = explanation['specs']
        print("\n   📊 Key Specifications:")
        spec_items = [
            ('RAM', specs.get('ram'), 'GB'),
            ('Storage', specs.get('storage'), 'GB'),
            ('Camera', specs.get('camera'), 'MP'),
            ('Battery', specs.get('battery'), 'mAh'),
            ('Price', specs.get('price'), '$'),
        ]
        
        for spec_name, spec_val, unit in spec_items:
            if spec_val not in [None, 'N/A']:
                print(f"      • {spec_name:15s}: {spec_val:>10} {unit}")
        
        # Why recommended
        print("\n   🎯 Why Recommended:")
        for reason in explanation['reasons']:
            print(f"      {reason}")
    
    def test_prompt(self, prompt: str, top_n: int = 3, use_mcdm: bool = False):
        """Test a prompt and display enhanced recommendations"""
        
        print("\n" + "="*90)
        print("🔍 TESTING QUERY")
        print("="*90)
        print(f"\n📝 Your Query:\n   \"{prompt}\"\n")
        
        # Parse and recommend
        if self.recommender.use_semantic:
            print("🧠 Processing query with SEMANTIC BERT NLP...")
        else:
            print("🧠 Processing query with enhanced NLP...")
        
        preferences = {'query': prompt}
        
        try:
            recommendations = self.recommender.recommend_by_preferences(
                preferences, top_n=top_n, use_mcdm=use_mcdm
            )
            
            if not recommendations:
                print("\n❌ No suitable devices found for your requirements.\n")
                return
            
            # Display detection results
            print("✅ Query processed\n")
            print("📋 Analysis Results:")
            
            if preferences.get('use_case'):
                confidence = preferences.get('use_case_confidence', 0)
                print(f"   • Primary Use Case: {preferences['use_case'].upper()} (confidence: {confidence:.2f})")
            
            # Show multi-intent if available
            if preferences.get('multi_intent'):
                intents_display = ', '.join([f"{uc.capitalize()}({conf:.2f})" 
                                            for uc, conf in preferences['multi_intent'][:3]])
                print(f"   • Multi-Intent Detected: {intents_display}")
            
            if preferences.get('exclusions'):
                print(f"   • Exclusions: {', '.join(preferences['exclusions'])}")
            
            if preferences.get('implicit_reasoning'):
                print(f"   • Implicit Insights: {preferences['implicit_reasoning'][0]}")
            
            # Show extracted specs
            spec_keys = ['min_ram_gb', 'min_battery', 'min_camera_mp', 
                        'min_storage', 'budget', 'require_5g']
            found_specs = {k: preferences[k] for k in spec_keys if preferences.get(k)}
            if found_specs:
                print(f"   • Extracted Specs: {found_specs}")
            
            print("="*90)
            print("🏆 TOP RECOMMENDATIONS WITH EXPLANATIONS")
            print("="*90)
            
            for rank, (device_id, score, explanation) in enumerate(recommendations, 1):
                self._display_recommendation(device_id, explanation, rank)
            
            print(f"\n{'='*90}\n")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
    
    def interactive_mode(self):
        """Run interactive testing"""
        
        print("\n" + "="*90)
        print("💻 INTERACTIVE MODE - ENHANCED V2")
        print("="*90)
        print("\nEnter phone recommendation queries. Type 'quit' to exit.\n")
        
        while True:
            try:
                prompt = input("🎤 Enter your query:\n> ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for testing!\n")
                    break
                
                if not prompt:
                    print("⚠️ Please enter a valid query.\n")
                    continue
                
                # Ask about scoring method
                use_mcdm_input = input("Use TOPSIS multi-criteria scoring? (y/n, default=n): ").strip().lower()
                use_mcdm = use_mcdm_input in ['y', 'yes']
                
                self.test_prompt(prompt, top_n=3, use_mcdm=use_mcdm)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted.\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\n❌ Error: {e}\n")
                continue
    
    def batch_test_v2(self):
        """Test with comprehensive sample queries"""
        
        test_cases = [
            ("Gaming phone with 12GB RAM and 120Hz display under $1000", False),
            ("Professional photographer with $1500 budget", False),
            ("Budget traveler needing great battery life", False),
            ("Fast phone with 5G not from Apple", False),
            ("Gaming phone but NOT Samsung, under $800", False),
        ]
        
        print("\n" + "="*90)
        print("📂 BATCH TESTING - Enhanced V2 with Various Queries")
        print("="*90)
        
        for i, (prompt, use_mcdm) in enumerate(test_cases, 1):
            self.test_prompt(prompt, top_n=2, use_mcdm=use_mcdm)
            
            if i < len(test_cases):
                input("Press Enter to continue to next test...")


def main():
    """Main entry point"""
    
    print("\n" + "="*90)
    print("🚀 ENHANCED RECOMMENDATION SYSTEM V2 - TESTING TOOL")
    print("Priority 1-4 Optimizations + Semantic BERT NLP Implemented")
    print("="*90)
    print("\nOptions:")
    print("1. Interactive mode (enter queries one by one)")
    print("2. Batch test mode (run predefined queries)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        tester = InteractivePromptTesterV2(device_limit=1000)
        tester.interactive_mode()
    
    elif choice == "2":
        tester = InteractivePromptTesterV2(device_limit=1000)
        tester.batch_test_v2()
    
    elif choice == "3":
        print("\n👋 Exiting...\n")
        sys.exit(0)
    
    else:
        print("\n❌ Invalid option. Exiting...\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}\n")
        sys.exit(1)
