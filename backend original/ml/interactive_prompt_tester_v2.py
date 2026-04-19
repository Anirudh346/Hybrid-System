"""
INTERACTIVE PROMPT TESTER V2 - With Enhanced Recommendations & XAI
Tests the optimized recommendation system (Priority 1-4 implementations)
"""

import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dataset_loader import PhoneDatasetLoader
    from recommender import DeviceRecommender
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print(f"Make sure dataset_loader.py and recommender.py exist in {Path(__file__).parent}")
    sys.exit(1)
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
        print("With Priority 1-4 Optimizations")
        print("="*90)
        print("\n⏳ Loading dataset... (this may take a moment)")
        
        # Load dataset
        self.loader = PhoneDatasetLoader()
        self.devices = self.loader.load_csv_files(limit=device_limit)
        print(f"✅ Loaded {len(self.devices)} devices\n")
        
        # Train recommender V2
        print("🤖 Training enhanced recommendation engine...")
        self.recommender = DeviceRecommender()
        self.recommender.fit(self.devices)
        print("✅ Recommender ready\n")
        
        # Create device lookup
        self.device_lookup = {str(d.get('id', '')): d for d in self.devices}
    
    def _display_recommendation(self, device_id: str, explanation: dict, rank: int):
        """Display a single recommendation with full explanation"""
        
        device = self.device_lookup.get(device_id)
        if not device:
            return
        
        brand = device.get('brand', 'Unknown')
        model = device.get('model_name', 'Unknown')
        score = explanation['score']
        
        # Header
        print(f"\n{'─'*90}")
        print(f"#{rank} 📱 {brand.upper()} {model}")
        print(f"{'─'*90}")
        
        # Score bar visualization
        bar_length = 30
        filled = int(bar_length * score)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\n   ⭐ Score: [{bar}] {score:.1%}")
        
        # Specifications
        specs = explanation['specs']
        print("\n   📊 Key Specifications:")
        spec_items = [
            ('RAM', specs.get('ram'), 'GB'),
            ('Storage', specs.get('storage'), 'GB'),
            ('Camera', specs.get('camera'), 'MP'),
            ('Battery', specs.get('battery'), 'mAh'),
            ('Refresh', specs.get('refresh'), 'Hz'),
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
            use_case = self.recommender.nlp_parser.detect_use_case(prompt)[0]
            exclusions = self.recommender.nlp_parser.parse_exclusions(prompt)
            
            print("✅ Query processed\n")
            print("📋 Analysis Results:")
            print(f"   • Detected Use Case: {use_case.upper()}")
            if exclusions:
                print(f"   • Exclusions: {', '.join(exclusions)}")
            
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
    print("Priority 1-4 Optimizations Implemented")
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
