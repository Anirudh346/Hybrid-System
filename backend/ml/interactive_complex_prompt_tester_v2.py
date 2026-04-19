"""
INTERACTIVE COMPLEX PROMPT TESTER V2
Runs the semantic recommender against the complex prompt suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
from ml.test_complex_prompts import COMPLEX_PROMPTS
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveComplexPromptTesterV2:
    """Interactive tester that can run the complex prompt list in batch."""

    def __init__(self, device_limit: int = 5000):
        """Initialize the tester."""
        print("\n" + "=" * 90)
        print("ENHANCED PHONE RECOMMENDATION SYSTEM V2 - COMPLEX PROMPT TESTER")
        print("Priority 1-4 Optimizations + Semantic BERT NLP")
        print("=" * 90)
        print("\nLoading dataset... (this may take a moment)")

        # Load dataset
        self.loader = PhoneDatasetLoader()
        self.devices = self.loader.load_csv_files(limit=device_limit)
        print(f"Loaded {len(self.devices)} devices\n")

        # Train recommender
        print("Training enhanced recommendation engine with semantic embeddings...")
        self.recommender = DeviceRecommender(use_semantic=True)
        self.recommender.fit(self.devices)

        if self.recommender.use_semantic:
            print("Recommender ready with SEMANTIC NLP enabled\n")
        else:
            print("Recommender ready (semantic NLP not available, using legacy)\n")

        # Create device lookup
        self.device_lookup = {str(d.get('id', '')): d for d in self.devices}

    def _display_recommendation(self, device_id: str, explanation: dict, rank: int):
        """Display a single recommendation with explanation."""
        device = self.device_lookup.get(device_id)
        if not device:
            return

        brand = device.get('brand', 'Unknown')
        model = device.get('model_name', 'Unknown')
        score = explanation.get('score', 0)
        confidence = explanation.get('confidence', 0)

        print(f"\n{'-' * 90}")
        print(f"#{rank} {brand.upper()} {model}")
        print(f"{'-' * 90}")

        # Score with visual bar
        bar_length = 30
        filled = int(bar_length * score)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\n   ⭐ Recommendation Score: [{bar}] {score:.1%}")
        print(f"   💪 Confidence Level:    {confidence:.1%}")

        # Feature contributions breakdown
        if 'feature_contributions' in explanation:
            contribs = explanation['feature_contributions']
            print(f"\n   📈 Feature Contribution Breakdown:")
            print(f"      • Brand Match:           {contribs.get('brand_match', 0):.2f}/1.0")
            print(f"      • Price Fit:             {contribs.get('price_fit', 0):.2f}/1.0")
            print(f"      • Use Case Alignment:    {contribs.get('use_case_alignment', 0):.2f}/1.0")
            print(f"      • Specs Quality:         {contribs.get('specs_quality', 0):.2f}/1.0")

        specs = explanation['specs']
        print("\n   📊 Key Specifications:")
        spec_items = [
            ('Processor', specs.get('processor'), ''),
            ('RAM', specs.get('ram'), 'GB'),
            ('Storage', specs.get('storage'), 'GB'),
            ('Camera', specs.get('camera'), 'MP'),
            ('Battery', specs.get('battery'), 'mAh'),
            ('Price', specs.get('price'), '$'),
        ]

        for spec_name, spec_val, unit in spec_items:
            if spec_val not in [None, 'N/A', '']:
                if unit:
                    print(f"      - {spec_name:15s}: {spec_val:>10} {unit}")
                else:
                    print(f"      - {spec_name:15s}: {spec_val}")

        print("\n   🎯 Why Recommended:")
        for reason in explanation['reasons']:
            print(f"      {reason}")

    def test_prompt(self, prompt: str, top_n: int = 3, use_mcdm: bool = False):
        """Test a prompt and display enhanced recommendations."""
        print("\n" + "=" * 90)
        print("TESTING QUERY")
        print("=" * 90)
        print(f"\nQuery:\n  \"{prompt}\"\n")

        if self.recommender.use_semantic:
            print("Processing query with SEMANTIC BERT NLP...")
        else:
            print("Processing query with enhanced NLP...")

        preferences = {'query': prompt}

        try:
            recommendations = self.recommender.recommend_by_preferences(
                preferences, top_n=top_n, use_mcdm=use_mcdm
            )

            if not recommendations:
                print("\nNo suitable devices found for your requirements.\n")
                if 'gaming' in str(preferences.get('use_case', '')).lower() and preferences.get('brand_preference'):
                    print("Tip: No gaming-capable phones were found for that brand in the current dataset constraints.")
                return

            print("\nAnalysis Results:")
            if preferences.get('use_case'):
                confidence = preferences.get('use_case_confidence', 0)
                print(f"  - Primary Use Case: {preferences['use_case'].upper()} (confidence: {confidence:.2f})")

            if preferences.get('multi_intent'):
                intents_display = ', '.join([
                    f"{uc.capitalize()}({conf:.2f})" for uc, conf in preferences['multi_intent'][:3]
                ])
                print(f"  - Multi-Intent Detected: {intents_display}")

            if preferences.get('exclusions'):
                print(f"  - Exclusions: {', '.join(preferences['exclusions'])}")

            if preferences.get('implicit_reasoning'):
                print(f"  - Implicit Insights: {preferences['implicit_reasoning'][0]}")

            spec_keys = [
                'min_ram_gb', 'min_battery', 'min_camera_mp',
                'min_storage', 'budget', 'require_5g'
            ]
            found_specs = {k: preferences[k] for k in spec_keys if preferences.get(k)}
            if found_specs:
                print(f"  - Extracted Specs: {found_specs}")

            print("\n" + "=" * 90)
            print("TOP RECOMMENDATIONS WITH EXPLANATIONS")
            print("=" * 90)

            for rank, (device_id, score, explanation) in enumerate(recommendations, 1):
                self._display_recommendation(device_id, explanation, rank)

            print(f"\n{'=' * 90}\n")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            print(f"\nError: {e}\n")

    def batch_test_complex_prompts(self):
        """Run all prompts from the complex prompt suite."""
        print("\n" + "=" * 90)
        print("BATCH TESTING - COMPLEX PROMPTS")
        print("=" * 90)

        for i, prompt in enumerate(COMPLEX_PROMPTS, 1):
            clean_prompt = ' '.join(prompt.split())
            print(f"\nPrompt {i}/{len(COMPLEX_PROMPTS)}")
            self.test_prompt(clean_prompt, top_n=3, use_mcdm=False)

            if i < len(COMPLEX_PROMPTS):
                input("Press Enter to continue to next test...")

    def interactive_mode(self):
        """Run interactive testing."""
        print("\n" + "=" * 90)
        print("INTERACTIVE MODE - COMPLEX PROMPT TESTER")
        print("=" * 90)
        print("\nEnter phone recommendation queries. Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input("Enter your query:\n> ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for testing!\n")
                    break

                if not prompt:
                    print("Please enter a valid query.\n")
                    continue

                use_mcdm_input = input("Use TOPSIS multi-criteria scoring? (y/n, default=n): ").strip().lower()
                use_mcdm = use_mcdm_input in ['y', 'yes']

                self.test_prompt(prompt, top_n=3, use_mcdm=use_mcdm)

            except KeyboardInterrupt:
                print("\n\nSession interrupted.\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\nError: {e}\n")
                continue


def main():
    """Main entry point."""
    print("\n" + "=" * 90)
    print("ENHANCED RECOMMENDATION SYSTEM V2 - COMPLEX PROMPT TEST TOOL")
    print("Priority 1-4 Optimizations + Semantic BERT NLP Implemented")
    print("=" * 90)
    print("\nOptions:")
    print("1. Interactive mode (enter queries one by one)")
    print("2. Batch test mode (run complex prompt suite)")
    print("3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        tester = InteractiveComplexPromptTesterV2(device_limit=5000)
        tester.interactive_mode()

    elif choice == "2":
        tester = InteractiveComplexPromptTesterV2(device_limit=5000)
        tester.batch_test_complex_prompts()

    elif choice == "3":
        print("\nExiting...\n")
        sys.exit(0)

    else:
        print("\nInvalid option. Exiting...\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}\n")
        sys.exit(1)
