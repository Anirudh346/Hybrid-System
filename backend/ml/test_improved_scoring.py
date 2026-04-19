"""
Test script to demonstrate improved XAI-aligned scoring in recommender.py
Compares old vs new scoring approach
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.recommender import DeviceRecommender, DataImputer
from ml.dataset_loader import PhoneDatasetLoader
import json

def test_improved_scoring():
    """Test the new XAI-aligned scoring mechanism"""
    
    print("\n" + "="*90)
    print("IMPROVED SCORING TEST - XAI-Aligned Feature Contribution Scoring")
    print("="*90)
    
    # Load dataset
    print("\nLoading dataset...")
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=500)
    print(f"Loaded {len(devices)} devices")
    
    # Train recommender
    print("\nTraining recommender with improved scoring...")
    recommender = DeviceRecommender(use_semantic=False)  # Use TF-IDF for clarity
    recommender.fit(devices)
    
    # Test Case 1: Gaming Preference
    print("\n" + "-"*90)
    print("TEST 1: Gaming Phone Recommendation")
    print("-"*90)
    
    gaming_prefs = {
        'query': 'Best phone for gaming with high refresh rate and fast processor',
        'use_case': 'gaming',
        'use_case_confidence': 0.85,
        'budget': 800,
        'min_ram_gb': 8
    }
    
    print("\nQuery: Best phone for gaming with high refresh rate and fast processor")
    print("Preferences:")
    print(f"  - Use Case: Gaming (confidence: 0.85)")
    print(f"  - Budget: $800")
    print(f"  - Min RAM: 8GB")
    
    results = recommender.recommend_by_preferences(gaming_prefs, top_n=3)
    
    for rank, (device_id, score, explanation) in enumerate(results, 1):
        device = recommender.raw_devices[[i for i, d in enumerate(recommender.raw_devices) if str(d.get('id')) == device_id][0]]
        print(f"\n  #{rank} - Score: {score:.2%}")
        print(f"      Device: {device.get('brand')} {device.get('model_name')}")
        print(f"      Confidence: {explanation.get('confidence', 0):.2%}")
        if 'feature_contributions' in explanation:
            contribs = explanation['feature_contributions']
            print(f"      Feature Contributions:")
            print(f"        - Brand Match: {contribs.get('brand_match', 0):.2f}")
            print(f"        - Price Fit: {contribs.get('price_fit', 0):.2f}")
            print(f"        - Use Case Alignment: {contribs.get('use_case_alignment', 0):.2f}")
            print(f"        - Specs Quality: {contribs.get('specs_quality', 0):.2f}")
        print(f"      Top Reasons:")
        for i, reason in enumerate(explanation.get('reasons', [])[:3], 1):
            print(f"        {i}. {reason}")
    
    # Test Case 2: Photography Preference
    print("\n" + "-"*90)
    print("TEST 2: Photography Phone Recommendation")
    print("-"*90)
    
    photo_prefs = {
        'query': 'Best camera phone for professional photography',
        'use_case': 'photography',
        'use_case_confidence': 0.8,
        'budget': 1000,
    }
    
    print("\nQuery: Best camera phone for professional photography")
    print("Preferences:")
    print(f"  - Use Case: Photography (confidence: 0.8)")
    print(f"  - Budget: $1000")
    
    results = recommender.recommend_by_preferences(photo_prefs, top_n=3)
    
    for rank, (device_id, score, explanation) in enumerate(results, 1):
        device = recommender.raw_devices[[i for i, d in enumerate(recommender.raw_devices) if str(d.get('id')) == device_id][0]]
        print(f"\n  #{rank} - Score: {score:.2%}")
        print(f"      Device: {device.get('brand')} {device.get('model_name')}")
        print(f"      Confidence: {explanation.get('confidence', 0):.2%}")
        if 'feature_contributions' in explanation:
            contribs = explanation['feature_contributions']
            print(f"      Feature Contributions:")
            print(f"        - Use Case Alignment: {contribs.get('use_case_alignment', 0):.2f}")
            print(f"        - Specs Quality: {contribs.get('specs_quality', 0):.2f}")
        print(f"      Top Reasons:")
        for i, reason in enumerate(explanation.get('reasons', [])[:3], 1):
            print(f"        {i}. {reason}")
    
    # Test Case 3: Budget Preference
    print("\n" + "-"*90)
    print("TEST 3: Budget Phone Recommendation")
    print("-"*90)
    
    budget_prefs = {
        'query': 'Best affordable phone under 300 dollars',
        'budget': 300,
        'use_case_confidence': 0.3,
    }
    
    print("\nQuery: Best affordable phone under 300 dollars")
    print("Preferences:")
    print(f"  - Budget: $300")
    
    results = recommender.recommend_by_preferences(budget_prefs, top_n=3)
    
    for rank, (device_id, score, explanation) in enumerate(results, 1):
        device = recommender.raw_devices[[i for i, d in enumerate(recommender.raw_devices) if str(d.get('id')) == device_id][0]]
        print(f"\n  #{rank} - Score: {score:.2%}")
        print(f"      Device: {device.get('brand')} {device.get('model_name')}")
        print(f"      Specs: RAM={device.get('specs', {}).get('ram_gb')}GB, Storage={device.get('specs', {}).get('storage_gb')}GB")
        
    print("\n" + "="*90)
    print("Key Improvements:")
    print("="*90)
    print("✓ Feature Contribution Scoring: Explicit weighting of brand, price, use case, specs")
    print("✓ Weighted Importance: Features weighted by relevance to preferences")
    print("✓ Use-Case Alignment: Specific spec evaluation for gaming, photography, battery, etc.")
    print("✓ Confidence Scoring: Explainability through confidence metrics")
    print("✓ Better Explanations: Detailed reasons based on feature analysis")
    print("✓ Hybrid Approach: Blends textual/semantic scores with feature contributions")
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    try:
        test_improved_scoring()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
