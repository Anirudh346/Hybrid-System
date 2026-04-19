"""
Test XAI Explainer with sample data
Run this to see how the XAI module generates explanations
"""

from ml.xai_explainer import xai_explainer

# Sample device data
sample_device = {
    'id': '1',
    'brand': 'Samsung',
    'model_name': 'Galaxy S23 Ultra',
    'device_type': 'mobile',
    'specs': {
        'Chipset': 'Snapdragon 8 Gen 2',
        'Internal': '256GB 12GB RAM',
        'Main Camera': '200 MP + 10 MP + 10 MP + 12 MP',
        'Selfie camera': '12 MP',
        'Battery': '5000 mAh',
        'Charging': '45W wired',
        'Display': '6.8" Dynamic AMOLED 2X, 120Hz',
    },
    'variants': [
        {'price': 1199, 'storage': '256GB', 'ram': '12GB'}
    ]
}

sample_preferences = {
    'budget': 1200,
    'use_case': 'gaming',
    'brand_preference': ['Samsung', 'OnePlus']
}

# All devices for comparison
all_devices = [
    sample_device,
    {
        'id': '2',
        'brand': 'OnePlus',
        'model_name': '11 Pro',
        'device_type': 'mobile',
        'specs': {
            'Chipset': 'Snapdragon 8 Gen 2',
            'Internal': '256GB 16GB RAM',
            'Main Camera': '50 MP + 32 MP + 48 MP',
            'Battery': '5000 mAh',
            'Charging': '100W wired',
            'Display': '6.7" AMOLED, 120Hz',
        },
        'variants': [
            {'price': 899, 'storage': '256GB', 'ram': '16GB'}
        ]
    }
]

# Generate explanation
explanation = xai_explainer.explain_recommendation(
    device=sample_device,
    preferences=sample_preferences,
    score=0.85,
    all_devices=all_devices
)

# Print results
print("=" * 80)
print("XAI EXPLANATION DEMO")
print("=" * 80)
print(f"\nDevice: {sample_device['brand']} {sample_device['model_name']}")
print(f"Score: {explanation.overall_score:.2f}")
print(f"Confidence: {explanation.confidence:.2f}")
print(f"\n{explanation.match_summary}")

print(f"\n{'TOP REASONS':=^80}")
for i, reason in enumerate(explanation.top_reasons, 1):
    print(f"{i}. {reason}")

print(f"\n{'FEATURE CONTRIBUTIONS':=^80}")
for contrib in explanation.feature_contributions:
    weighted_score = contrib.contribution_score * contrib.importance
    print(f"\n{contrib.feature_name}: {contrib.value}")
    print(f"  Score: {contrib.contribution_score:.2f} × Weight: {contrib.importance:.2f} = {weighted_score:.3f}")
    print(f"  {contrib.explanation}")

if explanation.comparable_alternatives:
    print(f"\n{'COMPARABLE ALTERNATIVES':=^80}")
    for alt in explanation.comparable_alternatives:
        print(f"\n{alt['brand']} {alt['model_name']} - ${alt['price']:.0f}")
        print(f"  {alt['reason']}")

if explanation.counterfactual:
    print(f"\n{'COUNTERFACTUAL':=^80}")
    print(explanation.counterfactual)

print("\n" + "=" * 80)
print("\n✅ XAI is working! Users will see these detailed explanations for every recommendation.")
print("\nNext steps:")
print("1. Set up MongoDB to test with real data")
print("2. Import device data from CSV files")
print("3. Test API endpoint: POST /api/recommendations with explain=true")
