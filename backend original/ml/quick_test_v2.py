"""Quick test of V2 recommender improvements"""

import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender

print('\n' + '='*90)
print('TESTING ENHANCED RECOMMENDER V2 - PRIORITY 1-4 OPTIMIZATIONS')
print('='*90)

# Load dataset
loader = PhoneDatasetLoader()
devices = loader.load_csv_files(limit=500)
print(f'\nLoaded {len(devices)} devices')

# Train
print('\nTraining recommender...')
recommender = DeviceRecommender()
recommender.fit(devices)
print('Ready')

# Test cases
test_queries = [
    ('Gaming phone with 12GB RAM and 120Hz display under $1000', False),
    ('Best camera phone for $500', False),
    ('Battery phone not Samsung under $400', False),
    ('Display phone with 120Hz refresh', False),
    ('Photography - professional with $1500 budget', False),
]

for i, (query, use_mcdm) in enumerate(test_queries, 1):
    print(f'\n{"-"*90}')
    print(f'TEST {i}: "{query}"')
    print(f'Use TOPSIS: {use_mcdm}')
    print('-'*90)
    
    try:
        results = recommender.recommend_by_preferences({'query': query}, top_n=2, use_mcdm=use_mcdm)
        
        if results:
            for rank, (device_id, score, explanation) in enumerate(results, 1):
                device = next((d for d in devices if str(d.get('id')) == device_id), {})
                brand = device.get('brand', 'Unknown')
                model = device.get('model_name', 'Unknown')
                
                print(f'\n  #{rank} {brand} {model}')
                print(f'      Score: {score:.1%}')
                print(f'      Reasons:')
                for reason in explanation['reasons'][:3]:
                    print(f'        {reason}')
        else:
            print('  No recommendations found')
    
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

print(f'\n{"="*90}')
print('Testing complete!')
print('='*90 + '\n')
