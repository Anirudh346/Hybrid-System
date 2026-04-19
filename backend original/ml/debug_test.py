#!/usr/bin/env python
"""Debug test for NoneType error"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Loading...")
loader = PhoneDatasetLoader()
devices = loader.load_csv_files(limit=1000)
print(f'Loaded {len(devices)} devices')

print("Checking device specs...")
for i, d in enumerate(devices[:5]):
    specs = d.get('specs', {})
    print(f"Device {i}: {d.get('brand')} {d.get('model_name')}")
    print(f"  Specs type: {type(specs)}")
    if isinstance(specs, dict):
        print(f"  - ram_gb: {specs.get('ram_gb')} (type: {type(specs.get('ram_gb'))})")
        print(f"  - battery: {specs.get('battery_mah')} (type: {type(specs.get('battery_mah'))})")
        print(f"  - storage: {specs.get('storage_gb')} (type: {type(specs.get('storage_gb'))})")

print("\nTraining...")
recommender = DeviceRecommender()
recommender.fit(devices)
print('Recommender trained')

print("\nTesting battery query...")
try:
    recs = recommender.recommend_by_preferences({'query': 'I need a good battery phone, 5000mAh+'}, top_n=3)
    print(f'SUCCESS: Got {len(recs)} recommendations')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
