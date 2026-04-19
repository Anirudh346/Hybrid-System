#!/usr/bin/env python
"""Trace exact location of NoneType error"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
import logging
import traceback

logging.basicConfig(level=logging.WARNING)

print("Loading...")
loader = PhoneDatasetLoader()
devices = loader.load_csv_files(limit=1000)
print(f'Loaded {len(devices)} devices\n')

print("Training...")
recommender = DeviceRecommender()
recommender.fit(devices)
print('Recommender trained\n')

# Test prompts that cause the error
test_queries = [
    "I need a phone with 8GB RAM",
    "I want at least 8GB of RAM",
]

for prompt in test_queries:
    print(f"\nTesting: {prompt}")
    try:
        recs = recommender.recommend_by_preferences({'query': prompt}, top_n=3)
        print(f'  SUCCESS: Got {len(recs)} recommendations')
    except Exception as e:
        print(f'  ERROR: {e}')
        print("\n  Full traceback:")
        traceback.print_exc()
        break
