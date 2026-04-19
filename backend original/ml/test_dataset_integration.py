"""
Integration test script for dataset loading and recommendation system
Tests the phone dataset integration with the recommendation engine
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
from utils.device_filter import DeviceFilter, SpecRequirements, UseCase, ComparisonHelper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test loading the phone dataset"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Dataset Loading")
    logger.info("="*70)
    
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=500)  # Load first 500 for testing
    
    logger.info(f"✅ Loaded {len(devices)} devices")
    
    # Print statistics
    stats = loader.get_statistics()
    logger.info("\n📊 Dataset Statistics:")
    logger.info(f"   Total Devices: {stats.get('total_devices')}")
    logger.info(f"   Device Types: {stats.get('by_type')}")
    logger.info(f"   Unique Brands: {stats.get('brands')}")
    logger.info(f"\n   Price Range: ${stats['price']['min']:.0f} - ${stats['price']['max']:.0f} (Avg: ${stats['price']['avg']:.0f})")
    logger.info(f"   RAM Range: {stats['ram']['min']:.0f} - {stats['ram']['max']:.0f}GB (Avg: {stats['ram']['avg']:.1f}GB)")
    logger.info(f"   Battery Range: {stats['battery']['min']:.0f} - {stats['battery']['max']:.0f}mAh (Avg: {stats['battery']['avg']:.0f}mAh)")
    
    # Show sample devices
    logger.info("\n📱 Sample Devices:")
    for i, device in enumerate(devices[:5]):
        logger.info(f"\n   {i+1}. {device['brand']} {device['model_name']}")
        specs = device.get('specs', {})
        logger.info(f"      Type: {device['device_type']}")
        logger.info(f"      RAM: {specs.get('ram_gb')}GB | Storage: {specs.get('storage_gb')}GB")
        logger.info(f"      Camera: {specs.get('main_camera_mp')}MP | Battery: {specs.get('battery_mah')}mAh")
        logger.info(f"      Refresh: {specs.get('refresh_rate_hz')}Hz | Price: ${specs.get('price')}")
    
    return devices


def test_feature_filtering(devices):
    """Test feature-based device filtering"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Feature-Based Filtering")
    logger.info("="*70)
    
    # Test 1: Gaming phones
    logger.info("\n🎮 Gaming Phones (8GB+ RAM, 120Hz+):")
    gaming_reqs = SpecRequirements(
        min_ram_gb=8,
        min_refresh_rate=120,
        min_battery_mah=5000
    )
    gaming_phones = DeviceFilter.filter_by_specs(devices, gaming_reqs)
    logger.info(f"   Found {len(gaming_phones)} gaming phones")
    for device in gaming_phones[:3]:
        specs = device['specs']
        logger.info(f"   - {device['brand']} {device['model_name']}: {specs.get('ram_gb')}GB, {specs.get('refresh_rate_hz')}Hz, ${specs.get('price')}")
    
    # Test 2: Budget phones
    logger.info("\n💰 Budget Phones (< $500):")
    budget_reqs = SpecRequirements(max_price=500)
    budget_phones = DeviceFilter.filter_by_specs(devices, budget_reqs)
    logger.info(f"   Found {len(budget_phones)} budget phones")
    for device in budget_phones[:3]:
        specs = device['specs']
        logger.info(f"   - {device['brand']} {device['model_name']}: ${specs.get('price')}")
    
    # Test 3: Camera phones
    logger.info("\n📷 Camera Phones (48MP+ main camera):")
    camera_reqs = SpecRequirements(min_camera_mp=48)
    camera_phones = DeviceFilter.filter_by_specs(devices, camera_reqs)
    logger.info(f"   Found {len(camera_phones)} camera phones")
    for device in camera_phones[:3]:
        specs = device['specs']
        logger.info(f"   - {device['brand']} {device['model_name']}: {specs.get('main_camera_mp')}MP camera")
    
    # Test 4: Battery phones
    logger.info("\n🔋 Battery Phones (5500mAh+ battery):")
    battery_reqs = SpecRequirements(min_battery_mah=5500)
    battery_phones = DeviceFilter.filter_by_specs(devices, battery_reqs)
    logger.info(f"   Found {len(battery_phones)} battery phones")
    for device in battery_phones[:3]:
        specs = device['specs']
        logger.info(f"   - {device['brand']} {device['model_name']}: {specs.get('battery_mah')}mAh")
    
    # Test 5: Brand filtering
    logger.info("\n🏢 Apple Devices Only:")
    apple_reqs = SpecRequirements(brands_include=['Apple'])
    apple_devices = DeviceFilter.filter_by_specs(devices, apple_reqs)
    logger.info(f"   Found {len(apple_devices)} Apple devices")
    for device in apple_devices[:3]:
        logger.info(f"   - {device['model_name']}")


def test_use_case_scoring(devices):
    """Test use-case based device scoring"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Use-Case Based Scoring")
    logger.info("="*70)
    
    use_cases = [UseCase.GAMING, UseCase.PHOTOGRAPHY, UseCase.BATTERY, UseCase.BUDGET]
    
    for use_case in use_cases:
        logger.info(f"\n🎯 Top devices for {use_case.value.upper()}:")
        
        scored_devices = []
        for device in devices:
            score = DeviceFilter.score_device_for_use_case(device, use_case)
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        
        for i, (device, score) in enumerate(scored_devices[:3]):
            logger.info(f"   {i+1}. {device['brand']} {device['model_name']} - Score: {score:.1f}/100")


def test_recommender_system(devices):
    """Test the recommendation system"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Recommendation Engine")
    logger.info("="*70)
    
    # Train recommender
    recommender = DeviceRecommender()
    recommender.fit(devices)
    logger.info("✅ Recommender trained successfully")
    
    # Test 1: Gaming preferences
    logger.info("\n🎮 Gaming Phone Recommendations (8GB RAM, 120Hz+):")
    gaming_prefs = {
        'use_case': 'gaming',
        'min_ram_gb': 8,
        'min_refresh_rate': 120,
    }
    gaming_recs = recommender.recommend_by_preferences(gaming_prefs, top_n=5)
    for device_id, score, _ in gaming_recs:
        device = next((d for d in devices if d.get('id') == device_id), None)
        if device:
            logger.info(f"   - {device['brand']} {device['model_name']}: {score:.3f}")
    
    # Test 2: Budget recommendations
    logger.info("\n💰 Budget Phone Recommendations (<$400):")
    budget_prefs = {
        'budget': 400,
        'use_case': 'budget'
    }
    budget_recs = recommender.recommend_by_preferences(budget_prefs, top_n=5)
    for device_id, score, _ in budget_recs:
        device = next((d for d in devices if d.get('id') == device_id), None)
        if device:
            specs = device['specs']
            logger.info(f"   - {device['brand']} {device['model_name']} (${specs.get('price')}): {score:.3f}")
    
    # Test 3: Photography recommendations
    logger.info("\n📷 Photography Phone Recommendations (48MP+ camera):")
    photo_prefs = {
        'use_case': 'photography',
        'min_camera_mp': 48,
    }
    photo_recs = recommender.recommend_by_preferences(photo_prefs, top_n=5)
    for device_id, score, _ in photo_recs:
        device = next((d for d in devices if d.get('id') == device_id), None)
        if device:
            specs = device['specs']
            logger.info(f"   - {device['brand']} {device['model_name']} ({specs.get('main_camera_mp')}MP): {score:.3f}")
    
    # Test 4: Battery recommendations
    logger.info("\n🔋 Battery Phone Recommendations (5500mAh+):")
    battery_prefs = {
        'use_case': 'battery',
        'min_battery': 5500,
    }
    battery_recs = recommender.recommend_by_preferences(battery_prefs, top_n=5)
    for device_id, score, _ in battery_recs:
        device = next((d for d in devices if d.get('id') == device_id), None)
        if device:
            specs = device['specs']
            logger.info(f"   - {device['brand']} {device['model_name']} ({specs.get('battery_mah')}mAh): {score:.3f}")
    
    # Test 5: Natural language query
    logger.info("\n🎯 Natural Language Query: 'Best flagship phone':")
    nl_prefs = {
        'query': 'best flagship phone'
    }
    nl_recs = recommender.recommend_by_preferences(nl_prefs, top_n=5)
    for device_id, score, _ in nl_recs:
        device = next((d for d in devices if d.get('id') == device_id), None)
        if device:
            logger.info(f"   - {device['brand']} {device['model_name']}: {score:.3f}")


def test_comparison_helper(devices):
    """Test device comparison helper"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Device Comparison")
    logger.info("="*70)
    
    # Get top 3 most expensive devices
    expensive = sorted(devices, key=lambda d: d['specs'].get('price', 0), reverse=True)[:3]
    
    logger.info("\n💎 Comparing Top 3 Most Expensive Devices:")
    comparison = ComparisonHelper.compare_devices(expensive)
    
    for i, device_info in enumerate(comparison['devices']):
        logger.info(f"   {i+1}. {device_info['brand']} {device_info['model']}")
    
    logger.info("\n📊 Specifications Comparison:")
    for spec_key, spec_data in comparison['specs'].items():
        logger.info(f"\n   {spec_key.upper()}:")
        for i, value in enumerate(spec_data['values']):
            logger.info(f"      Device {i+1}: {value}")


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("PHONE DATASET & RECOMMENDATION SYSTEM INTEGRATION TESTS")
    logger.info("="*70)
    
    try:
        # Test 1: Load dataset
        devices = test_dataset_loading()
        
        # Test 2: Feature filtering
        test_feature_filtering(devices)
        
        # Test 3: Use case scoring
        test_use_case_scoring(devices)
        
        # Test 4: Recommendation system
        test_recommender_system(devices)
        
        # Test 5: Comparison helper
        test_comparison_helper(devices)
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
