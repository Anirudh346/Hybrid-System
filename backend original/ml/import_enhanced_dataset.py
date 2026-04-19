"""
Enhanced CSV to MongoDB Import Script
Imports device data from GSMArenaDataset CSV files into MongoDB with feature extraction
"""

import asyncio
import sys
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import logging

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.device import Device, DeviceVariant
from config import settings
from ml.dataset_loader import PhoneDatasetLoader
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def import_dataset_to_mongodb(dataset_path: str = None, append: bool = False, limit: int = None):
    """
    Import GSMArena dataset to MongoDB with enhanced feature extraction
    
    Args:
        dataset_path: Path to GSMArenaDataset folder
        append: If True, append to existing data. If False, clear collection first
        limit: Maximum number of devices to import (for testing)
    """
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_url)
    database = client[settings.database_name]
    
    # Initialize Beanie
    await init_beanie(database=database, document_models=[Device])
    
    logger.info(f"✅ Connected to MongoDB: {settings.database_name}")
    
    # Clear existing devices if not appending
    if not append:
        logger.info("🗑️  Clearing existing devices...")
        await Device.delete_all()
        logger.info("✅ Cleared existing devices")
    
    # Load dataset
    logger.info("📁 Loading dataset...")
    loader = PhoneDatasetLoader(dataset_path)
    devices_data = loader.load_csv_files(limit=limit)
    
    if not devices_data:
        logger.error("❌ No devices loaded from dataset")
        client.close()
        return
    
    logger.info(f"📊 Dataset Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # Process and insert devices
    logger.info(f"💾 Importing {len(devices_data)} devices to MongoDB...")
    
    devices_to_insert = []
    variant_pattern = re.compile(r'^(\d+)\s*(GB|TB)\s*(\d+)\s*GB\s*RAM$', re.IGNORECASE)
    
    for device_data in devices_data:
        # Extract variants from specifications
        variants = []
        
        # Try to create variants from available storage/RAM combinations
        specs = device_data.get('specs', {})
        
        # Create a default variant if we have pricing info
        if specs.get('storage_gb', 0) > 0 and specs.get('ram_gb', 0) > 0:
            storage_gb = int(specs.get('storage_gb', 0))
            ram_gb = int(specs.get('ram_gb', 0))
            
            variant = DeviceVariant(
                id=f"{storage_gb}GB-{ram_gb}GB",
                label=f"{storage_gb}GB / {ram_gb}GB RAM",
                storage=f"{storage_gb}GB",
                storage_in_gb=storage_gb,
                ram=f"{ram_gb}GB",
                ram_in_gb=ram_gb,
                price=str(specs.get('price', 0)) if specs.get('price', 0) > 0 else None
            )
            variants.append(variant)
        
        # Create Device document
        device = Device(
            brand=device_data['brand'],
            model_name=device_data['model_name'],
            model_image=device_data.get('model_image', ''),
            device_type=device_data['device_type'],
            specs=specs,
            variants=variants if variants else [],
        )
        
        devices_to_insert.append(device)
    
    # Bulk insert
    try:
        if devices_to_insert:
            await Device.insert_many(devices_to_insert)
            logger.info(f"✅ Imported {len(devices_to_insert)} devices")
        else:
            logger.warning("⚠️  No devices to insert")
    except Exception as e:
        logger.error(f"❌ Error importing devices: {str(e)}")
    
    # Verify import
    count = await Device.count()
    logger.info(f"📈 Total devices in database: {count}")
    
    # Print sample devices
    logger.info("\n📱 Sample imported devices:")
    sample_devices = await Device.find_all().limit(5).to_list()
    for device in sample_devices:
        logger.info(f"   - {device.brand} {device.model_name}")
        if device.specs:
            logger.info(f"     RAM: {device.specs.get('ram_gb', 'N/A')}GB")
            logger.info(f"     Storage: {device.specs.get('storage_gb', 'N/A')}GB")
            logger.info(f"     Camera: {device.specs.get('main_camera_mp', 'N/A')}MP")
            logger.info(f"     Battery: {device.specs.get('battery_mah', 'N/A')}mAh")
            logger.info(f"     Price: ${device.specs.get('price', 'N/A')}")
    
    logger.info("\n🎉 Import complete!")
    
    # Close connection
    client.close()


async def test_feature_extraction(dataset_path: str = None):
    """Test feature extraction on a sample"""
    logger.info("🧪 Testing feature extraction...")
    
    loader = PhoneDatasetLoader(dataset_path)
    devices = loader.load_csv_files(limit=10)
    
    if devices:
        device = devices[0]
        logger.info(f"\n📱 Sample Device: {device['brand']} {device['model_name']}")
        logger.info(f"   Device Type: {device['device_type']}")
        logger.info(f"\n   Extracted Specs:")
        for key, value in device['specs'].items():
            if value or isinstance(value, bool):
                logger.info(f"   - {key}: {value}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import enhanced dataset to MongoDB')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to GSMArenaDataset folder (auto-detected if not provided)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing data instead of replacing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of devices to import (for testing)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test feature extraction without importing'
    )
    
    args = parser.parse_args()
    
    if args.test:
        await test_feature_extraction(args.dataset_path)
    else:
        await import_dataset_to_mongodb(args.dataset_path, append=args.append, limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
