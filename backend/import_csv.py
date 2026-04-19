"""
CSV to MongoDB Import Script
Imports device data from GSMArenaDataset CSV files into MongoDB
"""

import asyncio
import csv
import os
import re
from datetime import datetime
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import sys

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent))

from models.device import Device, DeviceVariant
from config import settings


async def import_csv_to_mongodb(csv_folder_path: str, append: bool = False):
    """
    Import CSV files from GSMArenaDataset folder to MongoDB
    
    Args:
        csv_folder_path: Path to folder containing CSV files
        append: If True, append to existing data. If False, clear collection first
    """
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_url)
    database = client[settings.database_name]
    
    # Initialize Beanie
    await init_beanie(database=database, document_models=[Device])
    
    print(f"✅ Connected to MongoDB: {settings.database_name}")
    
    # Clear existing devices if not appending
    if not append:
        print("🗑️  Clearing existing devices...")
        await Device.delete_all()
        print("✅ Cleared existing devices")
    
    # Get all CSV files
    csv_folder = Path(csv_folder_path)
    if not csv_folder.exists():
        print(f"❌ Folder not found: {csv_folder_path}")
        return
    
    csv_files = list(csv_folder.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in: {csv_folder_path}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    total_imported = 0
    variant_pattern = re.compile(r'^(\d+)\s*(GB|TB)\s*(\d+)\s*GB\s*RAM$', re.IGNORECASE)
    
    for csv_file in csv_files:
        print(f"\n📄 Processing: {csv_file.name}")
        
        devices_to_insert = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Skip rows without required fields
                    if not row.get('Brand') or not row.get('Model Name'):
                        continue
                    
                    # Detect device type
                    model_name = row.get('Model Name', '').lower()
                    if 'watch' in model_name or 'band' in model_name:
                        device_type = 'smartwatch'
                    elif 'tablet' in model_name or 'ipad' in model_name or 'tab' in model_name:
                        device_type = 'tablet'
                    else:
                        device_type = 'mobile'
                    
                    # Parse variants from column names
                    variants = []
                    for key, value in row.items():
                        match = variant_pattern.match(key)
                        if match:
                            storage_val = int(match.group(1))
                            storage_unit = match.group(2).upper()
                            ram_val = int(match.group(3))
                            storage_in_gb = storage_val * 1024 if storage_unit == 'TB' else storage_val
                            
                            variant = DeviceVariant(
                                id=f"{storage_val}{storage_unit}-{ram_val}GB",
                                label=f"{storage_val}{storage_unit} / {ram_val}GB RAM",
                                storage=f"{storage_val}{storage_unit}",
                                storage_in_gb=storage_in_gb,
                                ram=f"{ram_val}GB",
                                ram_in_gb=ram_val,
                                price=value if value else None
                            )
                            variants.append(variant)
                    
                    # Create device document
                    device = Device(
                        brand=row['Brand'],
                        model_name=row['Model Name'],
                        model_image=row.get('Model Image'),
                        device_type=device_type,
                        specs={k: v for k, v in row.items() 
                               if k not in ['Brand', 'Model Name', 'Model Image'] 
                               and not variant_pattern.match(k)},
                        variants=variants,
                        scraped_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    
                    devices_to_insert.append(device)
            
            # Bulk insert devices from this CSV
            if devices_to_insert:
                await Device.insert_many(devices_to_insert)
                print(f"   ✅ Imported {len(devices_to_insert)} devices")
                total_imported += len(devices_to_insert)
            else:
                print(f"   ⚠️  No devices found")
                
        except Exception as e:
            print(f"   ❌ Error processing {csv_file.name}: {str(e)}")
            continue
    
    print(f"\n🎉 Import complete! Total devices imported: {total_imported}")
    
    # Close connection
    client.close()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import CSV data to MongoDB')
    parser.add_argument(
        'csv_folder',
        type=str,
        help='Path to folder containing CSV files (e.g., GSMArenaDataset/)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing data instead of replacing'
    )
    
    args = parser.parse_args()
    
    await import_csv_to_mongodb(args.csv_folder, append=args.append)


if __name__ == "__main__":
    asyncio.run(main())
