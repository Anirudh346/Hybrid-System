import sys
sys.path.insert(0, 'C:\\Users\\aniru\\Downloads\\SmartAI-SmartDeviceFilter-main rudra\\SmartAI-SmartDeviceFilter-main\\SmartAI-SmartDeviceFilter-main\\backend')

from database import engine, SessionLocal
from models.device import Device
from sqlalchemy import inspect, text

# Get database connection
db = SessionLocal()

try:
    # Check if table exists
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables in database: {tables}")
    
    if 'devices' in tables:
        columns = inspector.get_columns('devices')
        print(f"\nColumns in 'devices' table:")
        for col in columns[:10]:
            print(f"  - {col['name']}: {col['type']}")
        print(f"  ... ({len(columns)} total columns)")
        
        # Check row count
        count = db.query(Device).count()
        print(f"\nTotal devices in database: {count}")
        
        if count > 0:
            device = db.query(Device).first()
            print(f"\nFirst device:")
            print(f"  ID: {device.id}")
            print(f"  Brand: {device.brand}")
            print(f"  Model: {device.model_name}")
            print(f"  Battery: {device.battery_capacity}")
    else:
        print("'devices' table not found!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
