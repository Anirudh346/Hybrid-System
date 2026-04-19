import sys
sys.path.insert(0, 'C:\\Users\\aniru\\Downloads\\SmartAI-SmartDeviceFilter-main rudra\\SmartAI-SmartDeviceFilter-main\\SmartAI-SmartDeviceFilter-main\\backend')

from database import engine, SessionLocal
from sqlalchemy import inspect, text

# Get database connection
db = SessionLocal()

try:
    # Get actual columns from database
    result = db.execute(text("DESCRIBE devices"))
    columns = result.fetchall()
    
    print("Actual columns in 'devices' table in MySQL:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")
    
finally:
    db.close()
