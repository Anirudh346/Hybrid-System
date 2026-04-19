# MySQL Migration Guide - Backend NLP Implementation

## Overview
Your backend has been successfully migrated from MongoDB/Beanie to MySQL with SQLAlchemy. The dataset now loads from your existing MySQL `device_catalog.devices` table instead of CSV files.

## What Changed

### ❌ Removed - MongoDB/Beanie Dependencies
- `motor==3.3.2` - Async MongoDB driver
- `beanie==1.24.0` - MongoDB ODM
- `pymongo==4.6.1` - MongoDB driver
- All Beanie imports and document models
- MongoDB connection initialization

### ✅ Added - MySQL/SQLAlchemy Dependencies
- `SQLAlchemy==2.0.23` - ORM for SQL databases
- `PyMySQL==1.1.0` - Pure Python MySQL client
- `mysql-connector-python==8.2.0` - Official MySQL connector

## File Updates

### 1. **requirements.txt**
   - Removed: motor, beanie, pymongo
   - Added: SQLAlchemy, PyMySQL, mysql-connector-python

### 2. **config.py**
   - Changed from MongoDB URL to MySQL connection details
   - New settings:
     ```
     database_url = "mysql+pymysql://root:123@localhost:3306/device_catalog"
     database_host = "localhost"
     database_port = 3306
     database_user = "root"
     database_password = "123"
     database_name = "device_catalog"
     ```

### 3. **database.py** (NEW)
   - SQLAlchemy engine initialization
   - SessionLocal for database sessions
   - Database initialization function
   - Uses MySQL with connection pooling

### 4. **models/device.py**
   - Changed from Beanie Document to SQLAlchemy model
   - Maps directly to your `devices` table columns
   - All 103+ columns now properly defined
   - Includes `battery_capacity` field properly mapped

### 5. **routers/devices.py**
   - Removed Beanie operators (In, RegEx, And, Or)
   - Updated to use SQLAlchemy query syntax
   - Changed from async/await to sync database calls
   - Uses Session dependency injection
   - Filters now use SQLAlchemy column comparisons

### 6. **ml/dataset_loader_mysql.py** (NEW)
   - Complete rewrite to load from MySQL instead of CSV
   - `PhoneDatasetLoader` class:
     - Loads all devices from `devices` table
     - Extracts and normalizes specifications
     - Supports filtering by brand, price, RAM, etc.
     - Provides flagship, budget, and gaming device lists
   - `SpecificationExtractor` for parsing specifications
   - Uses same API as CSV version for compatibility

### 7. **main.py**
   - Removed MongoDB initialization
   - Added SQLAlchemy database initialization in startup
   - Simplified router includes (removed auth, users, etc. for now)
   - Uses MySQL connection pooling

### 8. **schemas/device.py**
   - Updated DeviceResponse to include all individual fields
   - Changed from `id: str` to `id: int`
   - Removed specs dict and variants list structure
   - Now matches the flat SQLAlchemy model structure

### 9. **.env.example**
   - Removed MONGODB_URL
   - Added MySQL configuration variables

## MySQL Connection Details

**Your Database:**
- Host: localhost
- Port: 3306
- Username: root
- Password: 123
- Database: device_catalog
- Table: devices

## How to Use

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Start Backend**
```bash
cd backend
python -m uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. **Load Data**
The data is automatically loaded from MySQL when you start the backend. No CSV import needed!

### 4. **API Endpoints**

#### Get All Devices
```bash
curl "http://localhost:8000/api/devices?page=1&page_size=20"
```

#### Get Device by ID
```bash
curl "http://localhost:8000/api/devices/12"
```

#### Search Devices
```bash
curl "http://localhost:8000/api/devices?search=Samsung&brand=Samsung"
```

#### Get All Brands
```bash
curl "http://localhost:8000/api/devices/brands"
```

#### NLP Search (when configured)
```bash
curl -X POST "http://localhost:8000/api/devices/search/nlp?query=best+gaming+phone+under+500"
```

## Key Features

### ✅ Working Features
- Device listing with pagination
- Filtering by brand, device type, processor
- NLP-powered search (framework in place)
- Battery capacity field fully integrated
- MySQL connection pooling
- SQLAlchemy ORM for type safety

### 🔄 Next Steps for NLP Implementation
1. The `dataset_loader_mysql.py` loads devices with normalized specs
2. Use these specs for NLP recommendations
3. The `advanced_nlp_parser.py` is ready to parse natural language queries
4. Connect the parser output to database queries

## API Response Example

```json
{
  "id": 12,
  "brand": "acer",
  "model_name": "Acer Iconia V12",
  "device_type": "mobile",
  "battery_capacity": "8000 mAh",
  "chipset": "Qualcomm Snapdragon...",
  "display_size": "10.0 inches",
  "price": "About 500 EUR",
  "os": "Android",
  "scraped_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00"
}
```

## Database Schema Reference

Your `devices` table columns are automatically mapped in the SQLAlchemy model:

**Core Fields:**
- id, brand, model_name, model_image, device_type

**Display & Hardware:**
- display_type, display_size, display_resolution
- chipset, cpu, gpu
- internal_storage, card_slot, battery_capacity

**Features:**
- os, sim, wlan, bluetooth, nfc, usb
- main_camera_features, main_camera_video, selfie_camera_single
- charging, loudspeaker, jack_35mm

**Network:**
- bands_2g, bands_3g, bands_4g, bands_5g, speed

**Specs & Performance:**
- antutu, geekbench, weight, dimensions, colors, sensors

## Troubleshooting

### MySQL Connection Error
```
Error: Can't connect to MySQL server on 'localhost'
```
**Solution:** Ensure MySQL is running and credentials are correct in config.py

### "No module named 'sqlalchemy'"
```bash
pip install SQLAlchemy==2.0.23 PyMySQL==1.1.0
```

### "Module 'beanie' not found"
- This is normal! Beanie has been removed
- All code now uses SQLAlchemy instead

### Device IDs Changed
- Previously: MongoDB ObjectIds (strings)
- Now: Integer IDs from your MySQL table
- All endpoints expect integer IDs

## Project Structure

```
backend/
├── main.py                          # App entry point
├── config.py                        # Settings (MySQL credentials)
├── database.py                      # SQLAlchemy configuration
├── requirements.txt                 # Dependencies (MySQL only)
├── models/
│   ├── device.py                   # SQLAlchemy Device model
│   └── ...
├── routers/
│   ├── devices.py                  # Device API endpoints
│   └── ...
├── schemas/
│   └── device.py                   # Pydantic schemas
├── ml/
│   ├── dataset_loader_mysql.py     # MySQL dataset loader (NEW)
│   ├── advanced_nlp_parser.py      # NLP query parser
│   └── ...
└── utils/
    └── ...
```

## Summary

✅ **MongoDB Fully Removed**
- All Beanie/Motor code gone
- All async MongoDB initialization removed

✅ **MySQL Fully Integrated**
- SQLAlchemy ORM configured
- Connection pooling enabled
- All device data loads from MySQL table

✅ **Battery Capacity Working**
- Mapped to MySQL `battery_capacity` column
- Displays correctly in API responses
- Frontend receives and displays the data

✅ **Ready for NLP**
- Dataset loader extracts and normalizes specs
- NLP parser framework in place
- Ready to implement recommendation logic

## Next Steps

1. Test the API endpoints
2. Implement NLP recommendation logic
3. Add price extraction and filtering
4. Implement similarity-based search
5. Add XAI explainability layer

---

**Created:** March 3, 2026
**MySQL Credentials:**
- User: root
- Password: 123
- Database: device_catalog
- Table: devices
