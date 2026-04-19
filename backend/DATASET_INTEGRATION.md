# Phone Dataset Integration & Recommendation System

## Overview

This document describes the integration of the GSMArena phone datasets into the SmartAI Smart Device Filter project. The system uses advanced feature extraction and machine learning to provide intelligent phone recommendations based on user preferences and requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Phone Datasets (GSMArenaDataset)          │
│                   ~130 CSV files with 50,000+ phones         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                 ┌──────────────────────────┐
                 │   Dataset Loader         │
                 │  (dataset_loader.py)     │
                 │  - CSV parsing           │
                 │  - Feature extraction    │
                 │  - Data normalization    │
                 └────────────┬─────────────┘
                              │
                 ┌────────────┴──────────────┐
                 │                           │
                 ▼                           ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │   MongoDB Database   │      │  Feature Filter      │
    │  (Beanie ODM)        │      │ (device_filter.py)   │
    │  - Device models     │      │ - Spec requirements  │
    │  - Specs storage     │      │ - Use-case scoring   │
    └──────────────────────┘      │ - Comparison         │
                 │                 └──────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────┐
    │    Recommendation Engine             │
    │    (recommender.py)                  │
    │  - TF-IDF + Cosine Similarity       │
    │  - Feature-based scoring            │
    │  - Use-case specific ranking        │
    └──────────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────┐
    │        API Endpoints                 │
    │    (routers/recommendations.py)      │
    │  - Natural language queries          │
    │  - Structured preferences            │
    │  - XAI explanations                  │
    └──────────────────────────────────────┘
```

## Dataset Structure

### Location
```
SmartAI-SmartDeviceFilter-main/DatasetPhones/GSMArenaDataset/
├── Apple.csv
├── Samsung.csv
├── Xiaomi.csv
├── Huawei.csv
├── OnePlus.csv
├── Google.csv
└── ... (~130 brand CSV files)
```

### CSV Format
Each CSV contains columns for:
- **Identification**: Brand, Model Name, Model Image
- **Network**: Technology, 2G/3G/4G/5G bands
- **Design**: Dimensions, Weight, Build material
- **Display**: Type, Size, Resolution, Refresh Rate
- **Performance**: Chipset, CPU, GPU, RAM
- **Storage**: Internal storage, Expandable storage
- **Camera**: Main camera, Selfie camera, Video recording
- **Battery**: Capacity, Charging technology
- **Connectivity**: Bluetooth, NFC, USB, WiFi
- **Sensors**: Accelerometer, Gyro, Compass, etc.
- **Pricing**: Variants with storage/RAM combinations

## Feature Extraction

### Extracted Specifications (Standardized)

```python
specs = {
    # Numeric specifications
    'ram_gb': int,                    # RAM in GB
    'storage_gb': int,                # Storage in GB
    'battery_mah': int,               # Battery in mAh
    'main_camera_mp': float,          # Main camera in MP
    'selfie_camera_mp': float,        # Selfie camera in MP
    'display_size_inches': float,     # Display size
    'refresh_rate_hz': int,           # Refresh rate (default 60)
    'price': float,                   # Price in USD
    'weight_g': float,                # Weight in grams
    
    # Feature flags (boolean)
    'has_5g': bool,                   # 5G support
    'has_nfc': bool,                  # NFC support
    'has_wireless_charging': bool,    # Wireless charging
    'has_fast_charging': bool,        # Fast charging (25W+)
    'has_dual_sim': bool,             # Dual SIM support
    'has_expandable_storage': bool,   # MicroSD support
    'has_jack_35mm': bool,            # 3.5mm headphone jack
    
    # Text specifications
    'os': str,                        # Operating System
    'chipset': str,                   # Chipset/Processor
    'display_type': str,              # Display technology (OLED, LCD, etc)
    'design_material': str,           # Build material (glass, metal, plastic)
}
```

## Key Components

### 1. Dataset Loader (`ml/dataset_loader.py`)

Loads and processes phone datasets from CSV files.

**Main Classes:**
- `SpecificationExtractor`: Extracts and normalizes specs from raw data
- `PhoneDatasetLoader`: Loads CSV files and manages dataset

**Key Methods:**
```python
# Load all datasets
loader = PhoneDatasetLoader()
devices = loader.load_csv_files()

# Filter by features
budget_phones = loader.get_devices_by_price_range(200, 500)
gaming_phones = loader.get_gaming_devices()
camera_phones = loader.get_camera_phones()

# Get statistics
stats = loader.get_statistics()
```

### 2. Feature Filter (`utils/device_filter.py`)

Advanced device filtering based on specifications.

**Main Classes:**
- `DeviceFilter`: Core filtering logic
- `SpecRequirements`: Dataclass for filter criteria
- `UseCase`: Enum for predefined use cases
- `ComparisonHelper`: Device comparison utilities

**Use Cases:**
- `GAMING`: High RAM (8GB+), High refresh rate (120Hz+)
- `PHOTOGRAPHY`: High camera (48MP+), Good RAM for processing
- `BATTERY`: Large battery (5500mAh+), Fast charging
- `DISPLAY`: High refresh rate (90Hz+), Quality panel
- `BUDGET`: Low price (<$500), Good value
- `VIDEO`: High RAM, Large storage, High refresh rate
- `PRODUCTIVITY`: High RAM, 5G support, Good display

**Example Usage:**
```python
# Define requirements
reqs = SpecRequirements(
    min_ram_gb=8,
    min_camera_mp=48,
    max_price=800,
    require_5g=True
)

# Filter devices
filtered = DeviceFilter.filter_by_specs(devices, reqs)

# Score devices for use case
score = DeviceFilter.score_device_for_use_case(device, UseCase.GAMING)

# Compare devices
comparison = ComparisonHelper.compare_devices(devices)
best = ComparisonHelper.get_best_device_for_spec(devices, 'battery_mah')
```

### 3. Recommendation Engine (`ml/recommender.py`)

ML-based recommendation system using TF-IDF and feature scoring.

**Key Features:**
- Content-based filtering using TF-IDF
- Cosine similarity scoring
- Feature-based scoring adjustments
- Use-case specific ranking
- Budget-aware recommendations

**Methods:**
```python
recommender = DeviceRecommender()
recommender.fit(devices)

# Natural language + structured preferences
recommendations = recommender.recommend_by_preferences({
    'query': 'best gaming phone',
    'budget': 1000,
    'use_case': 'gaming',
    'min_ram_gb': 8,
    'require_5g': True
}, top_n=10)

# Direct feature-based recommendation
recommendations = recommender.recommend_by_features(
    min_ram_gb=8,
    min_camera_mp=48,
    max_price=500,
    use_case='photography',
    top_n=10
)
```

### 4. Enhanced Data Import (`ml/import_enhanced_dataset.py`)

Script to import datasets into MongoDB with feature extraction.

**Usage:**
```bash
# Import all datasets
python import_enhanced_dataset.py

# Import with custom path
python import_enhanced_dataset.py --dataset-path /path/to/GSMArenaDataset

# Test feature extraction without importing
python import_enhanced_dataset.py --test

# Import limited devices for testing
python import_enhanced_dataset.py --limit 100

# Append to existing data instead of replacing
python import_enhanced_dataset.py --append
```

## API Integration

### Recommendation Endpoint

**POST** `/recommendations`

**Request:**
```json
{
  "query": "best gaming phone under 1000",
  "budget": 1000,
  "device_type": ["mobile"],
  "use_case": "gaming",
  "min_ram_gb": 8,
  "min_refresh_rate": 120,
  "require_5g": true,
  "prefer_fast_charging": true,
  "top_n": 10,
  "explain": true
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "device_id": "string",
      "brand": "string",
      "model_name": "string",
      "score": 0.95,
      "specs": {
        "ram_gb": 12,
        "storage_gb": 256,
        "price": 899,
        "battery_mah": 5000,
        "refresh_rate_hz": 144
      },
      "explanation": {
        "overall_score": 95,
        "match_summary": "string",
        "feature_contributions": [...]
      }
    }
  ],
  "parsed_preferences": {},
  "total_candidates": 45
}
```

## Testing

### Run Integration Tests

```bash
cd backend/ml
python test_dataset_integration.py
```

**Tests Include:**
1. Dataset loading and statistics
2. Feature-based filtering
3. Use-case scoring
4. Recommendation engine
5. Device comparison

### Example Test Output
```
TEST 1: Dataset Loading
✅ Loaded 500 devices
📊 Dataset Statistics:
   Total Devices: 500
   Device Types: {'mobile': 480, 'tablet': 15, 'smartwatch': 5}
   Unique Brands: 45
   Price Range: $50 - $2500 (Avg: $450)
   RAM Range: 1 - 16GB (Avg: 6.2GB)
   Battery Range: 2000 - 7000mAh (Avg: 4200mAh)

TEST 2: Feature-Based Filtering
🎮 Gaming Phones (8GB+ RAM, 120Hz+):
   Found 45 gaming phones
   - Apple iPhone 15 Pro Max: 8GB, 120Hz, $1200
   - Samsung Galaxy S24 Ultra: 12GB, 120Hz, $1399
```

## Configuration

### MongoDB Setup

Ensure MongoDB is running with the correct configuration in `config.py`:

```python
settings.mongodb_url = "mongodb://localhost:27017"
settings.database_name = "smartai_devices"
```

### Dataset Path

Default dataset path (auto-detected):
```
SmartAI-SmartDeviceFilter-main/DatasetPhones/GSMArenaDataset/
```

Can be overridden:
```python
loader = PhoneDatasetLoader("/custom/path/to/GSMArenaDataset")
```

## Performance Considerations

### Dataset Size
- **Total devices**: ~50,000+
- **Total CSV files**: ~130 brand files
- **Import time**: ~5-10 minutes for full dataset
- **Database size**: ~500MB-1GB

### Recommendation Performance
- **Training time**: ~1-2 seconds for 50,000 devices
- **Query response time**: <500ms for recommendations
- **Scoring**: O(n) complexity where n = number of devices

### Optimization Tips
1. Use feature filtering first to reduce dataset
2. Cache recommender model for repeated queries
3. Index MongoDB by brand, device_type, price range
4. Use limit parameter during development

## Future Enhancements

1. **Collaborative Filtering**: Learn from user preferences and behavior
2. **Real-time Price Updates**: Integrate with price tracking APIs
3. **Review Aggregation**: Include user reviews in scoring
4. **Specification Predictions**: ML model for predicting future specs
5. **Personalized Profiles**: User preference learning
6. **Image Recognition**: Analyze device images for additional features
7. **Multilingual NLP**: Support multiple languages for queries
8. **A/B Testing**: Experiment with different recommendation strategies

## Troubleshooting

### Issue: No devices imported
**Solution**: Check MongoDB connection and dataset path

```python
# Verify dataset path
from pathlib import Path
dataset_path = Path("DatasetPhones/GSMArenaDataset")
print(f"Path exists: {dataset_path.exists()}")
print(f"CSV files: {list(dataset_path.glob('*.csv'))}")
```

### Issue: Low recommendation scores
**Solution**: Verify device specs are extracted correctly

```python
# Check extracted specs
device = devices[0]
print(device['specs'])
```

### Issue: Slow queries
**Solution**: Use feature filtering before recommendation

```python
# Instead of full dataset
filtered = DeviceFilter.filter_by_specs(devices, reqs)
recommender.fit(filtered)
```

## References

- **GSMArena Dataset**: https://gsmarena.com/
- **TF-IDF**: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf
- **Cosine Similarity**: https://scikit-learn.org/stable/modules/metrics.pairwise.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **MongoDB Beanie**: https://roman-right.github.io/beanie/

## Support

For questions or issues, please refer to:
1. [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md)
2. [QUICK_START.md](../QUICK_START.md)
3. Project README files

---
**Last Updated**: January 2025
**Version**: 1.0
