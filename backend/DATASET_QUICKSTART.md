# Quick Start: Using Phone Datasets for Recommendations

## 5-Minute Setup

### Step 1: Import Dataset to MongoDB

```bash
cd backend

# Basic import (imports all datasets)
python -m ml.import_enhanced_dataset

# For testing with limited devices
python -m ml.import_enhanced_dataset --limit 1000

# Test feature extraction first
python -m ml.import_enhanced_dataset --test
```

### Step 2: Run Integration Tests

```bash
python -m ml.test_dataset_integration
```

This will test:
- ✅ Dataset loading
- ✅ Feature extraction
- ✅ Device filtering
- ✅ Recommendation engine
- ✅ Device comparison

## Usage Examples

### Example 1: Get Gaming Phone Recommendations

```python
from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender

# Load dataset
loader = PhoneDatasetLoader()
devices = loader.load_csv_files()

# Train recommender
recommender = DeviceRecommender()
recommender.fit(devices)

# Get gaming phone recommendations
recommendations = recommender.recommend_by_preferences({
    'use_case': 'gaming',
    'min_ram_gb': 8,
    'budget': 1000,
    'require_5g': True
}, top_n=10)

# Print results
for device_id, score in recommendations[:3]:
    device = next(d for d in devices if d['id'] == device_id)
    print(f"{device['brand']} {device['model_name']}: {score:.3f}")
```

### Example 2: Filter Phones by Specifications

```python
from utils.device_filter import DeviceFilter, SpecRequirements

# Define requirements
reqs = SpecRequirements(
    min_ram_gb=8,
    min_camera_mp=48,
    max_price=500,
    require_5g=True
)

# Filter devices
filtered = DeviceFilter.filter_by_specs(devices, reqs)
print(f"Found {len(filtered)} matching phones")

# Print first 5
for device in filtered[:5]:
    specs = device['specs']
    print(f"- {device['brand']} {device['model_name']}")
    print(f"  RAM: {specs['ram_gb']}GB | Camera: {specs['main_camera_mp']}MP")
    print(f"  Price: ${specs['price']} | 5G: {specs['has_5g']}")
```

### Example 3: Score Phones for Specific Use Case

```python
from utils.device_filter import DeviceFilter, UseCase

# Get top phones for each use case
for use_case in [UseCase.GAMING, UseCase.PHOTOGRAPHY, UseCase.BATTERY]:
    scores = []
    for device in devices:
        score = DeviceFilter.score_device_for_use_case(device, use_case)
        scores.append((device, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Top 3 for {use_case.value}:")
    for i, (device, score) in enumerate(scores[:3]):
        print(f"  {i+1}. {device['brand']} {device['model_name']} - {score:.0f}/100")
```

### Example 4: Compare Multiple Devices

```python
from utils.device_filter import ComparisonHelper

# Get top 3 flagship phones
flagships = sorted(devices, 
                  key=lambda d: d['specs'].get('price', 0), 
                  reverse=True)[:3]

# Create comparison
comparison = ComparisonHelper.compare_devices(flagships)

print("Device Comparison:")
for i, device_info in enumerate(comparison['devices']):
    print(f"  Device {i+1}: {device_info['brand']} {device_info['model']}")

print("\nSpecification Comparison:")
for spec_name, spec_data in comparison['specs'].items():
    print(f"\n{spec_name}:")
    for i, value in enumerate(spec_data['values']):
        print(f"  Device {i+1}: {value}")
```

## API Usage

### Natural Language Query

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best gaming phone under 1000 with 5G",
    "top_n": 10
  }'
```

### Structured Query with Features

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 800,
    "min_ram_gb": 8,
    "min_camera_mp": 48,
    "require_5g": true,
    "use_case": "gaming",
    "top_n": 10,
    "explain": true
  }'
```

## Available Datasets

### Brands Available
- Apple, Samsung, Google, OnePlus, Xiaomi
- Huawei, Oppo, Vivo, Realme, Motorola
- Sony, LG, HTC, Nokia, Asus
- Lenovo, Coolpad, Tecno, Infinix, iQOO
- And ~100+ more brands

### Device Types
- **Mobile**: Smartphones (largest collection)
- **Tablet**: iPad, Galaxy Tab, etc.
- **Smartwatch**: Wear OS, watchOS devices

### Price Ranges
- Budget: $50-300
- Mid-range: $300-700
- Premium: $700-1500
- Flagship: $1500+

## Statistics Command

Get dataset overview:

```python
loader = PhoneDatasetLoader()
devices = loader.load_csv_files()

stats = loader.get_statistics()
print(f"Total Devices: {stats['total_devices']}")
print(f"Brands: {stats['brands']}")
print(f"Price Range: ${stats['price']['min']} - ${stats['price']['max']}")
print(f"Average RAM: {stats['ram']['avg']:.1f}GB")
print(f"Average Battery: {stats['battery']['avg']:.0f}mAh")
```

## Filtering Examples

### Budget Gaming Phone
```python
reqs = SpecRequirements(
    device_type='mobile',
    min_ram_gb=6,
    min_refresh_rate=90,
    max_price=400
)
```

### Photography Flagship
```python
reqs = SpecRequirements(
    min_camera_mp=64,
    min_ram_gb=12,
    min_storage_gb=256
)
```

### Battery Beast
```python
reqs = SpecRequirements(
    min_battery_mah=5500,
    require_fast_charging=True
)
```

### 5G Productivity
```python
reqs = SpecRequirements(
    require_5g=True,
    min_ram_gb=8,
    min_storage_gb=128
)
```

## Common Issues

### Issue: "No devices found"
- Ensure MongoDB is running
- Check dataset was imported: `python -m ml.import_enhanced_dataset`

### Issue: Slow recommendations
- Use feature filtering first to reduce dataset
- Import limited dataset for testing: `--limit 1000`

### Issue: Missing specifications
- Some older devices may have incomplete data
- Use `spec.get('field', 0)` with defaults

## Next Steps

1. **Customize Scoring**: Edit `_adjust_scores_by_use_case` in `recommender.py`
2. **Add Filters**: Extend `SpecRequirements` class with new fields
3. **Integrate Reviews**: Add review aggregation to scoring
4. **Train Collaboratively**: Use user feedback to improve recommendations
5. **Cache Results**: Add Redis caching for repeated queries

## Documentation

- **Full Guide**: [DATASET_INTEGRATION.md](DATASET_INTEGRATION.md)
- **Implementation Details**: [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md)
- **Features**: [COMPLEX_QUERY_SUPPORT.md](../COMPLEX_QUERY_SUPPORT.md)

---

**Ready to recommend phones!** 🚀📱
