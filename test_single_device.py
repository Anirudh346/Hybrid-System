import requests
import json

# Test individual device endpoint
response = requests.get('http://localhost:8000/api/devices/12')
if response.status_code == 200:
    device = response.json()
    print(f"Device ID 12:")
    print(f"  Name: {device.get('model_name')}")
    print(f"  Brand: {device.get('brand')}")
    print(f"  Battery Capacity: {device.get('battery_capacity')} ✅")
else:
    print(f"Error: {response.status_code}")

# Test another device
response = requests.get('http://localhost:8000/api/devices/13')
if response.status_code == 200:
    device = response.json()
    print(f"\nDevice ID 13:")
    print(f"  Name: {device.get('model_name')}")
    print(f"  Brand: {device.get('brand')}")
    print(f"  Battery Capacity: {device.get('battery_capacity')} ✅")
