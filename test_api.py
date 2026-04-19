import requests
import json

try:
    r = requests.get('http://localhost:8000/api/devices?limit=2')
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
    if r.status_code == 200:
        data = r.json()
        print(f"First device: {json.dumps(data['devices'][0], indent=2)[:500]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
