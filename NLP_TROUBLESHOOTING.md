# NLP Search Troubleshooting Guide

If the NLP search is not working, follow these diagnostic steps:

## Quick Checks

### 1. **Is the backend running?**
```bash
# Check if backend is accessible
curl http://localhost:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "environment": "development",
  "database": "smartai_devices"
}
```

If this fails: Start the backend server
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### 2. **Is the parse endpoint registered?**
```bash
curl http://localhost:8000/api/docs
```
- Open `http://localhost:8000/api/docs` in browser
- Search for `/api/recommend/parse` in the Swagger UI
- If not found, the endpoint is not registered (check `backend/main.py`)

### 3. **Can you call the parse endpoint directly?**
```bash
curl -X POST http://localhost:8000/api/recommend/parse \
  -H "Content-Type: application/json" \
  -d '{"query": "gaming phone under $500"}'
```

Expected response:
```json
{
  "parsed_preferences": {
    "query": "gaming phone under $500",
    "device_type": ["mobile"],
    "use_case": "gaming",
    "budget": 500,
    "brand_preference": [],
    "confidence": 0.85,
    ...
  },
  "query": "gaming phone under $500"
}
```

Possible errors:
- **400 Bad Request**: Query is empty or missing
- **500 Internal Server Error**: NLP model failed to load or parse error (check backend logs)
- **405 Method Not Allowed**: Endpoint not found (check router registration)

### 4. **Check browser console for errors**
Open DevTools (F12) → Console tab:
- Look for red error messages when you search
- Common issues:
  - `❌ NLP parse error: ...` — Backend returned an error
  - `Failed to fetch` — Backend not reachable (CORS or network)
  - `undefined is not a function` — API method not properly defined

## Common Issues & Solutions

### Issue: "Failed to fetch" / CORS error
**Symptoms**: Request blocked by CORS policy

**Causes**:
1. Backend not running
2. Frontend is on different origin than backend (localhost:5173 vs localhost:8000)
3. CORS not configured in backend

**Solutions**:
```python
# In backend/main.py, check CORS configuration:
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,  # Should include frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In backend/config.py, ensure frontend URL is in origins:
origins_list = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",
    "http://localhost:8080",
]
```

### Issue: "Error parsing query" or 500 error on parse endpoint
**Symptoms**: Backend returns 500 error when parsing

**Causes**:
1. NLP models failed to load (transformers/BERT models)
2. advanced_nlp_parser or nlp_parser module has an error
3. Missing dependencies

**Solutions**:
```bash
# Check backend logs for detailed error
# Look for error traces in terminal running uvicorn

# Verify NLP module imports work:
cd backend
python -c "from ml.advanced_nlp_parser import advanced_parser; print('✅ advanced_parser loaded')"
python -c "from ml.nlp_parser import nlp_parser; print('✅ nlp_parser loaded')"

# Install missing transformers if needed:
pip install transformers torch
```

### Issue: Parse works, but products page shows no recommendations
**Symptoms**: Parsed preferences display but no devices shown

**Causes**:
1. No devices in database (dataset not uploaded)
2. Parsed preferences don't match any devices
3. Recommendation endpoint failing (different error from parse)
4. Devices have incompatible data format

**Solutions**:
```javascript
// Check browser console for error messages:
// Look for: "❌ Failed to fetch recommendations: ..."

// Verify dataset is loaded:
// 1. Upload a CSV via the Upload Dataset button
// 2. Check Products page without NLP search shows devices
// 3. Then try NLP search again
```

### Issue: Frontend still falling back to keyword search
**Symptoms**: NLP search not triggering, always uses keyword search fallback

**Causes**:
1. Parse endpoint is returning error
2. API method not defined correctly
3. Network timeout or slow backend

**Solutions**:
```javascript
// Add debugging in src/components/Navbar.jsx
const handleSearch = async (e) => {
  e.preventDefault();
  if (!searchQuery.trim()) return;

  try {
    console.log('🔍 Calling parse endpoint...');
    const parseResponse = await api.recommendations.parse(searchQuery);
    console.log('✅ Parse response:', parseResponse.data);
    // ... rest of code
  } catch (err) {
    console.error('❌ Parse failed:', {
      status: err.response?.status,
      data: err.response?.data,
      message: err.message,
    });
  }
};
```

### Issue: "Cannot read property 'parse' of undefined"
**Symptoms**: JavaScript error saying `parse` is undefined

**Causes**:
1. API client not imported
2. API client doesn't have recommendations.parse method
3. Typo in method name

**Solutions**:
```javascript
// In Navbar.jsx, ensure import exists:
import api from '../services/apiClient';

// In apiClient.js, verify parse method exists:
recommendations: {
  get: (data) => apiClient.post('/api/recommend', data),
  parse: (query) => apiClient.post('/api/recommend/parse', { query }),
},
```

### Issue: Query parses but recommendations are slow / timeout
**Symptoms**: Recommendation fetch takes >5 seconds or times out

**Causes**:
1. Recommendation endpoint is slow (ML ranking heavy)
2. Large dataset (many devices to rank)
3. Network latency
4. Server CPU/memory constraints

**Solutions**:
```javascript
// In Products.jsx, increase timeout (axios default is 0 = no timeout):
const response = await api.recommendations.get({
  ...preferences,
  top_n: 20,
  explain: false,
}, {
  timeout: 10000,  // 10 second timeout
});

// Or reduce top_n to speed up ranking:
top_n: 10,  // Instead of 20
```

## Debugging Workflow

1. **Open browser DevTools** (F12)
2. **Search for something** in the Navbar search bar
3. **Go to Network tab** and look for:
   - `POST /api/recommend/parse` request
   - Check response status (200 = success, 500 = error)
   - Check response body for error details
4. **Go to Console tab** and look for logs:
   - `🔍 Parsing query: ...` — request sent
   - `✅ Parse successful: ...` — response received
   - `❌ NLP parse error: ...` — error occurred
5. **Check backend terminal** for server-side error logs

## Testing Checklist

- [ ] Backend `/health` endpoint responds
- [ ] Backend `/api/docs` shows `/api/recommend/parse` endpoint
- [ ] Direct curl to `/api/recommend/parse` works and returns parsed preferences
- [ ] Frontend has no JavaScript console errors
- [ ] Dataset is uploaded and visible in Products page
- [ ] Typing in search bar triggers `handleSearch`
- [ ] Console shows `🔍 Parsing query:` log message
- [ ] Console shows either `✅ Parse successful:` or `❌ NLP parse error:`
- [ ] If successful, navigation happens and Products page shows parsed preferences chips
- [ ] If failed, fallback to keyword search occurs

## Contact Backend Logs

To see detailed error messages from the backend:

```bash
# Terminal running uvicorn shows all logs
# Look for lines like:
# ERROR in parse_query: <detailed error message>

# If using PM2 or systemd, check:
pm2 logs
journalctl -u backend-service
```

## Reset Everything

If all else fails, restart both services:

```bash
# Terminal 1: Frontend
cd frontend-directory
npm run dev

# Terminal 2: Backend
cd backend
python -m uvicorn main:app --reload --port 8000

# Clear browser cache:
# Open DevTools → Application → Clear Site Data
```

