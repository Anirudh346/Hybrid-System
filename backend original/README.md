# SmartAI Device Filter - Backend

FastAPI backend with MongoDB, JWT authentication, AI recommendations, and price tracking.

## Features

- 🔐 JWT Authentication
- 📱 Device CRUD with advanced filtering
- 🤖 AI-powered recommendations using NLP/ML
- ⭐ User favorites and saved searches
- 💰 Price tracking and history
- 🔄 Device comparison
- 📊 MongoDB with Beanie ODM

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The BERT model (dslim/bert-base-NER) will be automatically downloaded on first use.

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your MongoDB URL, secret key, etc.

### 3. Import CSV Data

```bash
python import_csv.py ../major/GSMArenaDataset
```

Or with append mode:

```bash
python import_csv.py ../major/GSMArenaDataset --append
```

### 5. Run Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/refresh` - Refresh token

### Devices
- `GET /api/devices` - List devices with filters
- `GET /api/devices/{id}` - Get device details
- `GET /api/devices/brands` - Get all brands
- `POST /api/devices/upload` - Upload CSV dataset

### User Features
- `GET /api/user/favorites` - Get favorites
- `POST /api/user/favorites` - Add favorite
- `DELETE /api/user/favorites/{device_id}` - Remove favorite
- `GET /api/user/searches` - Get saved searches
- `POST /api/user/searches` - Save search
- `PUT /api/user/searches/{id}` - Update search
- `DELETE /api/user/searches/{id}` - Delete search

### AI Recommendations
- `POST /api/recommend` - Get AI recommendations

Example request:
```json
{
  "query": "best gaming phone under $800",
  "top_n": 10
}
```

### Price Tracking
- `POST /api/price-track/subscribe` - Subscribe to price alerts
- `GET /api/price-track/{device_id}/history` - Get price history
- `DELETE /api/price-track/unsubscribe/{device_id}` - Unsubscribe

### Comparisons
- `GET /api/compare` - Get user's comparisons
- `POST /api/compare` - Create comparison
- `GET /api/compare/{id}` - Get comparison details
- `DELETE /api/compare/{id}` - Delete comparison

## Project Structure

```
backend/
├── main.py                 # FastAPI app
├── config.py               # Settings
├── requirements.txt        # Dependencies
├── import_csv.py          # CSV import script
├── models/                # Beanie document models
│   ├── device.py
│   ├── user.py
│   ├── favorite.py
│   ├── saved_search.py
│   ├── price_history.py
│   └── comparison.py
├── routers/               # API endpoints
│   ├── auth.py
│   ├── devices.py
│   ├── users.py
│   ├── recommendations.py
│   ├── price_tracking.py
│   └── comparisons.py
├── schemas/               # Pydantic schemas
│   ├── auth.py
│   ├── device.py
│   ├── user.py
│   ├── recommendation.py
│   ├── price_tracking.py
│   └── comparison.py
├── ml/                    # Machine learning
│   ├── recommender.py    # Content-based recommender
│   └── nlp_parser.py     # NLP query parser
└── utils/                 # Utilities
    └── auth.py           # JWT utilities
```

## ML/NLP Features

### Recommendation Engine

Uses TF-IDF vectorization and cosine similarity for content-based recommendations:

- Analyzes device specifications (Chipset, Camera, Display, etc.)
- Supports natural language queries
- Filters by budget, device type, brand
- Boosts relevant matches based on use case (gaming, photography, etc.)

### NLP Query Parser

Extracts structured data from natural language:

```python
"best gaming phone under $800" →
{
  "budget": 800,
  "device_type": ["mobile"],
  "use_case": "gaming",
  "query": "best gaming phone under $800"
}
```

## Development

### Generate Secret Key

```bash
openssl rand -hex 32
```

### MongoDB Indexes

Indexes are automatically created on startup via Beanie.

### Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests (coming soon)
pytest
```

## Deployment

### Environment Variables

Required for production:
- `MONGODB_URL`
- `SECRET_KEY`
- `SENDGRID_API_KEY` (for price alerts)
- `ALLOWED_ORIGINS`

### Docker (Coming Soon)

```bash
docker build -t smartai-backend .
docker run -p 8000:8000 smartai-backend
```

## License

MIT
