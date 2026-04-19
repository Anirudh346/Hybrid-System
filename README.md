# 📱 SmartAI Device Filter

A full-stack intelligent device recommendation and comparison platform powered by AI/ML. Browse, compare, and get personalized recommendations for 10,000+ mobile devices, tablets, and smartwatches using natural language queries.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![React](https://img.shields.io/badge/React-18.2-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688)

---

## 🌟 Key Features

### 🤖 AI-Powered Natural Language Search
- Search using natural language queries like: *"gaming phone under $500 with good camera"*
- Advanced NLP parsing with BERT models for intent understanding
- Extracts device type, brand preferences, budget constraints, and use cases
- Returns top 10 most relevant recommendations sorted by release date

### 🎯 Intelligent Filtering & Search
- **126 Brand Filters**: Filter by any smartphone manufacturer
- **10 Chipset Filters**: Qualcomm, MediaTek, Apple, Samsung, HiSilicon, Google, UNISOC, NVIDIA, Intel, Leadcore
- **Price Range Filtering**: All prices converted to USD
- **Release Date Sorting**: Newest devices appear first
- **Real-time Search**: Instant results as you type

### 💰 Multi-Currency Price Support
Automatic conversion from 10+ currencies to USD:
- EUR (Euro), INR (Indian Rupee), GBP (British Pound)
- CNY (Chinese Yuan), JPY (Japanese Yen)
- AUD, CAD, RUB, BRL, and more
- Handles missing prices gracefully (no more $0.00!)
- Preserves informational text like "Coming soon" or "Discontinued"

### 📊 Device Comparison
- Side-by-side comparison of multiple devices
- Comprehensive spec comparison across 40+ attributes
- Display, camera, performance, battery metrics

### 🔒 User Authentication & Personalization
- JWT-based authentication with refresh tokens
- Save favorite devices with personal notes
- Saved searches and filter combinations
- Price tracking and alerts

### 📈 Advanced Analytics
- Price history tracking
- Device popularity trends
- User preference learning

---

## 🏗️ Architecture

### Tech Stack

#### **Frontend**
- **React 18** - Modern UI library with hooks
- **React Router v6** - Client-side routing
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client with interceptors
- **Lucide React** - Beautiful icon set
- **Recharts** - Data visualization

#### **Backend**
- **FastAPI** - High-performance async Python framework
- **MySQL + PyMySQL** - Production-grade relational database
- **SQLAlchemy** - ORM for database operations
- **Pydantic v2** - Data validation and serialization
- **JWT (python-jose)** - Secure authentication
- **Uvicorn** - ASGI server

#### **AI/ML Stack**
- **Transformers** - Hugging Face transformers library
- **BERT Models**:
  - `sentence-transformers/all-MiniLM-L6-v2` - Sentence embeddings
  - `dslim/bert-base-NER` - Named entity recognition
  - `deepset/roberta-base-squad2` - Question answering
- **scikit-learn** - Machine learning utilities
- **pandas & numpy** - Data processing
- **spaCy** - Advanced NLP processing

### Project Structure

```
SmartAI-SmartDeviceFilter-main/
│
├── backend/                          # FastAPI Backend
│   ├── main.py                       # Application entry point
│   ├── config.py                     # Environment configuration
│   ├── database.py                   # MySQL connection management
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── models/                       # SQLAlchemy ORM Models
│   │   ├── device.py                 # Device model (40+ attributes)
│   │   ├── user.py                   # User authentication
│   │   ├── favorite.py               # User favorites
│   │   ├── saved_search.py           # Saved filter combinations
│   │   ├── price_history.py          # Price tracking
│   │   └── comparison.py             # Device comparisons
│   │
│   ├── routers/                      # API Endpoints
│   │   ├── auth.py                   # Authentication (login, register, token refresh)
│   │   ├── devices.py                # Device CRUD, filters, NLP search
│   │   ├── recommendations.py        # AI recommendations
│   │   ├── comparisons.py            # Device comparison
│   │   ├── price_tracking.py         # Price alerts
│   │   └── users.py                  # User profile management
│   │
│   ├── schemas/                      # Pydantic Request/Response Models
│   │   ├── device.py                 # Device schemas (DeviceResponse, BrandResponse, ChipsetResponse)
│   │   ├── auth.py                   # Auth schemas (Token, UserCreate)
│   │   ├── recommendation.py         # Recommendation schemas
│   │   └── ...
│   │
│   ├── ml/                           # Machine Learning Modules
│   │   ├── advanced_nlp_parser.py    # BERT-based NLP query parser
│   │   ├── nlp_parser.py             # Basic NLP with spaCy
│   │   ├── recommender.py            # Content-based recommendation engine
│   │   ├── xai_explainer.py          # Explainable AI for recommendations
│   │   ├── dataset_loader.py         # Dataset management
│   │   └── simple_recommender.py     # Lightweight recommender
│   │
│   ├── db/                           # Database Utilities
│   │   └── mysql.py                  # MySQL helper functions
│   │
│   └── utils/                        # Utility Functions
│       ├── auth.py                   # JWT token handling
│       └── ...
│
├── src/                              # React Frontend
│   ├── main.jsx                      # React entry point
│   ├── App.jsx                       # Main app component with routing
│   │
│   ├── pages/                        # Page Components
│   │   ├── Home.jsx                  # Landing page
│   │   ├── Products.jsx              # Product listing with filters
│   │   ├── ProductDetails.jsx        # Device detail view
│   │   ├── Login.jsx                 # Authentication
│   │   ├── Signup.jsx                # User registration
│   │   ├── AIRecommendation.jsx      # AI recommendation interface
│   │   ├── Comparison.jsx            # Device comparison
│   │   └── ...
│   │
│   ├── components/                   # Reusable Components
│   │   ├── Navbar.jsx                # Navigation with NLP search
│   │   ├── Sidebar.jsx               # Filter sidebar (brands, chipsets)
│   │   ├── DeviceCard.jsx            # Device card display
│   │   ├── PrivateRoute.jsx          # Auth-protected routes
│   │   └── ...
│   │
│   ├── contexts/                     # React Context
│   │   └── AuthContext.jsx           # Global auth state
│   │
│   └── services/                     # API Services
│       └── apiClient.js              # Axios configuration with interceptors
│
├── DatasetPhones/                    # Dataset Storage
│   └── GSMArenaDataset/              # 10,000+ device CSV files
│
├── package.json                      # Frontend dependencies
├── vite.config.js                    # Vite configuration
├── tailwind.config.js                # Tailwind CSS configuration
└── README.md                         # This file

```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **Node.js 18+** and npm
- **MySQL 8.0+**
- **Git**

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd SmartAI-SmartDeviceFilter-main
```

### 2️⃣ Backend Setup

#### Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- FastAPI, Uvicorn, SQLAlchemy, PyMySQL
- Transformers, sentence-transformers, torch
- scikit-learn, pandas, numpy
- python-jose, passlib, python-dotenv

**Note**: First run will download BERT models (~500MB) automatically.

#### Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# Database Configuration
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/device_catalog
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=root
DATABASE_PASSWORD=YOUR_PASSWORD
DATABASE_NAME=device_catalog

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS (Frontend URLs)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:5174

# Redis (Optional - for caching)
REDIS_URL=redis://localhost:6379/0

# Email (Optional - for price alerts)
SENDGRID_API_KEY=your-sendgrid-api-key
FROM_EMAIL=noreply@smartai.com
```

#### Setup MySQL Database

```sql
-- Create database
CREATE DATABASE device_catalog CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create user (optional)
CREATE USER 'smartai_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON device_catalog.* TO 'smartai_user'@'localhost';
FLUSH PRIVILEGES;
```

#### Import Device Dataset

```bash
# From backend directory
python import_devices_to_mysql.py
```

This will:
- Create all necessary tables
- Import 10,000+ devices from CSV files
- Set up indexes for performance
- Verify data integrity

#### Run Backend Server

```bash
# Development mode with auto-reload
cd backend
python main.py

# OR using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3️⃣ Frontend Setup

#### Install Node Dependencies

```bash
# From project root
npm install
```

#### Configure Environment (Optional)

Create `.env` in the project root:

```bash
VITE_API_URL=http://localhost:8000
```

#### Run Frontend Development Server

```bash
npm run dev
```

Frontend will be available at:
- **App**: http://localhost:5173

#### Build for Production

```bash
npm run build
npm run preview
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication Endpoints

#### Register New User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

username=john@example.com&password=SecurePass123!
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### Refresh Token
```http
POST /api/auth/refresh
Authorization: Bearer <refresh_token>
```

### Device Endpoints

#### Get All Devices (with filters)
```http
GET /api/devices?page=1&page_size=20&brand=Apple&min_price=500&max_price=1000
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20)
- `brand` (string): Filter by brand
- `chipset` (string): Filter by chipset manufacturer
- `min_price` (float): Minimum price in USD
- `max_price` (float): Maximum price in USD
- `search` (string): Search in model name/brand

**Response:**
```json
{
  "devices": [
    {
      "id": 1,
      "brand": "Apple",
      "model_name": "iPhone 15 Pro Max",
      "price": "$1199.00",
      "chipset": "Apple A17 Pro",
      "announced": "2023, September",
      "display_size": "6.7 inches",
      "main_camera_features": "48 MP (wide), 12 MP (telephoto), 12 MP (ultrawide)",
      ...
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

#### Get Single Device
```http
GET /api/devices/{device_id}
```

#### Get All Brands
```http
GET /api/devices/brands
```

**Response:**
```json
{
  "brands": ["Apple", "Samsung", "Google", "OnePlus", ...]
}
```

#### Get All Chipsets
```http
GET /api/devices/chipsets
```

**Response:**
```json
{
  "chipsets": [
    "Qualcomm",
    "MediaTek",
    "Apple",
    "Samsung",
    "HiSilicon",
    "Google",
    "UNISOC",
    "NVIDIA",
    "Intel",
    "Leadcore Technology"
  ]
}
```

### NLP Search Endpoint

#### Natural Language Search
```http
POST /api/devices/search/nlp?query=gaming%20phone%20under%20500&limit=10
```

**Query Parameters:**
- `query` (string): Natural language search query
- `limit` (int): Number of results (default: 10)

**Example Queries:**
- "gaming phone under $500"
- "best camera phone"
- "phone with long battery life"
- "flagship phone not Samsung"
- "budget phone with good chipset"

**Response:**
```json
[
  {
    "id": 234,
    "brand": "OnePlus",
    "model_name": "OnePlus 12R",
    "price": "$499.99",
    "chipset": "Qualcomm Snapdragon 8 Gen 2",
    "score": 0.95,
    "reasoning": "Excellent gaming performance, within budget"
  },
  ...
]
```

### Recommendation Endpoint

#### Get AI Recommendations
```http
POST /api/recommend
Content-Type: application/json
Authorization: Bearer <access_token>

{
  "query": "I need a phone for photography with good low light performance",
  "top_n": 10,
  "budget": 800,
  "preferred_brand": "Google"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "device": {
        "id": 156,
        "brand": "Google",
        "model_name": "Pixel 8 Pro",
        ...
      },
      "score": 0.94,
      "reasoning": "Excellent camera with advanced night mode, within budget",
      "explanation": {
        "camera_score": 0.98,
        "chipset_score": 0.92,
        "budget_match": true
      }
    }
  ],
  "parsed_preferences": {
    "use_case": "photography",
    "budget": 800,
    "preferred_brand": "Google",
    "key_features": ["camera", "low light"]
  }
}
```

### Comparison Endpoints

#### Create Comparison
```http
POST /api/comparisons
Content-Type: application/json
Authorization: Bearer <access_token>

{
  "device_ids": [1, 15, 42],
  "name": "Flagship Comparison 2024"
}
```

#### Get User Comparisons
```http
GET /api/comparisons
Authorization: Bearer <access_token>
```

### Price Tracking Endpoints

#### Subscribe to Price Alerts
```http
POST /api/price-track/subscribe
Content-Type: application/json
Authorization: Bearer <access_token>

{
  "device_id": 123,
  "target_price": 699.99
}
```

#### Get Price History
```http
GET /api/price-track/{device_id}/history
```

---

## 🎨 Frontend Features

### 1. Natural Language Search (Navbar)

Located in the navigation bar, users can type queries like:
- "best battery phone"
- "gaming phone under 600 dollars"
- "phone with good camera not Apple"

The system:
1. Parses the query using BERT NLP
2. Extracts structured preferences (brand, budget, features)
3. Returns top 10 relevant devices
4. Sorts by release date (newest first)

### 2. Advanced Filtering (Sidebar)

- **Brand Filter**: 126 brands loaded dynamically from database
- **Chipset Filter**: 10 major chipset manufacturers
- **Price Range**: Min/max USD price sliders
- **Search**: Real-time search in visible devices

### 3. Device Cards

Each device card displays:
- Brand logo/image
- Model name
- Price in USD (or informational text like "Coming soon")
- Key specs (chipset, camera, display)
- Quick action buttons (favorite, compare, details)

### 4. Product Details Page

Comprehensive device information:
- Full specifications across 40+ attributes
- Display, camera, performance, connectivity
- Battery, charging, sensors
- Benchmark scores (AnTuTu, Geekbench)
- Available colors and variants
- Price history chart

### 5. Comparison View

Side-by-side comparison of up to 4 devices:
- Highlights differences
- Color-coded better/worse values
- Export comparison as PDF/Image

### 6. AI Recommendation Page

Interactive form for personalized recommendations:
- Natural language query input
- Structured preference form (budget, use case, brand)
- AI-powered results with explanations
- Confidence scores and reasoning

---

## 🧠 Machine Learning Models

### 1. Advanced NLP Parser (`advanced_nlp_parser.py`)

**Models Used:**
- `sentence-transformers/all-MiniLM-L6-v2`: Semantic similarity
- `dslim/bert-base-NER`: Named entity recognition
- `deepset/roberta-base-squad2`: Question answering

**Capabilities:**
- Extracts budget constraints ($500, 500 dollars, under 600)
- Identifies brand preferences (Apple, Samsung, not Xiaomi)
- Detects use cases (gaming, photography, battery life)
- Understands device types (phone, tablet, smartwatch)
- Handles negations (not Samsung, excluding Apple)

**Example:**
```python
query = "gaming phone under $600 with good camera, not Samsung"
result = parser.parse_complex_query(query)

# Output:
{
  "device_type": "phone",
  "budget": 600,
  "excluded_brands": ["Samsung"],
  "use_case": "gaming",
  "key_features": ["camera"],
  "preferences": {
    "chipset": "high-performance",
    "camera": "good"
  }
}
```

### 2. Recommendation Engine (`recommender.py`)

**Algorithm:** Content-based filtering with TF-IDF

**Features:**
- Uses 40+ device attributes for similarity
- Weighted scoring based on user preferences
- Boost factors for gaming, photography, battery use cases
- Explains why each device was recommended

**Scoring Factors:**
- Chipset performance (AnTuTu, Geekbench scores)
- Camera quality (MP, features, video capabilities)
- Display quality (size, resolution, refresh rate)
- Battery capacity and charging speed
- Price-to-value ratio
- Brand preferences

### 3. Explainable AI (`xai_explainer.py`)

Provides transparency on recommendations:
- Feature importance visualization
- Decision tree reasoning
- Confidence scores per attribute
- "Why this device?" explanations

---

## 🔧 Recent Improvements

### ✅ Currency Conversion System
- Converts 10+ currencies to USD (EUR, INR, GBP, CNY, JPY, AUD, CAD, RUB, BRL)
- Exchange rates embedded in code (configurable)
- Example: "About 290 EUR" → "$319.00"

### ✅ Improved Price Display
- Empty prices show as blank instead of "$0.00"
- Preserves informational text like "Coming soon", "Discontinued"
- Filters out invalid prices (< $1)

### ✅ Date-Based Sorting
- Parses various date formats ("2024, January", "2023, Q4", "2024-Jan-15")
- Sorts recommendations by release date (newest first)
- Handles missing dates gracefully

### ✅ Filter System Overhaul
- Backend endpoints for dynamic filter loading
- `/api/devices/brands` returns all 126 brands
- `/api/devices/chipsets` returns 10 chipset manufacturers
- Frontend correctly parses nested response format

### ✅ NLP Search Limit
- Returns top 10 devices for NLP queries (configurable)
- Prevents overwhelming results
- Optimized for UX

### ✅ Windows Encoding Fix
- Replaced emoji characters with ASCII-safe text
- Fixed `UnicodeEncodeError` on Windows console
- Backend now starts reliably on Windows

---

## 📊 Database Schema

### Devices Table

```sql
CREATE TABLE devices (
    id INT PRIMARY KEY AUTO_INCREMENT,
    brand VARCHAR(100),
    model_name VARCHAR(255),
    model_image TEXT,
    technology VARCHAR(100),
    announced VARCHAR(100),
    status VARCHAR(100),
    dimensions VARCHAR(100),
    weight VARCHAR(100),
    build TEXT,
    sim VARCHAR(100),
    display_type TEXT,
    display_size VARCHAR(100),
    display_resolution VARCHAR(100),
    os VARCHAR(255),
    chipset VARCHAR(255),
    cpu VARCHAR(255),
    gpu VARCHAR(255),
    internal_storage VARCHAR(255),
    card_slot VARCHAR(255),
    main_camera_features TEXT,
    main_camera_video VARCHAR(255),
    selfie_camera_single VARCHAR(255),
    loudspeaker VARCHAR(100),
    jack_35mm VARCHAR(100),
    wlan VARCHAR(255),
    bluetooth VARCHAR(255),
    nfc VARCHAR(100),
    usb VARCHAR(255),
    price VARCHAR(255),
    battery_capacity VARCHAR(255),
    charging TEXT,
    antutu VARCHAR(100),
    geekbench VARCHAR(100),
    speed VARCHAR(255),
    colors TEXT,
    sensors TEXT,
    bands_2g TEXT,
    bands_3g TEXT,
    bands_4g TEXT,
    bands_5g TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_brand (brand),
    INDEX idx_chipset (chipset),
    INDEX idx_price (price),
    INDEX idx_announced (announced)
);
```

**Total Records:** 10,000+ devices

**Brands:** 126 manufacturers (Apple, Samsung, Google, OnePlus, Xiaomi, Oppo, Vivo, Realme, Motorola, Nokia, etc.)

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Test MySQL connection
python test_mysql_conn.py

# Test NLP parser
python ml/test_complex_prompts.py

# Test dataset integration
python ml/test_dataset_integration.py

# Test API endpoints
python test_api.py
```

### Frontend Tests

```bash
# Run development server and test manually
npm run dev

# Build production bundle
npm run build

# Preview production build
npm run preview
```

### Test NLP Queries

```bash
# From backend directory
python ml/interactive_prompt_tester_v2.py
```

**Test Cases:**
- "best gaming phone under $800"
- "phone with excellent camera for photography"
- "budget phone with long battery life"
- "flagship phone not Samsung"
- "phone with 5G support and fast charging"

---

## 🚢 Deployment

### Backend Deployment (Render/Railway/Heroku)

1. Set environment variables in platform dashboard
2. Add `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
3. Ensure MySQL database is accessible
4. Deploy!

### Frontend Deployment (Vercel/Netlify)

1. Connect GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Add environment variable: `VITE_API_URL=https://your-backend.com`
5. Deploy!

**Vercel Configuration** (`vercel.json`):
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style

**Python:**
- Follow PEP 8
- Use type hints
- Document functions with docstrings

**JavaScript/React:**
- Use ES6+ syntax
- Functional components with hooks
- PropTypes or TypeScript for type checking

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👨‍💻 Authors

- **SmartAI Team** - Initial work

---

## 🙏 Acknowledgments

- **GSMArena** - Device dataset
- **Hugging Face** - Pre-trained BERT models
- **FastAPI** - Amazing Python framework
- **React** - Powerful UI library
- **Tailwind CSS** - Utility-first CSS framework

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: support@smartai.com
- Documentation: [Swagger UI](http://localhost:8000/docs)

---

## 🗺️ Roadmap

### Phase 1 (Completed ✅)
- [x] Natural language search with NLP
- [x] Multi-currency price conversion
- [x] Brand and chipset filtering
- [x] Date-based sorting
- [x] MySQL database integration
- [x] AI-powered recommendations

### Phase 2 (In Progress 🚧)
- [ ] Redis caching for faster responses
- [ ] Email price alerts with SendGrid
- [ ] Advanced comparison analytics
- [ ] User preference learning
- [ ] Mobile app (React Native)

### Phase 3 (Planned 📋)
- [ ] Real-time price scraping
- [ ] Community reviews and ratings
- [ ] Social features (share, discuss)
- [ ] Affiliate links for purchases
- [ ] Multi-language support

---

## 📈 Performance

- **Backend Response Time:** < 200ms (average)
- **NLP Query Processing:** < 500ms (with model caching)
- **Database Queries:** < 50ms (indexed)
- **Frontend Load Time:** < 2s (production build)
- **Lighthouse Score:** 95+ (Performance, Accessibility)

---

## 🔐 Security

- JWT tokens with 30-minute expiry
- Bcrypt password hashing (12 rounds)
- CORS protection
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection (React automatic escaping)
- Rate limiting (configurable)
- HTTPS recommended for production

---

## 💡 Tips & Tricks

### Optimize NLP Search
```python
# Cache BERT models on first load (automatic)
# Use shorter queries for faster inference
# Limit results to top 10 for best UX
```

### Improve Performance
```bash
# Enable Redis caching
REDIS_URL=redis://localhost:6379/0

# Add database connection pooling
DATABASE_URL=mysql+pymysql://user:pass@host:3306/db?pool_size=20&max_overflow=10
```

### Debug Mode
```bash
# Backend
python main.py --reload --log-level debug

# Frontend
npm run dev -- --debug
```

---

**Built with ❤️ by SmartAI Team**

*Last Updated: March 2026*
- **Filters**: Use the sidebar to filter by:
  - Device Type (Mobile, Tablet, Smartwatch)
  - Brand
  - Processor
  - Price Range

### Viewing Device Details

Click on any device card to view:
- High-resolution device image
- Complete specifications
- All features and technical details
- Pricing information

## Project Structure

```
src/
├── components/
│   ├── Navbar.jsx          # Navigation bar with search and upload
│   ├── Sidebar.jsx          # Filter sidebar
│   └── ProductCard.jsx      # Device card component
├── pages/
│   ├── Home.jsx             # Home page with featured devices
│   ├── Products.jsx         # Product listing page
│   └── ProductDetail.jsx    # Device detail page
├── services/
│   └── api.js               # API service for data management
├── App.jsx                  # Main app component
├── main.jsx                 # Entry point
└── index.css                # Global styles
```

## CSV Format

The application expects CSV files with the following structure:
- `Brand` - Device brand name
- `Model Name` - Device model name
- `Model Image` - URL to device image
- Additional columns for specifications (OS, Chipset, CPU, Display, Camera, etc.)

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## License

MIT

