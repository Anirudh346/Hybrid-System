# SmartAI Device Filter - Complete Setup Guide

## 🎯 Overview

Full-stack AI-powered device recommendation platform with:
- ✅ FastAPI Backend with MongoDB
- ✅ JWT Authentication
- ✅ AI/ML Recommendations (NLP + Content-based filtering)
- ✅ User Favorites & Saved Searches
- ✅ Price Tracking & History
- ✅ Device Comparison (up to 4 devices)
- ✅ React Frontend with Dark Mode

---

## 📋 Prerequisites

- **Python 3.9+**
- **Node.js 16+**
- **MongoDB Atlas account** (free tier) OR local MongoDB
- **Git**

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd "C:\Users\rudra\Desktop\Major Project\SmartAI-SmartDeviceFilter-main"
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The BERT NER model will be automatically downloaded on first use.

#### Configure Environment

```bash
# Copy example env file
cp .env.example .env
```

Edit `backend/.env`:

```env
# MongoDB - Get connection string from MongoDB Atlas
MONGODB_URL=mongodb+srv://your-username:your-password@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=smartai_devices

# JWT - Generate with: openssl rand -hex 32
SECRET_KEY=your-generated-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis (optional for now)
REDIS_URL=redis://localhost:6379/0

# Email (optional for price alerts)
SENDGRID_API_KEY=
FROM_EMAIL=noreply@smartai.com

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Environment
ENVIRONMENT=development
```

#### Generate Secret Key

**PowerShell:**
```powershell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
```

Or use Python:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

#### Import CSV Data to MongoDB

```bash
# Import from existing GSMArenaDataset
python import_csv.py ../major/GSMArenaDataset

# Or if you have data elsewhere:
python import_csv.py path/to/your/csv/folder
```

#### Run Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be running at: **http://localhost:8000**

API Docs: **http://localhost:8000/docs**

---

### 3. Frontend Setup

Open new terminal:

```bash
cd SmartAI-SmartDeviceFilter-main
```

#### Install Dependencies

```bash
npm install
```

#### Configure Environment

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

#### Run Frontend

```bash
npm run dev
```

Frontend will be running at: **http://localhost:5173**

---

## 📱 Usage Guide

### First Time Setup

1. **Start Backend** (Terminal 1):
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start Frontend** (Terminal 2):
   ```bash
   cd SmartAI-SmartDeviceFilter-main
   npm run dev
   ```

3. **Visit**: http://localhost:5173

4. **Create Account**: Click "Sign Up" in top-right corner

5. **Browse Devices**: Explore devices loaded from MongoDB

---

### Features Walkthrough

#### 1. **AI Recommendations** 🤖

- Navigate to "AI Recommendation" (requires login)
- Enter natural language query:
  - "best gaming phone under $800"
  - "camera phone for photography"
  - "affordable Samsung tablet"
- Get AI-scored recommendations with reasoning

#### 2. **User Favorites** ⭐

- Click heart icon on any device card
- View all favorites in user menu
- Add notes to favorites

#### 3. **Device Comparison** 🔄

- Select up to 4 devices using checkboxes
- Click "Compare" in floating bar
- View side-by-side spec comparison
- Save comparison (requires login)

#### 4. **Price Tracking** 💰

- View price history charts on device pages
- Subscribe to price drop alerts
- Track across Amazon/Flipkart (when scraper runs)

#### 5. **Saved Searches** 🔍

- Apply filters on Products page
- Click "Save Search"
- Quick access to favorite filter combinations

---

## 🗄️ MongoDB Atlas Setup

### Create Free Cluster

1. Go to https://www.mongodb.com/cloud/atlas/register
2. Create free M0 cluster
3. Choose cloud provider & region
4. Create database user
5. Whitelist IP: `0.0.0.0/0` (allow all for development)
6. Get connection string
7. Replace `<password>` in connection string
8. Paste into `backend/.env` as `MONGODB_URL`

---

## 🧪 Testing the System

### Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# Register user (PowerShell)
$body = @{
    email = "test@example.com"
    password = "testpassword123"
    full_name = "Test User"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/auth/register" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

### Test Frontend

1. Open http://localhost:5173
2. Click "Sign Up"
3. Create account
4. Browse products
5. Try AI recommendation

---

## 📊 Project Structure

```
SmartAI-SmartDeviceFilter-main/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Entry point
│   ├── config.py              # Settings
│   ├── import_csv.py          # Data import script
│   ├── models/                # MongoDB models
│   ├── routers/               # API endpoints
│   ├── schemas/               # Pydantic schemas
│   ├── ml/                    # AI/ML modules
│   ├── utils/                 # JWT utilities
│   └── requirements.txt
│
├── SmartAI-SmartDeviceFilter-main/  # React Frontend
│   ├── src/
│   │   ├── components/        # Reusable components
│   │   ├── contexts/          # React contexts
│   │   ├── pages/             # Route pages
│   │   ├── services/          # API client
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── major/
│   └── GSMArenaDataset/       # CSV data files
│
└── main.py                    # Original scraper
```

---

## 🔧 Development Tips

### Backend Development

```bash
# Auto-reload on code changes
uvicorn main:app --reload

# View logs
# Check terminal for request logs

# Access interactive API docs
http://localhost:8000/docs
```

### Frontend Development

```bash
# Hot module replacement
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🐛 Troubleshooting

### MongoDB Connection Failed

- Check internet connection
- Verify connection string in `.env`
- Ensure IP is whitelisted in MongoDB Atlas
- Check username/password are correct

### Backend Won't Start

```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check for port conflicts
# Kill process on port 8000 if needed
```

### Frontend Won't Start

```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

### BERT Model Download Issues

The BERT model (dslim/bert-base-NER) downloads automatically on first use. If you encounter issues:

```bash
python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER')"
```

### CORS Errors

- Ensure `ALLOWED_ORIGINS` in `backend/.env` includes frontend URL
- Check both servers are running
- Clear browser cache

---

## 📦 Production Deployment

### Backend (Railway/Render/Heroku)

1. Create account on hosting platform
2. Connect GitHub repository
3. Set environment variables:
   - `MONGODB_URL`
   - `SECRET_KEY`
   - `ALLOWED_ORIGINS`
4. Deploy from `backend/` directory
5. Run command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel/Netlify)

1. Connect GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Set environment variable:
   - `VITE_API_BASE_URL=https://your-backend-url.com`
5. Deploy

---

## 🔑 API Endpoints Summary

### Authentication
- `POST /api/auth/register` - Register
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/refresh` - Refresh token

### Devices
- `GET /api/devices` - List with filters
- `GET /api/devices/{id}` - Get device
- `GET /api/devices/brands` - List brands
- `POST /api/devices/upload` - Upload CSV

### User
- `GET /api/user/favorites` - List favorites
- `POST /api/user/favorites` - Add favorite
- `DELETE /api/user/favorites/{id}` - Remove
- `GET /api/user/searches` - Saved searches
- `POST /api/user/searches` - Create search

### AI
- `POST /api/recommend` - Get recommendations

### Comparison
- `GET /api/compare` - List comparisons
- `POST /api/compare` - Create comparison
- `DELETE /api/compare/{id}` - Delete

### Price Tracking
- `POST /api/price-track/subscribe` - Subscribe
- `GET /api/price-track/{id}/history` - Get history

---

## 📝 Next Steps

1. ✅ **Complete Setup** - Follow this guide
2. ✅ **Test All Features** - Create account, browse, compare
3. 🔄 **Setup Price Scraper** - Configure Celery for automated scraping
4. 📧 **Configure Email** - Setup SendGrid for price alerts
5. 🚀 **Deploy to Production** - Use Vercel + Railway
6. 📱 **Mobile Optimization** - Test responsive design
7. 🎨 **Custom Branding** - Update logos and colors
8. 📊 **Analytics** - Add Google Analytics

---

## 🆘 Support

- **Documentation**: See `backend/README.md`
- **API Docs**: http://localhost:8000/docs
- **Issues**: Check terminal logs

---

## 📜 License

MIT License - Feel free to use for your projects!

---

**Built with ❤️ using FastAPI, React, MongoDB, and AI/ML**
