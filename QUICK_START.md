# Quick Start Guide

## Current Status

✅ **Backend configured**: JWT secret key generated and .env created
✅ **Frontend dependencies installed**: Ready to run
✅ **CSV data available**: `major/dataset.csv` and `major/Alcatel.csv`

## Next Steps

### Option A: Use MongoDB Atlas (Recommended - No Installation Required)

1. **Set up MongoDB Atlas** (5 minutes):
   - Follow the guide in `MONGODB_SETUP.md`
   - Get your connection string
   - Update `backend/.env` with your MongoDB Atlas connection string

2. **Import CSV Data**:
   ```bash
   cd backend
   python import_csv.py ../major
   ```

3. **Start the servers**:
   
   **Backend** (in one terminal):
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```
   
   **Frontend** (in another terminal):
   ```bash
   cd SmartAI-SmartDeviceFilter-main
   npm run dev
   ```

4. **Access the app**: http://localhost:3000

### Option B: Install MongoDB Locally

1. **Install MongoDB**:
   - Download from https://www.mongodb.com/try/download/community
   - Install as Windows Service
   - Start the service

2. **The .env is already configured for local MongoDB**:
   ```
   MONGODB_URL=mongodb://localhost:27017
   ```

3. **Follow steps 2-4 from Option A above**

## Testing the Setup

Once both servers are running:

1. **Backend API**: http://127.0.0.1:8000/docs (FastAPI Swagger UI)
2. **Frontend**: http://localhost:3000
3. **Health Check**: http://127.0.0.1:8000/health

## Troubleshooting

### MongoDB Connection Error
- Verify your MONGODB_URL in `backend/.env`
- For Atlas: Ensure your IP is whitelisted
- For local: Ensure MongoDB service is running

### Import Script Fails
- Check that CSV files exist in `major/` folder
- Ensure MongoDB is connected
- Check Python dependencies are installed

### Backend Won't Start
- Check all required packages are installed: `pip install -r requirements.txt`
- Verify .env file exists in backend folder
- Check terminal for specific error messages

## Environment Variables Summary

Your `backend/.env` file is configured with:
- ✅ JWT Secret Key: Generated securely
- ⚠️  MongoDB URL: Currently set to `mongodb://localhost:27017` (update if using Atlas)
- ✅ CORS Origins: Configured for localhost:3000 and localhost:5173
- ✅ Database Name: smartai_devices

## What's Next?

After successful setup:
- Create a user account
- Test AI recommendations with queries like "best gaming phone under $500"
- Add devices to favorites
- Compare multiple devices side-by-side
- View price history (once data is imported)

For detailed information, see `IMPLEMENTATION_GUIDE.md`
