from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import settings

# Import routers
from routers import auth, devices, users, recommendations, price_tracking, comparisons


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection on startup and close on shutdown"""
    # Startup
    print(f"Starting SmartAI Device Filter API...")
    if settings.db_type.lower() == 'mysql':
        print(f"📦 Using MySQL: {settings.mysql_url}")
        print(f"✅ Database: {settings.database_name}")
    else:
        print(f"📦 Using MongoDB: {settings.mongodb_url}")
    
    yield
    
    # Shutdown
    print("🔴 Shutting down SmartAI Device Filter API...")


# Create FastAPI app
app = FastAPI(
    title="SmartAI Device Filter API",
    description="AI-powered smart device recommendation and filtering system",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(devices.router, prefix="/api/devices", tags=["Devices"])
app.include_router(users.router, prefix="/api/user", tags=["User"])
app.include_router(recommendations.router, prefix="/api/recommend", tags=["AI Recommendations"])
app.include_router(price_tracking.router, prefix="/api/price-track", tags=["Price Tracking"])
app.include_router(comparisons.router, prefix="/api/compare", tags=["Comparisons"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SmartAI Device Filter API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "database": settings.database_name
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
