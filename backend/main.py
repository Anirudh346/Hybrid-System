from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from config import Settings
from database import init_db, engine
from models.device import Base

# Import routers
from routers import devices

# Load settings
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection on startup and close on shutdown"""
    # Startup
    init_db()
    print(f"[OK] Connected to MySQL database: {settings.database_name}")
    
    # Disable heavy NLP preload by default in cloud deploys to keep startup fast.
    preload_nlp = os.getenv("PRELOAD_NLP", "false").strip().lower() in {"1", "true", "yes"}
    if preload_nlp:
        try:
            print("[INFO] Pre-loading NLP models (this may take 30-60 seconds)...")
            from routers.devices import nlp_service_preload
            nlp_service_preload()
            print("[OK] NLP models pre-loaded successfully")
        except Exception as e:
            print(f"[WARN] NLP pre-loading skipped: {str(e)}")
            print("   (NLP will lazy-load on first request)")
    else:
        print("[INFO] NLP preload disabled (set PRELOAD_NLP=true to enable)")
    
    yield
    
    # Shutdown
    engine.dispose()
    print("[CLOSE] Closed MySQL connection")


# Create FastAPI app
app = FastAPI(
    title="SmartAI Device Filter API",
    description="AI-powered smart device recommendation and filtering system",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(devices.router, prefix="/api/devices", tags=["Devices"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartAI Device Filter API",
        "version": "2.0.0",
        "status": "running",
        "database": settings.database_name
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "database": "mysql",
        "ready": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

