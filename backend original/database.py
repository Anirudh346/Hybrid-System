from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import settings
from models.base import Base

# Create SQLAlchemy engine for MySQL
engine = create_engine(
    settings.mysql_url,
    echo=False,
    pool_pre_ping=True
)

# Create session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
