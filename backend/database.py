"""
SQLAlchemy Database Configuration for MySQL
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from config import Settings
import logging

logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

def _normalize_mysql_url(raw_url: str) -> tuple[str, dict]:
    """Normalize provider URLs for SQLAlchemy + PyMySQL.

    Some providers expose URLs like mysql://... and may include ssl query values
    as strings (e.g. ssl=true), but PyMySQL expects ssl to be a dict.
    """
    url = make_url(raw_url)
    is_local_host = url.host in {"localhost", "127.0.0.1", "::1"}

    # Ensure explicit PyMySQL driver.
    if url.drivername == "mysql":
        url = url.set(drivername="mysql+pymysql")

    query = dict(url.query)
    connect_args = {}

    ssl_value = query.get("ssl")
    ssl_mode = query.get("ssl_mode") or query.get("sslmode")

    # Convert common string SSL flags to a PyMySQL-compatible dict.
    if isinstance(ssl_value, str):
        if ssl_value.strip().lower() in {"1", "true", "yes", "require", "required"}:
            connect_args["ssl"] = {}
            query.pop("ssl", None)

    # Also support SQLAlchemy-style ssl_mode from URL.
    if isinstance(ssl_mode, str):
        if ssl_mode.strip().lower() in {"require", "required", "verify_ca", "verify_identity"}:
            connect_args.setdefault("ssl", {})
            query.pop("ssl_mode", None)
            query.pop("sslmode", None)

    # Railway and similar hosted MySQL providers typically expect SSL.
    # If no SSL option is provided, enable it automatically for non-local hosts.
    if not is_local_host and "ssl" not in connect_args:
        connect_args["ssl"] = {}

    normalized_url = str(url.set(query=query))
    return normalized_url, connect_args


# MySQL connection string
DATABASE_URL, CONNECT_ARGS = _normalize_mysql_url(settings.database_url)

# Log safe connection metadata so deployment logs can confirm which env values are active.
try:
    parsed_db_url = make_url(DATABASE_URL)
    print(
        "[DB TARGET] "
        f"driver={parsed_db_url.drivername} "
        f"user={parsed_db_url.username} "
        f"host={parsed_db_url.host} "
        f"port={parsed_db_url.port} "
        f"db={parsed_db_url.database} "
        f"ssl_enabled={'ssl' in CONNECT_ARGS}"
    )
except Exception as exc:
    print(f"[DB TARGET] Unable to parse DATABASE_URL for diagnostics: {exc}")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    pool_pre_ping=True,  # Test connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args=CONNECT_ARGS,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base for all models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
    except Exception as e:
        logger.error(f"✗ Error creating database tables: {str(e)}")
        raise
