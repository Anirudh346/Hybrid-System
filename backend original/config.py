from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB
    mongodb_url: str
    database_name: str = "device_catalog"
    # Database selection: 'mongodb' or 'mysql'
    db_type: str = "mysql"
    # MySQL (SQLAlchemy) URL using PyMySQL
    mysql_url: str = "mysql+pymysql://device_user:123@localhost:3306/device_catalog"
    
    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Email
    sendgrid_api_key: str = ""
    from_email: str = "noreply@smartai.com"
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Environment
    environment: str = "development"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )
    
    @property
    def origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


settings = Settings(
    mongodb_url="mongodb://localhost:27017",
    secret_key="your-secret-key-here-change-in-production",
    mysql_url="mysql+pymysql://device_user:123@localhost:3306/device_catalog"
)
