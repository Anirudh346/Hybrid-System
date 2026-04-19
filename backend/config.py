from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MySQL Database
    database_url: str = "mysql+pymysql://root:123@localhost:3306/device_catalog"
    database_host: str = "localhost"
    database_port: int = 3306
    database_user: str = "root"
    database_password: str = "123"
    database_name: str = "device_catalog"
    
    # JWT
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Email
    sendgrid_api_key: str = ""
    from_email: str = "noreply@smartai.com"
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://192.168.29.153:5174"
    
    # Environment
    environment: str = "development"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )
    
    @property
    def origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


settings = Settings(
    secret_key="smartai-dev-key-2024"
)
