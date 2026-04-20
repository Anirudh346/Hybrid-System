from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MySQL Database
    database_url: str = Field(
        default="mysql+pymysql://root:123@localhost:3306/device_catalog",
        validation_alias=AliasChoices("DATABASE_URL", "MYSQL_URL", "MYSQL_PUBLIC_URL"),
    )
    database_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("DATABASE_HOST", "MYSQLHOST"),
    )
    database_port: int = Field(
        default=3306,
        validation_alias=AliasChoices("DATABASE_PORT", "MYSQLPORT"),
    )
    database_user: str = Field(
        default="root",
        validation_alias=AliasChoices("DATABASE_USER", "MYSQLUSER"),
    )
    database_password: str = Field(
        default="123",
        validation_alias=AliasChoices("DATABASE_PASSWORD", "MYSQLPASSWORD", "MYSQL_ROOT_PASSWORD"),
    )
    database_name: str = Field(
        default="device_catalog",
        validation_alias=AliasChoices("DATABASE_NAME", "MYSQL_DATABASE", "MYSQLDATABASE"),
    )
    
    # JWT
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        validation_alias=AliasChoices("SECRET_KEY"),
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL"),
    )
    
    # Email
    sendgrid_api_key: str = Field(default="", validation_alias=AliasChoices("SENDGRID_API_KEY"))
    from_email: str = Field(default="noreply@smartai.com", validation_alias=AliasChoices("FROM_EMAIL"))
    
    # CORS
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:5174,http://192.168.29.153:5174",
        validation_alias=AliasChoices("ALLOWED_ORIGINS"),
    )
    
    # Environment
    environment: str = Field(default="development", validation_alias=AliasChoices("ENVIRONMENT"))
    
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
