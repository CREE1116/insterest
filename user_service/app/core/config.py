from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "User Service"
    API_V1_STR: str = "/api/v1"
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "postgres-service")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "auth_db")
    POSTGRES_SCHEMA: str = "user_service"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "k8s-local-secret-key")
    ALGORITHM: str = "HS256"
    UPLOAD_DIR: str = "/app/profiles"
    UPLOAD_SERVICE_URL: str = os.getenv("UPLOAD_SERVICE_URL", "http://upload-service:8001")

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
