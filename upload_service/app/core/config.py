from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Upload Service"
    API_V1_STR: str = "/api/v1"
    
    # SECURITY
    SECRET_KEY: str = "DEVELOPMENT_SECRET_KEY_CHANGE_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    
    # DATABASE
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "auth_db"  # auth_db와 동일한 DB 사용
    POSTGRES_SCHEMA: str = "upload" # 업로드 전용 스키마
    # KAFKA
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    GENERATION_SERVICE_URL: str = "http://generation-service:8002"

    # UPLOAD
    UPLOAD_DIR: str = "uploads" # 프로젝트 루트 기준 상대 경로

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
