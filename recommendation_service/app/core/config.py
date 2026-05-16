from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Recommendation Service"
    API_V1_STR: str = "/api/v1"
    
    # DATABASE
    POSTGRES_SERVER: str = "postgres-service"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "auth_db"  # Shared DB
    POSTGRES_SCHEMA: str = "search"
    
    # REDIS (Vector Store)
    REDIS_HOST: str = "redis"
    RECO_REDIS_PORT: int = 6379
    
    # KAFKA
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    
    # EXTERNAL SERVICES
    UPLOAD_SERVICE_URL: str = "http://upload-service:8001"
    USER_SERVICE_URL: str = "http://user-service:8006"
    INTERACTION_SERVICE_URL: str = "http://interaction-service:8004"
    COMMENT_SERVICE_URL: str = "http://comment-service:8005"
    
    # ML MODEL
    # paraphrase-multilingual-mpnet-base-v2: SOTA for multilingual (including Korean)
    MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    PROJECTED_DIM: int = 128  # Target dimension for Item/User Towers

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    MODEL_SAVE_PATH: str = "discovery_engine.pth"

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
