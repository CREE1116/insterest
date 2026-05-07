from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Generation Service"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "k8s-local-secret-key"
    ALGORITHM: str = "HS256"
    
    # Paths
    OUTPUT_DIR: str = "/app/outputs"
    HF_HOME: str = "/app/models"
    
    # Gemini API Key (Required)
    GEMINI_API_KEY: str = ""
    
    # Upload Service URL (for integration)
    UPLOAD_SERVICE_URL: str = "http://upload-service:8001"
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore" # 정의되지 않은 환경 변수는 무시
    )

settings = Settings()
