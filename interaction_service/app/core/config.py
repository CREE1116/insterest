from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    PROJECT_NAME: str = "Interaction Service"
    API_V1_STR: str = "/api/v1"
    POSTGRES_SERVER: str = "postgres-service"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "auth_db"
    POSTGRES_SCHEMA: str = "interaction"
    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")
settings = Settings()
