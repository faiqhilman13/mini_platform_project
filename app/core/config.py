from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Mini IDP"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./mini_idp.db" # Default to SQLite for MVP
    REDIS_URL: str = "redis://localhost:6379/0" # Default Redis URL

    # If using .env file for configuration:
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings() 