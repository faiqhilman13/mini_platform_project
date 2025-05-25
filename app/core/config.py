from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Mini IDP - AI Workflow Platform"
    API_V1_STR: str = "/api/v1"

    # Database settings
    DATABASE_URL: str = "sqlite:///./mini_idp.db" # Use SQLite for MVP

    # File Upload Settings
    UPLOADED_FILES_DIR: str = "./uploaded_files"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings() 