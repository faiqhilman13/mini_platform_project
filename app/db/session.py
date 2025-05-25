from sqlmodel import create_engine, SQLModel, Session
from app.core.config import settings

# The database URL is taken from settings
# For SQLite, it will be something like: "sqlite:///./mini_idp.db"
# connect_args is specific to SQLite to allow shared cache and disable same-thread check for background tasks.
engine = create_engine(settings.DATABASE_URL, echo=True, connect_args={"check_same_thread": False})

def create_db_and_tables():
    """
    Creates all database tables defined by SQLModel metadata.
    This should be called on application startup.
    """
    # Import all models here that need to be created
    # This seems like a common pattern, but can also be managed by Alembic later
    from app.models import file_models # Ensures FileUploadResponse is known (though not a table model)
    from app.models import pipeline_models # Ensures PipelineRun and UploadedFileLog are known
    
    SQLModel.metadata.create_all(engine)

def get_session():
    """
    FastAPI dependency to get a database session.
    Ensures the session is closed after the request.
    """
    with Session(engine) as session:
        yield session 