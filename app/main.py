from dotenv import load_dotenv
load_dotenv()

# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager # Import asynccontextmanager

from app.core.config import settings # Ensure settings is imported if not already
from app.routers import upload # Import the new router
from app.routers import pipelines as pipelines_router # Import the new pipeline router
from app.routers import rag as rag_router  # Import the new RAG router
from app.routers import data as data_router  # Import the data router
from app.routers.ml import ml_router  # Import the ML router - now uses existing pipeline infrastructure
from app.db.session import create_db_and_tables # Import the function

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("INFO:     Creating database and tables...")
    create_db_and_tables()
    print("INFO:     Database and tables created (if they didn't exist).")
    yield
    # Code to run on shutdown (if any)
    print("INFO:     Application shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME, # Use project name from settings
    version="0.1.0",
    description="API for the Internal Developer Platform for AI Workflows",
    lifespan=lifespan # Use the lifespan context manager
)

# CORS Middleware (for development)
# In a production environment, you should restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create database tables on startup - REMOVED @app.on_event("startup")
# In a more complex setup, you might use Alembic for migrations.
# def on_startup():
#     print("INFO:     Creating database and tables...")
#     create_db_and_tables()
#     print("INFO:     Database and tables created (if they didn't exist).")

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Checks the health of the application.
    """
    return {"status": "success", "message": "API is healthy"}

# Include routers
app.include_router(upload.router)
app.include_router(pipelines_router.router) # Include the pipeline router
app.include_router(rag_router.router) # Include the RAG router
app.include_router(data_router.router) # Include the data router
app.include_router(ml_router)  # Include the ML router

# Placeholder for future routers
# from routers import pipeline_router # Example
# app.include_router(pipeline_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Ensure reload for dev 