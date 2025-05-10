# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings # Ensure settings is imported if not already
from app.routers import upload # Import the new router
from app.db.session import create_db_and_tables # Import the function

app = FastAPI(
    title=settings.APP_NAME, # Use app name from settings
    version="0.1.0",
    description="API for the Internal Developer Platform for AI Workflows"
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

# Create database tables on startup
# In a more complex setup, you might use Alembic for migrations.
@app.on_event("startup")
def on_startup():
    print("INFO:     Creating database and tables...")
    create_db_and_tables()
    print("INFO:     Database and tables created (if they didn't exist).")

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Checks the health of the application.
    """
    return {"status": "success", "message": "API is healthy"}

# Include routers
app.include_router(upload.router)

# Placeholder for future routers
# from app.routers import pipeline_router # Example
# app.include_router(pipeline_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Ensure reload for dev 