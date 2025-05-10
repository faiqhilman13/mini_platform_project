# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlmodel import Session
from app.services import file_service
from app.models.file_models import FileUploadResponse
from app.core.config import settings
from app.db.session import get_session

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/upload",
    tags=["File Upload"],
)

@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_session)
):
    """
    Uploads a file (PDF, CSV, JSON) to the server.

    The file is saved locally, and metadata is logged to the database.
    """
    # Basic validation for allowed file types (as per PRD: PDF, CSV, JSON)
    allowed_content_types = ["application/pdf", "text/csv", "application/json"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed types are PDF, CSV, JSON."
        )

    # # Reason: File size limit can be configured globally or per endpoint.
    # # For now, let's assume a reasonable default if not specified.
    # MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    # if file.size > MAX_FILE_SIZE:
    #     raise HTTPException(
    #         status_code=413, # Payload Too Large
    #         detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
    #     )

    result = await file_service.save_uploaded_file_and_log(file=file, db=db)
    if "Failed to save file" in result.message or not result.file_log_id:
        # If the service indicates failure, return an appropriate HTTP error
        raise HTTPException(status_code=500, detail=result.message)
    return result
