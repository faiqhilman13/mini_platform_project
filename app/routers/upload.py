# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlmodel import Session
from app.services import file_service
from app.models.file_models import FileUploadResponse, UploadedFileLog
from app.core.config import settings
from app.db.session import get_session
from typing import List

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
    Uploads a file (PDF, CSV, JSON, Excel) to the server.

    The file is saved locally, and metadata is logged to the database.
    """
    # Enhanced validation for allowed file types (DS1.1.1: Added Excel support)
    allowed_content_types = [
        "application/pdf", 
        "text/csv", 
        "application/json",
        "text/plain",  # Added for text classifier temporary files
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel"  # .xls
    ]
    
    # Also check file extension as content-type detection can be unreliable
    allowed_extensions = [".pdf", ".csv", ".json", ".txt", ".xlsx", ".xls"]
    file_extension = None
    if file.filename:
        file_extension = "." + file.filename.split(".")[-1].lower()
    
    if file.content_type not in allowed_content_types and file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type} (extension: {file_extension}). Allowed types are PDF, CSV, JSON, Excel (.xlsx, .xls)."
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

@router.get("/files", response_model=List[dict])
async def get_uploaded_files(
    db: Session = Depends(get_session)
):
    """
    Retrieves a list of all uploaded files.
    
    Returns file metadata including ID, filename, file type, size, and upload timestamp.
    """
    try:
        files = await file_service.get_uploaded_files(db=db)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving uploaded files: {str(e)}")

@router.delete("/files/{file_id}")
async def delete_uploaded_file(
    file_id: int,
    db: Session = Depends(get_session)
):
    """
    Deletes an uploaded file and its associated data.
    
    This removes the file from the filesystem, database record, and any related pipeline runs.
    """
    try:
        result = await file_service.delete_uploaded_file(file_id=file_id, db=db)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        return {"message": result["message"], "success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
