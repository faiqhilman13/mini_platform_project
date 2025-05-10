from fastapi import UploadFile, Depends
from sqlmodel import Session
from app.models.file_models import FileUploadResponse, UploadedFileLog, UploadedFileLogCreate
from app.db.session import get_session # Import get_session
import os

# Define a directory to store uploads, ensure it exists
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

async def save_uploaded_file_and_log(
    file: UploadFile,
    db: Session = Depends(get_session) # Add db session dependency
) -> FileUploadResponse:
    """
    Saves the uploaded file to a local directory, logs metadata to the DB,
    and returns metadata including the database log ID.
    """
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    # # Reason: Using a temporary file and then moving it can be safer
    # # to prevent partial writes if the connection drops.
    # temp_file_path = f"{file_location}.tmp"
    # with open(temp_file_path, "wb+") as file_object:
    #     shutil.copyfileobj(file.file, file_object)
    # os.rename(temp_file_path, file_location)

    # Simpler approach for now: direct save
    # Be cautious with large files in a synchronous way if not careful
    try:
        with open(file_location, "wb+") as file_object:
            # Read file in chunks to handle large files efficiently
            while chunk := await file.read(1024*1024): # Read 1MB chunks
                file_object.write(chunk)
        
        file_size = os.path.getsize(file_location)

        # Create UploadedFileLog entry
        file_log_create = UploadedFileLogCreate(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=file_size,
            storage_location=file_location 
        )
        db_file_log = UploadedFileLog.model_validate(file_log_create)
        db.add(db_file_log)
        db.commit()
        db.refresh(db_file_log)

        return FileUploadResponse(
            filename=db_file_log.filename,
            content_type=db_file_log.content_type,
            size=db_file_log.size_bytes,
            message=f"File '{db_file_log.filename}' saved to {db_file_log.storage_location} and logged with ID {db_file_log.id}",
            file_log_id=db_file_log.id,
            file_uuid=db_file_log.file_uuid
        )
    except Exception as e:
        db.rollback() # Rollback DB changes on error
        print(f"Error saving file or logging: {e}")
        if os.path.exists(file_location):
            try:
                os.remove(file_location) # Attempt to clean up partially saved file
            except OSError as oe:
                print(f"Error removing partially saved file '{file_location}': {oe}")
        return FileUploadResponse(
            filename=file.filename,
            content_type=file.content_type,
            size=0, 
            message=f"Failed to save file '{file.filename}'. Error: {str(e)}",
            file_log_id=None,
            file_uuid=None
        )
    finally:
        await file.close() 