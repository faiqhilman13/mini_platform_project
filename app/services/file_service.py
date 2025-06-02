from fastapi import UploadFile, Depends
from sqlmodel import Session, select
from app.models.file_models import FileUploadResponse, UploadedFileLog, UploadedFileLogCreate
from app.models.pipeline_models import PipelineRun
from app.db.session import get_session # Import get_session
import os
import pandas as pd  # DS1.1.1: Added for CSV/Excel validation
from typing import List, Dict

# Define a directory to store uploads, ensure it exists
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def validate_dataset_file(file_path: str, content_type: str) -> tuple[bool, str]:
    """
    Validates CSV/Excel files to ensure they can be read properly.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Only validate dataset files (CSV/Excel)
        if content_type == "text/csv" or file_path.endswith('.csv'):
            # Try to read CSV file
            df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for validation
        elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                              "application/vnd.ms-excel"] or file_path.endswith(('.xlsx', '.xls')):
            # Try to read Excel file
            df = pd.read_excel(file_path, nrows=5)  # Read only first 5 rows for validation
        else:
            # Not a dataset file, skip validation
            return True, ""
        
        # Basic validation checks
        if df.empty:
            return False, "Dataset file is empty"
        
        if len(df.columns) == 0:
            return False, "Dataset file has no columns"
        
        if len(df.columns) > 1000:  # Reasonable limit
            return False, "Dataset file has too many columns (max 1000)"
            
        return True, ""
        
    except pd.errors.EmptyDataError:
        return False, "Dataset file is empty or has no valid data"
    except pd.errors.ParserError as e:
        return False, f"Unable to parse dataset file: {str(e)}"
    except Exception as e:
        return False, f"Error validating dataset file: {str(e)}"

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
        
        # DS1.1.1: Validate dataset files (CSV/Excel)
        is_valid, validation_error = validate_dataset_file(file_location, file.content_type)
        if not is_valid:
            # Clean up the file if validation fails
            if os.path.exists(file_location):
                os.remove(file_location)
            raise ValueError(f"Dataset validation failed: {validation_error}")

        # DS1.1.1: Determine if file is a dataset
        is_dataset = (
            file.content_type == "text/csv" or 
            file.content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                 "application/vnd.ms-excel"] or
            file_location.endswith(('.csv', '.xlsx', '.xls'))
        )
        
        # Create UploadedFileLog entry
        file_log_create = UploadedFileLogCreate(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=file_size,
            storage_location=file_location,
            is_dataset=is_dataset  # DS1.1.1: Track dataset files
        )
        db_file_log = UploadedFileLog.model_validate(file_log_create)
        db.add(db_file_log)
        db.commit()
        db.refresh(db_file_log)

        return FileUploadResponse(
            id=db_file_log.id,  # Add the id field for frontend
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
            id=None,  # No ID since upload failed
            filename=file.filename,
            content_type=file.content_type,
            size=0, 
            message=f"Failed to save file '{file.filename}'. Error: {str(e)}",
            file_log_id=None,
            file_uuid=None
        )
    finally:
        await file.close() 

async def get_uploaded_files(db: Session = Depends(get_session)) -> List[dict]:
    """
    Retrieves all uploaded files from the database in the format expected by the frontend.
    """
    query = select(UploadedFileLog).order_by(UploadedFileLog.upload_timestamp.desc())
    result = db.execute(query)
    files = result.scalars().all()
    
    # Convert to frontend format
    return [
        {
            "id": str(file.id),
            "filename": file.filename,
            "file_type": _determine_file_type(file.content_type, file.filename),
            "size_bytes": file.size_bytes,
            "upload_timestamp": file.upload_timestamp.isoformat()
        }
        for file in files
    ]

def _determine_file_type(content_type: str, filename: str) -> str:
    """
    Determines the file type based on content type and filename.
    """
    if content_type == "application/pdf" or filename.endswith('.pdf'):
        return "pdf"
    elif content_type == "text/csv" or filename.endswith('.csv'):
        return "csv"
    elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                          "application/vnd.ms-excel"] or filename.endswith(('.xlsx', '.xls')):
        return "xlsx"
    elif content_type == "application/json" or filename.endswith('.json'):
        return "json"
    elif content_type.startswith("text/"):
        return "text"
    else:
        return "unknown" 

async def delete_uploaded_file(file_id: int, db: Session) -> Dict[str, any]:
    """
    Deletes an uploaded file and its associated data.
    
    This function:
    1. Removes the file from the filesystem
    2. Deletes the database record
    3. Deletes any related pipeline runs
    
    Args:
        file_id: The ID of the file to delete
        db: Database session
        
    Returns:
        Dict with success status and message
    """
    try:
        # Find the file record
        query = select(UploadedFileLog).where(UploadedFileLog.id == file_id)
        result = db.execute(query)
        file_record = result.scalar_one_or_none()
        
        if not file_record:
            return {"success": False, "message": f"File with ID {file_id} not found"}
        
        # Store filename for response message
        filename = file_record.filename
        file_path = file_record.storage_location
        
        # Delete related pipeline runs first (foreign key constraint)
        pipeline_query = select(PipelineRun).where(PipelineRun.uploaded_file_log_id == file_id)
        pipeline_result = db.execute(pipeline_query)
        pipeline_runs = pipeline_result.scalars().all()
        
        for pipeline_run in pipeline_runs:
            db.delete(pipeline_run)
        
        # Delete the file from filesystem
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Warning: Could not delete file {file_path}: {e}")
                # Continue with database deletion even if file removal fails
        
        # Delete the database record
        db.delete(file_record)
        db.commit()
        
        return {
            "success": True, 
            "message": f"File '{filename}' and {len(pipeline_runs)} related pipeline runs deleted successfully"
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error deleting file {file_id}: {e}")
        return {"success": False, "message": f"Error deleting file: {str(e)}"} 