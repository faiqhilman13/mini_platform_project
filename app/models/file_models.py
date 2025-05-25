from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
import uuid
from datetime import datetime, timezone

# Forward reference for PipelineRun
# from .pipeline_models import PipelineRun # Assuming PipelineRun is in pipeline_models.py

class FileUploadResponse(SQLModel): # This is a Pydantic model for API response, not a table
    id: Optional[int] = None  # Frontend expects this field
    filename: str
    content_type: str | None = None
    size: int
    message: str = "File uploaded successfully"
    file_log_id: Optional[int] = None # To return the ID of the created log entry
    file_uuid: Optional[uuid.UUID] = None

class UploadedFileLogBase(SQLModel):
    filename: str
    content_type: Optional[str] = None
    size_bytes: int
    storage_location: str # For MVP, this will be the local path
    is_dataset: bool = False  # DS1.1.1: Track if file is a dataset (CSV/Excel)

class UploadedFileLog(UploadedFileLogBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_uuid: uuid.UUID = Field(default_factory=uuid.uuid4, index=True, nullable=False, unique=True)
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to PipelineRun (defined in pipeline_models.py)
    # pipeline_runs: List["PipelineRun"] = Relationship(back_populates="file_log")

class UploadedFileLogCreate(UploadedFileLogBase):
    pass

class UploadedFileLogRead(UploadedFileLogBase):
    id: int
    file_uuid: uuid.UUID
    upload_timestamp: datetime 