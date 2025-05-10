from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from enum import Enum
import uuid
from datetime import datetime

# Forward reference for UploadedFileLog if it's in a different module or defined later
# from .file_models import UploadedFileLog # Assuming UploadedFileLog will be in file_models.py

class PipelineRunStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class PipelineType(str, Enum):
    PDF_SUMMARIZER = "PDF_SUMMARIZER"
    RAG_CHATBOT = "RAG_CHATBOT"
    TEXT_CLASSIFIER = "TEXT_CLASSIFIER"

class PipelineRunBase(SQLModel):
    pipeline_name: PipelineType
    status: PipelineRunStatus = PipelineRunStatus.QUEUED
    output_reference: Optional[str] = None
    error_message: Optional[str] = None
    # Foreign key to the UploadedFileLog table
    uploaded_file_log_id: Optional[int] = Field(default=None, foreign_key="uploadedfilelog.id") # lowercase table name

class PipelineRun(PipelineRunBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_uuid: uuid.UUID = Field(default_factory=uuid.uuid4, index=True, nullable=False, unique=True)
    celery_task_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False) # Need to auto-update this

    # Relationship to UploadedFileLog (defined in file_models.py)
    # file_log: Optional["UploadedFileLog"] = Relationship(back_populates="pipeline_runs")

class PipelineRunCreate(PipelineRunBase):
    pass

class PipelineRunRead(PipelineRunBase):
    id: int
    run_uuid: uuid.UUID
    celery_task_id: Optional[str]
    created_at: datetime
    updated_at: datetime

class PipelineRunUpdate(SQLModel):
    status: Optional[PipelineRunStatus] = None
    output_reference: Optional[str] = None
    error_message: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow) 