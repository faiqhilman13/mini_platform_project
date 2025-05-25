from sqlmodel import SQLModel, Field, Relationship, JSON, Column
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
from datetime import datetime, timezone

# Forward reference for UploadedFileLog if it's in a different module or defined later
# from .file_models import UploadedFileLog # Assuming UploadedFileLog will be in file_models.py

class PipelineRunStatus(str, Enum):
    PENDING = "PENDING" # Status before flow execution starts (might not be used for sync flows)
    RUNNING = "RUNNING" # Flow is actively running (might not be directly observed for sync flows)
    COMPLETED = "COMPLETED" # Flow finished successfully
    FAILED = "FAILED"      # Flow finished with an error
    CANCELLED = "CANCELLED"  # Flow was cancelled (if cancellation is implemented)

class PipelineType(str, Enum):
    PDF_SUMMARIZER = "PDF_SUMMARIZER"
    RAG_CHATBOT = "RAG_CHATBOT"
    TEXT_CLASSIFIER = "TEXT_CLASSIFIER"
    ML_TRAINING = "ML_TRAINING"  # DS1.2.1: Added for ML pipeline support

class PipelineRunBase(SQLModel):
    pipeline_type: PipelineType
    status: PipelineRunStatus = Field(default=PipelineRunStatus.PENDING)
    config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    result: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = Field(default=None)
    orchestrator_run_id: Optional[str] = Field(default=None, index=True) # Changed from celery_task_id
    # Foreign key to link to the uploaded file
    uploaded_file_log_id: int = Field(foreign_key="uploadedfilelog.id")

class PipelineRun(PipelineRunBase, table=True):
    run_uuid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)}
    )
    # Relationship (Optional, can be added later if needed for ORM features)
    # uploaded_file: Optional["UploadedFileLog"] = Relationship(back_populates="pipeline_runs")

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
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Response model for triggering a pipeline run
class PipelineRunCreateResponse(SQLModel):
    run_uuid: uuid.UUID
    status: PipelineRunStatus
    uploaded_file_log_id: int
    message: str # Added message field

# Response model for getting pipeline run status
class PipelineRunStatusResponse(SQLModel):
    run_uuid: uuid.UUID
    pipeline_type: PipelineType
    status: PipelineRunStatus
    uploaded_file_log_id: int
    result: Optional[Dict[str, Any]] = None # Changed back to List[str]
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    orchestrator_run_id: Optional[str] = None # Changed from celery_task_id 