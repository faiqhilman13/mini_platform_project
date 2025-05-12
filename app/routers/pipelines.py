# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session, select
from app.core.config import settings
from app.db.session import get_session
from app.models.pipeline_models import (
    PipelineRunStatusResponse,
    PipelineRunCreateResponse,
    PipelineType
)
from app.models.file_models import UploadedFileLog
from app.services import pipeline_service
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/pipelines",
    tags=["Pipelines"],
)

class TriggerPipelineRequest(BaseModel):
    uploaded_file_log_id: int
    pipeline_type: PipelineType
    # config: Optional[Dict[str, Any]] = None # Add later if needed

@router.post("/trigger")
def trigger_pipeline(
    request_body: TriggerPipelineRequest,
    db: Session = Depends(get_session)
) -> PipelineRunCreateResponse:
    """
    Triggers a pipeline execution for a given uploaded file.
    Currently supports PDF_SUMMARIZER which runs synchronously.
    """
    logger.info(f"Received request to trigger pipeline: {request_body.pipeline_type.value} for file ID: {request_body.uploaded_file_log_id}")

    # 1. Get UploadedFileLog record to find the file path
    uploaded_file = db.get(UploadedFileLog, request_body.uploaded_file_log_id)
    if not uploaded_file:
        logger.error(f"UploadedFileLog not found for ID: {request_body.uploaded_file_log_id}")
        raise HTTPException(status_code=404, detail=f"Uploaded file log with id {request_body.uploaded_file_log_id} not found.")

    if not uploaded_file.storage_location:
        logger.error(f"Storage location not set for UploadedFileLog ID: {request_body.uploaded_file_log_id}")
        raise HTTPException(status_code=400, detail="File storage location is missing.")

    # 2. Dispatch based on pipeline type
    if request_body.pipeline_type == PipelineType.PDF_SUMMARIZER:
        try:
            # Call the updated service function which now runs the flow synchronously
            response = pipeline_service.create_and_dispatch_summary_pipeline(
                db=db,
                uploaded_file_log_id=request_body.uploaded_file_log_id,
                file_path=uploaded_file.storage_location, # Pass the file path
                original_filename=uploaded_file.filename
            )
            logger.info(f"Synchronous PDF summarization flow initiated and completed for file ID: {request_body.uploaded_file_log_id}, run_uuid: {response.run_uuid}")
            return response
        except HTTPException as http_exc:
            # Re-raise HTTPExceptions from the service layer
            raise http_exc
        except Exception as e:
            logger.exception(f"Failed to trigger/run PDF summarizer pipeline for file ID {request_body.uploaded_file_log_id}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    else:
        logger.warning(f"Pipeline type '{request_body.pipeline_type.value}' not yet supported.")
        raise HTTPException(status_code=400, detail=f"Pipeline type '{request_body.pipeline_type.value}' not yet implemented.")

@router.get("/{run_uuid}/status", response_model=PipelineRunStatusResponse)
def get_pipeline_status(
    run_uuid: uuid.UUID,
    db: Session = Depends(get_session)
) -> PipelineRunStatusResponse:
    """
    Retrieves the status and results of a specific pipeline run.
    """
    logger.info(f"Received request for status of pipeline run UUID: {run_uuid}")
    try:
        status_response = pipeline_service.get_pipeline_run_status(db=db, run_uuid=run_uuid)
        if status_response is None:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        return status_response
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 404 Not Found) from the service layer
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving status for pipeline run UUID {run_uuid}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") 