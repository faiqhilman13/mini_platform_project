# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session
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
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/pipelines",
    tags=["Pipelines"],
)

class TriggerPipelineRequest(BaseModel):
    uploaded_file_log_id: int
    pipeline_type: PipelineType
    config: Optional[Dict[str, Any]] = None  # Now enabled to accept configuration data

@router.post("/trigger", response_model=PipelineRunCreateResponse)
def trigger_pipeline(
    request_body: TriggerPipelineRequest,
    db: Session = Depends(get_session)
) -> PipelineRunCreateResponse:
    """
    Triggers a pipeline execution for a given uploaded file and pipeline type.
    All supported pipelines run synchronously via Prefect flows.
    """
    logger.info(f"Received request to trigger pipeline: {request_body.pipeline_type.value} for file ID: {request_body.uploaded_file_log_id}")
    logger.info(f"DEBUGGING: Received config = {request_body.config}")

    try:
        # First check if the uploaded file log exists
        file_log = db.get(UploadedFileLog, request_body.uploaded_file_log_id)
        if not file_log:
            logger.error(f"Uploaded file log with id {request_body.uploaded_file_log_id} not found")
            raise HTTPException(status_code=404, detail=f"Uploaded file log with id {request_body.uploaded_file_log_id} not found")
            
        # Call the generalized service function
        response = pipeline_service.trigger_pipeline_flow(
            db=db,
            uploaded_file_log_id=request_body.uploaded_file_log_id,
            pipeline_type=request_body.pipeline_type,
            config=request_body.config
        )
        logger.info(f"Pipeline {request_body.pipeline_type.value} flow initiated for file ID: {request_body.uploaded_file_log_id}, run_uuid: {response.run_uuid}, status: {response.status}")
        return response
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., 404 for file not found, 400 for bad type, 500 from service)
        logger.error(f"HTTPException from service: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error triggering pipeline {request_body.pipeline_type.value} for file ID {request_body.uploaded_file_log_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

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
            logger.warning(f"Pipeline run UUID: {run_uuid} not found by service.")
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        return status_response
    except HTTPException as http_exc:
        raise http_exc # Re-raise if it's already an HTTPException
    except Exception as e:
        logger.exception(f"Error retrieving status for pipeline run UUID {run_uuid}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/runs", response_model=list)
def get_pipeline_runs(
    file_id: Optional[int] = None,
    db: Session = Depends(get_session)
) -> list:
    """
    Retrieves pipeline runs, optionally filtered by file_id.
    """
    logger.info(f"Received request for pipeline runs, file_id filter: {file_id}")
    try:
        runs = pipeline_service.get_pipeline_runs(db=db, file_id=file_id)
        return runs
    except Exception as e:
        logger.exception(f"Error retrieving pipeline runs")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") 