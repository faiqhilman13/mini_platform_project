# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session
from app.core.config import settings
from app.db.session import get_session
from app.models.pipeline_models import PipelineRunRead, PipelineType # Assuming PipelineType might be used in request
from app.services import pipeline_service as service
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/pipelines",
    tags=["Pipelines"],
)

class TriggerPipelineRequest(BaseModel):
    uploaded_file_log_id: int
    pipeline_type: PipelineType # For now, only PDF_SUMMARIZER is implemented via Celery
    # num_summary_sentences: Optional[int] = 3 # Could add params here

@router.post("/trigger", response_model=PipelineRunRead)
async def trigger_pipeline(
    request: TriggerPipelineRequest = Body(...),
    db: Session = Depends(get_session)
):
    """
    Triggers a processing pipeline for an uploaded file.
    For MVP, specifically triggers the PDF summarization pipeline.
    """
    logger.info(f"Received request to trigger pipeline: {request.model_dump()}")
    if request.pipeline_type == PipelineType.PDF_SUMMARIZER:
        try:
            pipeline_run = await service.create_and_dispatch_summary_pipeline(
                uploaded_file_log_id=request.uploaded_file_log_id,
                db=db
            )
            return pipeline_run
        except HTTPException as he:
            raise he # Re-raise known HTTP exceptions from service
        except Exception as e:
            logger.error(f"Error triggering summary pipeline: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error triggering pipeline: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Pipeline type '{request.pipeline_type}' not yet supported for async execution.")

@router.get("/{run_uuid}/status", response_model=PipelineRunRead)
async def get_status(
    run_uuid: uuid.UUID,
    db: Session = Depends(get_session)
):
    """
    Gets the status of a specific pipeline run.
    """
    logger.info(f"Received request for status of pipeline run_uuid: {run_uuid}")
    pipeline_run = await service.get_pipeline_run_status(run_uuid=run_uuid, db=db)
    if not pipeline_run:
        raise HTTPException(status_code=404, detail=f"Pipeline run with UUID {run_uuid} not found.")
    return pipeline_run 