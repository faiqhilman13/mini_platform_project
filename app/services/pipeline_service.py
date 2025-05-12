import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.pipeline_models import PipelineRun, PipelineType, PipelineRunStatus, PipelineRunCreateResponse, PipelineRunStatusResponse
from app.models.file_models import UploadedFileLog
from workflows.pipelines.summarizer import run_pdf_summary_pipeline # Import Prefect flow

logger = logging.getLogger(__name__)


def create_and_dispatch_summary_pipeline(
    db: Session,
    uploaded_file_log_id: int,
    file_path: str,
    original_filename: str
) -> PipelineRunCreateResponse:
    """
    Creates a PipelineRun record for a PDF summarization pipeline and runs the Prefect flow synchronously.

    Args:
        db (Session): Database session.
        uploaded_file_log_id (int): ID of the uploaded file log record.
        file_path (str): Absolute path to the uploaded file.
        original_filename (str): Original name of the uploaded file.

    Returns:
        PipelineRunCreateResponse: Response containing the run UUID and initial status.

    Raises:
        HTTPException: If the pipeline record cannot be created or the flow fails immediately.
    """
    logger.info(f"Initiating PDF summarization pipeline for file ID: {uploaded_file_log_id}")

    # Verify that the uploaded file log exists
    uploaded_file_log = db.get(UploadedFileLog, uploaded_file_log_id)
    if not uploaded_file_log:
        logger.error(f"UploadedFileLog with id {uploaded_file_log_id} not found")
        raise HTTPException(status_code=404, detail=f"UploadedFileLog with id {uploaded_file_log_id} not found")

    # 1. Create PipelineRun record
    pipeline_run = PipelineRun(
        uploaded_file_log_id=uploaded_file_log_id,
        pipeline_type=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.PENDING, # Start as PENDING
        # orchestrator_run_id will be null for sync run for now
    )
    try:
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)
        logger.info(f"Created PipelineRun record with UUID: {pipeline_run.run_uuid}")
    except Exception as e:
        db.rollback()
        logger.exception("Database error creating PipelineRun record.")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # 2. Run the Prefect flow synchronously
    try:
        # Update status to RUNNING before execution
        pipeline_run.status = PipelineRunStatus.RUNNING
        pipeline_run.updated_at = datetime.now(timezone.utc)
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)

        logger.info(f"Running Prefect flow 'run_pdf_summary_pipeline' for run UUID: {pipeline_run.run_uuid}")
        flow_result = run_pdf_summary_pipeline(pdf_path=file_path)
        logger.info(f"Prefect flow completed for run UUID: {pipeline_run.run_uuid} with status: {flow_result.get('status')}")

        # 3. Update PipelineRun record with flow result
        if flow_result.get("status") == "success":
            pipeline_run.status = PipelineRunStatus.COMPLETED
            pipeline_run.result = flow_result.get("summary")
            pipeline_run.error_message = None
        else:
            pipeline_run.status = PipelineRunStatus.FAILED
            pipeline_run.result = None
            pipeline_run.error_message = flow_result.get("message", "Flow failed without specific message.")

        pipeline_run.updated_at = datetime.now(timezone.utc)
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)

        return PipelineRunCreateResponse(
            run_uuid=pipeline_run.run_uuid,
            status=pipeline_run.status, # Return final status from sync run
            uploaded_file_log_id=pipeline_run.uploaded_file_log_id,
            message="PDF summarization flow executed synchronously."
        )

    except Exception as e:
        # Mark as FAILED if synchronous execution raises an exception
        db.rollback() # Rollback potential partial commits if exception occurs mid-process
        # Fetch the pipeline run again to update it, or create if the initial creation failed somehow
        pipeline_run_to_fail = db.get(PipelineRun, pipeline_run.run_uuid)
        if pipeline_run_to_fail:
            pipeline_run_to_fail.status = PipelineRunStatus.FAILED
            pipeline_run_to_fail.error_message = f"Error during synchronous flow execution: {e}"
            pipeline_run_to_fail.updated_at = datetime.now(timezone.utc)
            db.add(pipeline_run_to_fail)
            db.commit()

        logger.exception(f"Error running synchronous Prefect flow for run UUID: {pipeline_run.run_uuid}")
        # Raise HTTPException here because the initial trigger request failed
        raise HTTPException(status_code=500, detail=f"Failed to execute summarization flow: {e}")


def get_pipeline_run_status(db: Session, run_uuid: uuid.UUID) -> Optional[PipelineRunStatusResponse]:
    """
    Retrieves the status and result of a specific pipeline run.

    Args:
        db (Session): Database session.
        run_uuid (uuid.UUID): The UUID of the pipeline run.

    Returns:
        Optional[PipelineRunStatusResponse]: The status details of the pipeline run, or None if not found.
    """
    logger.debug(f"Fetching status for pipeline run UUID: {run_uuid}")
    
    # Use select().where() for querying by non-primary key
    statement = select(PipelineRun).where(PipelineRun.run_uuid == run_uuid)
    pipeline_run = db.exec(statement).one_or_none()

    if not pipeline_run:
        logger.warning(f"Pipeline run with UUID {run_uuid} not found.")
        # Return None instead of raising HTTPException
        return None

    logger.debug(f"Found pipeline run: {pipeline_run.run_uuid}, Status: {pipeline_run.status}")
    # Map the DB model to the response model
    return PipelineRunStatusResponse(
        run_uuid=pipeline_run.run_uuid,
        pipeline_type=pipeline_run.pipeline_type,
        status=pipeline_run.status,
        uploaded_file_log_id=pipeline_run.uploaded_file_log_id,
        result=pipeline_run.result,
        error_message=pipeline_run.error_message,
        created_at=pipeline_run.created_at,
        updated_at=pipeline_run.updated_at
    ) 