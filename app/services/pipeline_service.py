import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import os
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.pipeline_models import PipelineRun, PipelineType, PipelineRunStatus, PipelineRunCreateResponse, PipelineRunStatusResponse
from app.models.file_models import UploadedFileLog
from app.core.config import settings

# Import pipeline workflows
from workflows.pipelines.summarizer import run_pdf_summary_pipeline
from workflows.pipelines.rag_chatbot import process_document_rag_flow
from workflows.pipelines.ml_training import ml_training_flow
from workflows.pipelines.rag_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)


def trigger_pipeline_flow(
    db: Session,
    uploaded_file_log_id: int,
    pipeline_type: PipelineType,
    config: Optional[Dict[str, Any]] = None
) -> PipelineRunCreateResponse:
    """
    Creates a PipelineRun record and runs the specified Prefect flow synchronously.

    Args:
        db (Session): Database session.
        uploaded_file_log_id (int): ID of the uploaded file log record.
        pipeline_type (PipelineType): The type of pipeline to run.
        config (Optional[Dict[str, Any]]): Configuration for the pipeline.

    Returns:
        PipelineRunCreateResponse: Response containing the run UUID and initial status.

    Raises:
        HTTPException: If the pipeline record cannot be created or the flow fails.
    """
    logger.info(f"Initiating pipeline type: {pipeline_type} for file ID: {uploaded_file_log_id}")

    uploaded_file_log = db.get(UploadedFileLog, uploaded_file_log_id)
    if not uploaded_file_log:
        logger.error(f"UploadedFileLog with id {uploaded_file_log_id} not found")
        raise HTTPException(status_code=404, detail=f"UploadedFileLog with id {uploaded_file_log_id} not found")

    file_path = uploaded_file_log.storage_location
    original_filename = uploaded_file_log.filename

    pipeline_run = PipelineRun(
        uploaded_file_log_id=uploaded_file_log_id,
        pipeline_type=pipeline_type,
        status=PipelineRunStatus.PENDING,
        config=config
    )
    try:
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)
        logger.info(f"Created PipelineRun record with UUID: {pipeline_run.run_uuid} for {pipeline_type}")
    except Exception as e:
        db.rollback()
        logger.exception("Database error creating PipelineRun record.")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    flow_result: Optional[Dict[str, Any]] = None
    try:
        pipeline_run.status = PipelineRunStatus.RUNNING
        pipeline_run.updated_at = datetime.now(timezone.utc)
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)

        logger.info(f"Running Prefect flow for {pipeline_type}, run UUID: {pipeline_run.run_uuid}")

        if pipeline_type == PipelineType.PDF_SUMMARIZER:
            flow_result = run_pdf_summary_pipeline(pdf_path=file_path)
        elif pipeline_type == PipelineType.RAG_CHATBOT:
            flow_result = process_document_rag_flow(pdf_path=file_path, title=original_filename)
        elif pipeline_type == PipelineType.ML_TRAINING:
            logger.info(f"Running ML training for {file_path}")
            # For ML training, we need to extract config from pipeline_run config
            # or create a default config if none provided
            ml_config = pipeline_run.config or {}
            
            logger.info(f"DEBUGGING: Raw pipeline_run.config = {pipeline_run.config}")
            logger.info(f"DEBUGGING: ml_config keys = {list(ml_config.keys()) if ml_config else 'None'}")
            logger.info(f"DEBUGGING: Full ml_config = {ml_config}")
            
            # Extract target column from multiple possible keys
            target_column = (
                ml_config.get("target_variable") or 
                ml_config.get("target_column") or 
                ml_config.get("target") or 
                ml_config.get("targetColumn") or
                ml_config.get("config", {}).get("target_variable") or
                ml_config.get("config", {}).get("target_column") or
                ml_config.get("config", {}).get("target") or
                "target"  # fallback
            )
            
            logger.info(f"DEBUGGING: Extracted target column: '{target_column}' from config")
            logger.info(f"DEBUGGING: target_variable = {ml_config.get('target_variable')}")
            logger.info(f"DEBUGGING: target_column = {ml_config.get('target_column')}")
            logger.info(f"DEBUGGING: target = {ml_config.get('target')}")
            
            # Create ML training config
            training_config = {
                "file_path": file_path,
                "target_column": target_column,
                "problem_type": ml_config.get("problem_type", "classification"),
                "algorithms": ml_config.get("algorithms", [
                    {"name": "logistic_regression", "hyperparameters": {}},
                    {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}}
                ]),
                "preprocessing_config": ml_config.get("preprocessing_config", {}),
                "pipeline_run_id": str(pipeline_run.run_uuid)
            }
            
            logger.info(f"ML training config: {training_config}")
            try:
                flow_result = ml_training_flow(training_config)
                logger.info(f"ML training flow returned: {type(flow_result)}")
                logger.info(f"ML training flow result keys: {list(flow_result.keys()) if isinstance(flow_result, dict) else 'Not a dict'}")
                if isinstance(flow_result, dict):
                    logger.info(f"ML training flow success: {flow_result.get('success')}")
                    logger.info(f"ML training flow error: {flow_result.get('error')}")
            except Exception as ml_error:
                logger.error(f"Exception in ML training flow: {str(ml_error)}")
                flow_result = {"success": False, "error": str(ml_error)}
        else:
            logger.error(f"Unsupported pipeline type: {pipeline_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported pipeline type: {pipeline_type}")
        
        logger.info(f"Prefect flow {pipeline_type} completed for run UUID: {pipeline_run.run_uuid} with result: {flow_result}")

        if flow_result and (flow_result.get("status") == "success" or flow_result.get("success") == True):
            pipeline_run.status = PipelineRunStatus.COMPLETED
            # Store execution result
            if isinstance(flow_result, dict):
                # Convert MLTrainingResult to dict if needed
                if 'result' in flow_result and hasattr(flow_result['result'], 'to_dict'):
                    # Convert the result to a fully serializable dictionary
                    serializable_result = flow_result['result'].to_dict()
                    flow_result['result'] = serializable_result
                    logger.info(f"Converted MLTrainingResult to serializable format")
                
                # Ensure all nested objects are also serializable
                pipeline_run.result = flow_result
                logger.info(f"Stored pipeline result successfully")
            else:
                pipeline_run.result = {"success": False, "error": "Unknown result format"}
            pipeline_run.error_message = None
        else:
            pipeline_run.status = PipelineRunStatus.FAILED
            pipeline_run.result = None
            # Extract more specific error information
            if flow_result:
                error_msg = flow_result.get("error", flow_result.get("message", "Flow failed without specific message."))
                # If flow_result has detailed error information, use it
                if isinstance(flow_result, dict) and "result" in flow_result:
                    result_obj = flow_result["result"]
                    if hasattr(result_obj, 'summary') and 'error' in getattr(result_obj, 'summary', {}):
                        error_msg = result_obj.summary['error']
                pipeline_run.error_message = error_msg
            else:
                pipeline_run.error_message = "Flow execution error or no result."

        pipeline_run.updated_at = datetime.now(timezone.utc)
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)

        return PipelineRunCreateResponse(
            run_uuid=pipeline_run.run_uuid,
            status=pipeline_run.status,
            uploaded_file_log_id=pipeline_run.uploaded_file_log_id,
            message=f"{pipeline_type.value} flow executed synchronously."
        )

    except HTTPException as http_exc:
        db.rollback()
        pipeline_run_to_fail = db.get(PipelineRun, pipeline_run.run_uuid)
        if pipeline_run_to_fail:
            pipeline_run_to_fail.status = PipelineRunStatus.FAILED
            pipeline_run_to_fail.error_message = getattr(http_exc, 'detail', str(http_exc))
            pipeline_run_to_fail.updated_at = datetime.now(timezone.utc)
            db.add(pipeline_run_to_fail)
            db.commit()
        logger.warning(f"HTTPException during {pipeline_type} flow execution for run UUID {pipeline_run.run_uuid}: {http_exc.detail}")
        raise http_exc

    except Exception as e:
        db.rollback()
        pipeline_run_to_fail = db.get(PipelineRun, pipeline_run.run_uuid)
        if pipeline_run_to_fail:
            pipeline_run_to_fail.status = PipelineRunStatus.FAILED
            pipeline_run_to_fail.error_message = f"Error during {pipeline_type.value} flow execution: {str(e)}"
            pipeline_run_to_fail.updated_at = datetime.now(timezone.utc)
            db.add(pipeline_run_to_fail)
            db.commit()

        logger.exception(f"Error running synchronous Prefect flow for {pipeline_type}, run UUID: {pipeline_run.run_uuid}")
        raise HTTPException(status_code=500, detail=f"Failed to execute {pipeline_type.value} flow: {str(e)}")


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
    
    statement = select(PipelineRun).where(PipelineRun.run_uuid == run_uuid)
    pipeline_run = db.exec(statement).one_or_none()

    if not pipeline_run:
        logger.warning(f"Pipeline run with UUID {run_uuid} not found.")
        return None

    logger.debug(f"Found pipeline run: {pipeline_run.run_uuid}, Status: {pipeline_run.status}")
    return PipelineRunStatusResponse(
        run_uuid=pipeline_run.run_uuid,
        pipeline_type=pipeline_run.pipeline_type,
        status=pipeline_run.status,
        uploaded_file_log_id=pipeline_run.uploaded_file_log_id,
        result=pipeline_run.result,
        error_message=pipeline_run.error_message,
        created_at=pipeline_run.created_at,
        updated_at=pipeline_run.updated_at,
        orchestrator_run_id=pipeline_run.orchestrator_run_id
    )


def get_pipeline_runs(db: Session, file_id: Optional[int] = None) -> list:
    """
    Retrieves pipeline runs, optionally filtered by file_id.

    Args:
        db (Session): Database session.
        file_id (Optional[int]): Optional file ID to filter runs.

    Returns:
        list: List of pipeline run status responses.
    """
    logger.debug(f"Fetching pipeline runs, file_id filter: {file_id}")
    
    statement = select(PipelineRun)
    if file_id:
        statement = statement.where(PipelineRun.uploaded_file_log_id == file_id)
    
    # Order by most recent first
    statement = statement.order_by(PipelineRun.created_at.desc())
    
    pipeline_runs = db.exec(statement).all()
    
    logger.debug(f"Found {len(pipeline_runs)} pipeline runs")
    
    return [
        PipelineRunStatusResponse(
            run_uuid=run.run_uuid,
            pipeline_type=run.pipeline_type,
            status=run.status,
            uploaded_file_log_id=run.uploaded_file_log_id,
            result=run.result,
            error_message=run.error_message,
            created_at=run.created_at,
            updated_at=run.updated_at,
            orchestrator_run_id=run.orchestrator_run_id
        )
        for run in pipeline_runs
    ] 