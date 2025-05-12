import logging
import os
from datetime import datetime, timezone
from uuid import UUID as PyUUID # Added import and alias
from sqlmodel import Session, select
from app.core.celery_app import celery_app
from app.db.session import engine # Import engine directly for session creation in task
from app.models.pipeline_models import PipelineRun, PipelineRunStatus, PipelineRunUpdate
from app.models.file_models import UploadedFileLog # <--- ADDED IMPORT
from workflows.pipelines.summarizer import run_pdf_summary_pipeline as do_summarize
from celery.signals import task_prerun, task_postrun # For status updates
from celery import current_task # To get task_id

logger = logging.getLogger(__name__)

# Ensure engine is initialized (it should be by main app, but good for standalone task debugging)
# if not engine._engine:  # A bit of a hack to check if engine is initialized
#     logger.warning("Database engine not initialized directly in task. Relying on main app setup.")

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def summarize_pdf_task(self, run_uuid_str: str, uploaded_file_log_id: int, file_path: str, original_filename: str):
    """
    Celery task to summarize a PDF. 
    Updates PipelineRun status and stores results/errors.
    """
    logger.info(f"Starting summarize_pdf_task for run_uuid: {run_uuid_str}, file_log_id: {uploaded_file_log_id}")
    db_pipeline_run = None
    
    run_uuid_obj = PyUUID(run_uuid_str) # Convert string to UUID object

    with Session(engine) as db_session:
        try:
            # Fetch the PipelineRun record
            statement = select(PipelineRun).where(PipelineRun.run_uuid == run_uuid_obj) # Use UUID object
            db_pipeline_run = db_session.exec(statement).one_or_none()

            if not db_pipeline_run:
                logger.error(f"PipelineRun with UUID {run_uuid_str} not found.")
                # self.update_state(state='FAILURE', meta={'exc_type': 'NotFoundError', 'exc_message': 'PipelineRun not found'})
                # Depending on how you want to handle this, you might raise an error or just log and exit
                return {"status": "error", "message": "PipelineRun not found"}

            # Update status to PROCESSING (if not already updated by task_prerun signal)
            if db_pipeline_run.status != PipelineRunStatus.PROCESSING:
                db_pipeline_run.status = PipelineRunStatus.PROCESSING
                db_pipeline_run.updated_at = datetime.now(timezone.utc)
                db_session.add(db_pipeline_run)
                db_session.commit()
                db_session.refresh(db_pipeline_run)
                logger.info(f"PipelineRun {run_uuid_str} status updated to PROCESSING.")

            # Perform the summarization
            # Note: do_summarize returns a dict like {"summary": "...", "status": "success"}
            summary_result = do_summarize(pdf_path=file_path, num_summary_sentences=3)

            # Update PipelineRun with results
            if summary_result["status"] == "success":
                update_data = PipelineRunUpdate(
                    status=PipelineRunStatus.COMPLETED,
                    output_reference=summary_result["summary"], # Storing summary directly for now
                    error_message=None
                )
                logger.info(f"Summarization successful for {run_uuid_str}. Summary length: {len(summary_result['summary'])} chars.")
            else:
                update_data = PipelineRunUpdate(
                    status=PipelineRunStatus.FAILED,
                    error_message=summary_result.get("status", "Summarization failed without specific error.") # Use status as error
                )
                logger.error(f"Summarization failed for {run_uuid_str}: {update_data.error_message}")
            
            # Apply updates to the model instance
            for key, value in update_data.model_dump(exclude_unset=True).items():
                setattr(db_pipeline_run, key, value)
            db_pipeline_run.updated_at = datetime.now(timezone.utc) # Ensure updated_at is always set
            
            db_session.add(db_pipeline_run)
            db_session.commit()
            # db_session.refresh(db_pipeline_run) # Refresh might not be needed if we return from here

            return {"status": summary_result["status"], "run_uuid": str(run_uuid_str), "output": summary_result.get("summary")}
        
        except Exception as e:
            logger.exception(f"Exception in summarize_pdf_task for {run_uuid_str}: {e}")
            # Attempt to update DB status to FAILED if possible
            if 'db_pipeline_run' in locals() and db_pipeline_run:
                try:
                    db_pipeline_run.status = PipelineRunStatus.FAILED
                    db_pipeline_run.error_message = str(e)
                    db_pipeline_run.updated_at = datetime.now(timezone.utc)
                    db_session.add(db_pipeline_run)
                    db_session.commit()
                except Exception as db_exc:
                    logger.error(f"Failed to update PipelineRun status to FAILED for {run_uuid_str} after exception: {db_exc}")
            # Let Celery know about the failure
            # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
            raise # Re-raise the exception for Celery to mark as failure

# Celery Signals for automatic status updates (optional but good practice)
@task_prerun.connect
def update_status_on_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **extras):
    # args = positional arguments the task is called with.
    # kwargs = keyword arguments the task is called with.
    run_uuid_str = None
    if task.name == 'app.tasks.summarization_tasks.summarize_pdf_task':
        if args and len(args) > 0:
            run_uuid_str = args[0]  # First positional arg is run_uuid_str
        elif kwargs: # Fallback if task was called with keyword args
            run_uuid_str = kwargs.get('run_uuid_str')
        
        if not run_uuid_str:
            logger.warning(f"Task prerun: run_uuid_str not extracted. Signal args: {args}, Signal kwargs: {kwargs}. Task ID: {task_id}")
            return
            
        logger.info(f"Task prerun: Attempting to update status for run_uuid {run_uuid_str} to PROCESSING. Task ID: {task_id}")
        
        run_uuid_obj = PyUUID(run_uuid_str)

        with Session(engine) as session:
            try:
                statement = select(PipelineRun).where(PipelineRun.run_uuid == run_uuid_obj) # Use UUID object
                db_pipeline_run = session.exec(statement).one_or_none()

                if db_pipeline_run:
                    if db_pipeline_run.status == PipelineRunStatus.QUEUED: # Only update if it's still queued
                        db_pipeline_run.status = PipelineRunStatus.PROCESSING
                        db_pipeline_run.updated_at = datetime.now(timezone.utc)
                        session.add(db_pipeline_run)
                        session.commit()
                        logger.info(f"Task prerun: Status updated to PROCESSING for run_uuid {run_uuid_str}")
                    else:
                        logger.info(f"Task prerun: Status for run_uuid {run_uuid_str} is already {db_pipeline_run.status}, not updating.")
                else:
                    logger.warning(f"Task prerun: PipelineRun not found for run_uuid {run_uuid_str}. Cannot update status.")
            except Exception as e:
                logger.error(f"Task prerun: Error updating status for run_uuid {run_uuid_str}: {e}", exc_info=True)
                # Not re-raising, allow task to proceed and potentially fail with more context
    elif task.name == 'app.tasks.summarization_tasks.summarize_pdf_task': # If args/kwargs were empty but task name matches
        logger.warning(f"Task prerun: No directly passed args or kwargs for task {task.name}, but task name matched. Signal args: {args}, Signal kwargs: {kwargs}. Task ID: {task_id}")

@task_postrun.connect
def log_task_completion(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
    # task_postrun can be used for cleanup or final logging if needed, but main logic is in task itself.
    pass # Added pass to fix indentation error 