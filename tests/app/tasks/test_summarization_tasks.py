import pytest
from unittest.mock import patch, MagicMock, ANY
from uuid import uuid4, UUID as PyUUID
from datetime import datetime, timezone

from sqlmodel import Session, select # Keep select if used directly in tests

from app.tasks.summarization_tasks import summarize_pdf_task, update_status_on_prerun
from app.models.pipeline_models import PipelineRun, PipelineRunStatus, PipelineRunUpdate
from app.models.file_models import UploadedFileLog
from app.core.celery_app import celery_app # For task.name if needed

# --- Fixtures --- 
@pytest.fixture
def mock_db_session():
    session = MagicMock(spec=Session)
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.exec = MagicMock()
    return session

@pytest.fixture
def mock_pipeline_run_queued():
    run_uuid = uuid4()
    return PipelineRun(
        id=1,
        run_uuid=run_uuid,
        pipeline_name="pdf_summarizer",
        status=PipelineRunStatus.QUEUED,
        uploaded_file_log_id=10,
        celery_task_id="celery-id-123",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

@pytest.fixture
def mock_pipeline_run_processing(mock_pipeline_run_queued: PipelineRun):
    mock_pipeline_run_queued.status = PipelineRunStatus.PROCESSING
    return mock_pipeline_run_queued

# --- Tests for summarize_pdf_task --- 

@patch("app.tasks.summarization_tasks.Session", autospec=True)
@patch("app.tasks.summarization_tasks.do_summarize")
def test_summarize_pdf_task_success(
    mock_do_summarize: MagicMock, 
    MockSession: MagicMock, # Patched Session constructor
    mock_db_session: MagicMock, # Fixture for session instance
    mock_pipeline_run_processing: PipelineRun # Assumes prerun has set to PROCESSING
):
    """Test successful PDF summarization task execution."""
    MockSession.return_value.__enter__.return_value = mock_db_session # Handle context manager
    
    run_uuid_str = str(mock_pipeline_run_processing.run_uuid)
    file_log_id = mock_pipeline_run_processing.uploaded_file_log_id
    file_path = "/fake/path/to/doc.pdf"
    original_filename = "doc.pdf"

    # Mock DB query for PipelineRun
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = mock_pipeline_run_processing
    mock_db_session.exec.return_value = mock_query_result

    mock_do_summarize.return_value = {"summary": "This is a summary.", "status": "success"}

    # Mock the Celery task instance if needed for self.update_state (not used directly in this code path)
    mock_celery_task_instance = MagicMock()

    result = summarize_pdf_task.s(
        run_uuid_str, file_log_id, file_path, original_filename
    ).apply(task_id="test-task-id", instance=mock_celery_task_instance).get() # .get() to run synchronously

    # mock_db_session.exec.assert_called_once()
    # The prerun signal will also call exec, so we expect 2 calls if signals are active
    assert mock_db_session.exec.call_count == 2
    # Can add more specific check for the select statement if needed
    
    mock_do_summarize.assert_called_once_with(pdf_path=file_path, num_summary_sentences=3)
    
    # Check that db.add was called with the updated PipelineRun
    # The object passed to add will be the mock_pipeline_run_processing instance
    mock_db_session.add.assert_called()
    assert mock_pipeline_run_processing.status == PipelineRunStatus.COMPLETED
    assert mock_pipeline_run_processing.output_reference == "This is a summary."
    assert mock_pipeline_run_processing.error_message is None
    mock_db_session.commit.assert_called_once() # Should be committed once at the end

    assert result["status"] == "success"
    assert result["output"] == "This is a summary."

@patch("app.tasks.summarization_tasks.Session", autospec=True)
@patch("app.tasks.summarization_tasks.do_summarize")
def test_summarize_pdf_task_summarization_fails(
    mock_do_summarize: MagicMock, 
    MockSession: MagicMock,
    mock_db_session: MagicMock,
    mock_pipeline_run_processing: PipelineRun
):
    """Test task when do_summarize pipeline returns a failure status."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    run_uuid_str = str(mock_pipeline_run_processing.run_uuid)

    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = mock_pipeline_run_processing
    mock_db_session.exec.return_value = mock_query_result

    mock_do_summarize.return_value = {"summary": "", "status": "error: PDF content extraction failed"}

    result = summarize_pdf_task.s(run_uuid_str, 10, "/path", "file.pdf").apply().get()

    assert mock_pipeline_run_processing.status == PipelineRunStatus.FAILED
    assert "PDF content extraction failed" in mock_pipeline_run_processing.error_message
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called_once()
    assert result["status"] == "error: PDF content extraction failed"

@patch("app.tasks.summarization_tasks.Session", autospec=True)
def test_summarize_pdf_task_pipelinerun_not_found(
    MockSession: MagicMock, 
    mock_db_session: MagicMock
):
    """Test task when the PipelineRun record is not found in the DB."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    run_uuid_str = str(uuid4())

    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = None # Simulate PipelineRun not found
    mock_db_session.exec.return_value = mock_query_result

    result = summarize_pdf_task.s(run_uuid_str, 10, "/path", "file.pdf").apply().get()

    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()
    assert result["status"] == "error"
    assert result["message"] == "PipelineRun not found"

@patch("app.tasks.summarization_tasks.Session", autospec=True)
@patch("app.tasks.summarization_tasks.do_summarize")
def test_summarize_pdf_task_general_exception(
    mock_do_summarize: MagicMock, 
    MockSession: MagicMock,
    mock_db_session: MagicMock,
    mock_pipeline_run_processing: PipelineRun
):
    """Test task when a general exception occurs during processing."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    run_uuid_str = str(mock_pipeline_run_processing.run_uuid)

    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = mock_pipeline_run_processing
    mock_db_session.exec.return_value = mock_query_result

    mock_do_summarize.side_effect = Exception("Unexpected summarizer crash")

    with pytest.raises(Exception, match="Unexpected summarizer crash"):
        summarize_pdf_task.s(run_uuid_str, 10, "/path", "file.pdf").apply().get()

    # Check DB status was updated to FAILED
    assert mock_pipeline_run_processing.status == PipelineRunStatus.FAILED
    assert "Unexpected summarizer crash" in mock_pipeline_run_processing.error_message
    # Add may be called multiple times due to initial PROCESSING update + FAILED update
    assert mock_db_session.add.call_count >= 1
    assert mock_db_session.commit.call_count >= 1

# --- Tests for task_prerun_handler --- 

@patch("app.tasks.summarization_tasks.Session", autospec=True)
def test_update_status_on_prerun_success(
    MockSession: MagicMock,
    mock_db_session: MagicMock,
    mock_pipeline_run_queued: PipelineRun
):
    """Test task_prerun signal updates status from QUEUED to PROCESSING."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = mock_pipeline_run_queued
    mock_db_session.exec.return_value = mock_query_result

    mock_task = MagicMock()
    mock_task.name = 'app.tasks.summarization_tasks.summarize_pdf_task'
    run_uuid_str = str(mock_pipeline_run_queued.run_uuid)

    update_status_on_prerun(task=mock_task, args=[run_uuid_str])

    mock_db_session.exec.assert_called_once()
    assert mock_pipeline_run_queued.status == PipelineRunStatus.PROCESSING
    assert mock_pipeline_run_queued.updated_at is not None # Check it was touched
    mock_db_session.add.assert_called_once_with(mock_pipeline_run_queued)
    mock_db_session.commit.assert_called_once()

@patch("app.tasks.summarization_tasks.Session", autospec=True)
def test_update_status_on_prerun_already_processing(
    MockSession: MagicMock,
    mock_db_session: MagicMock,
    mock_pipeline_run_processing: PipelineRun # Status is already PROCESSING
):
    """Test prerun signal does not change status if already PROCESSING."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = mock_pipeline_run_processing
    mock_db_session.exec.return_value = mock_query_result

    original_updated_at = mock_pipeline_run_processing.updated_at

    mock_task = MagicMock()
    mock_task.name = 'app.tasks.summarization_tasks.summarize_pdf_task'
    run_uuid_str = str(mock_pipeline_run_processing.run_uuid)

    update_status_on_prerun(task=mock_task, args=[run_uuid_str])

    assert mock_pipeline_run_processing.status == PipelineRunStatus.PROCESSING
    assert mock_pipeline_run_processing.updated_at == original_updated_at # Not touched
    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()

@patch("app.tasks.summarization_tasks.Session", autospec=True)
def test_update_status_on_prerun_pipelinerun_not_found(
    MockSession: MagicMock,
    mock_db_session: MagicMock
):
    """Test prerun signal when PipelineRun is not found."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = None # Not found
    mock_db_session.exec.return_value = mock_query_result

    mock_task = MagicMock()
    mock_task.name = 'app.tasks.summarization_tasks.summarize_pdf_task'
    run_uuid_str = str(uuid4())

    update_status_on_prerun(task=mock_task, args=[run_uuid_str])

    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()

@patch("app.tasks.summarization_tasks.Session", autospec=True)
def test_update_status_on_prerun_wrong_task_name(
    MockSession: MagicMock,
    mock_db_session: MagicMock
):
    """Test prerun signal does nothing if task name does not match."""
    MockSession.return_value.__enter__.return_value = mock_db_session
    
    mock_task = MagicMock()
    mock_task.name = 'app.tasks.some_other_task' # Different task name

    update_status_on_prerun(task=mock_task, args=[str(uuid4())])

    mock_db_session.exec.assert_not_called()
    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()
