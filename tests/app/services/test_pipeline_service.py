import pytest
# import pytest_asyncio # No longer needed
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Session, select
from fastapi import HTTPException

from app.services.pipeline_service import (
    trigger_pipeline_flow,
    get_pipeline_run_status
)
from app.models.pipeline_models import (
    PipelineRun, PipelineRunStatus, PipelineType, PipelineRunCreateResponse, PipelineRunStatusResponse
)
from app.models.file_models import UploadedFileLog

# Reusable mock for db session
@pytest.fixture
def mock_db_session():
    session = MagicMock(spec=Session)
    # Store for added/updated objects to simulate commit and refresh behavior
    session._object_store = {}

    def add_side_effect(obj):
        # Simulate adding to a session, store by a unique key if possible
        if hasattr(obj, 'run_uuid'):
            session._object_store[obj.run_uuid] = obj
        elif hasattr(obj, 'id') and obj.id:
            session._object_store[obj.id] = obj
        # else: could add a generic list for objects without clear PKs if needed
        # print(f"Mock DB Add: {obj}")

    def commit_side_effect():
        # print(f"Mock DB Commit called. Current store: {session._object_store}")
        pass # Commit doesn't do much in mock other than being callable

    def refresh_side_effect(obj):
        # Simulate refreshing state from DB (no actual change in mock)
        # print(f"Mock DB Refresh: {obj}")
        pass

    def get_side_effect(model_cls, pk):
        # print(f"Mock DB Get: {model_cls.__name__} with PK {pk}")
        # Simulate fetching from our object store
        if model_cls == UploadedFileLog and isinstance(pk, int):
            # Assuming UploadedFileLog is added to store by its id
            return session._object_store.get(pk)
        if model_cls == PipelineRun and isinstance(pk, uuid.UUID):
            return session._object_store.get(pk)
        # Fallback for other potential gets or if key not found
        # print(f"Mock DB Get: Not found or type mismatch for {model_cls.__name__} with {pk}")
        return None

    session.add = MagicMock(side_effect=add_side_effect)
    session.commit = MagicMock(side_effect=commit_side_effect)
    session.refresh = MagicMock(side_effect=refresh_side_effect)
    session.get = MagicMock(side_effect=get_side_effect)
    session.exec = MagicMock() # For other query types if used by service
    return session

@pytest.fixture
def mock_uploaded_file_log():
    return UploadedFileLog(
        id=1,
        file_uuid=uuid.uuid4(),
        filename="test.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        storage_location="./uploaded_files/test.pdf", # Critical for the service
        upload_timestamp=datetime.now(timezone.utc)
    )

@pytest.fixture
def mock_pipeline_run_summarizer(mock_uploaded_file_log: UploadedFileLog):
    return PipelineRun(
        run_uuid=uuid.uuid4(),
        pipeline_type=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.PENDING,
        uploaded_file_log_id=mock_uploaded_file_log.id,
    )

# --- Tests for PDF_SUMMARIZER --- 

def test_trigger_pipeline_flow_summary_success(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test successful sync execution of a PDF_SUMMARIZER pipeline."""
    mock_flow_result = {"status": "success", "summary": ["This is the summary."]}
    
    # Ensure UploadedFileLog is in the mock session's store for .get()
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log
    
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline", return_value=mock_flow_result) as mock_summarize_flow:
        response = trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            pipeline_type=PipelineType.PDF_SUMMARIZER
        )

    assert response is not None
    assert isinstance(response, PipelineRunCreateResponse)
    assert response.run_uuid is not None
    assert response.status == PipelineRunStatus.COMPLETED
    assert "PDF_SUMMARIZER flow executed synchronously" in response.message

    # Verify interactions with DB
    # Initial PipelineRun creation (add, commit, refresh)
    # Status update to RUNNING (add, commit, refresh)
    # Status update to COMPLETED (add, commit, refresh)
    assert mock_db_session.add.call_count == 3
    assert mock_db_session.commit.call_count == 3
    assert mock_db_session.refresh.call_count == 3 

    # Verify the PipelineRun object state in our mock store
    final_run_state = mock_db_session._object_store.get(response.run_uuid)
    assert final_run_state is not None
    assert final_run_state.status == PipelineRunStatus.COMPLETED
    assert final_run_state.pipeline_type == PipelineType.PDF_SUMMARIZER
    assert final_run_state.result == mock_flow_result # Entire dict is stored
    assert final_run_state.error_message is None

    mock_summarize_flow.assert_called_once_with(pdf_path=mock_uploaded_file_log.storage_location)

def test_trigger_pipeline_flow_summary_failure_modes(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test PDF_SUMMARIZER pipeline failures: flow raises exception or returns error status."""
    # Ensure UploadedFileLog is in the mock session's store
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log

    # Case 1: Flow function raises an exception
    flow_exception = Exception("Summarizer flow processing error")
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline", side_effect=flow_exception) as mock_flow_exc:
        with pytest.raises(HTTPException) as exc_info_case1:
            trigger_pipeline_flow(
                db=mock_db_session,
                uploaded_file_log_id=mock_uploaded_file_log.id,
                pipeline_type=PipelineType.PDF_SUMMARIZER
            )
    
    assert exc_info_case1.value.status_code == 500
    assert str(flow_exception) in exc_info_case1.value.detail
    mock_flow_exc.assert_called_once()
    # Verify PipelineRun status is FAILED in the mock store
    # Need to find the run_uuid. Since it's raised, response isn't returned.
    # We can iterate through the store or capture the run_uuid if the test setup allows.
    # For simplicity, let's assume one run was attempted and is in the store.
    failed_run_uuid_case1 = next((uid for uid, obj in mock_db_session._object_store.items() if isinstance(obj, PipelineRun) and obj.pipeline_type == PipelineType.PDF_SUMMARIZER), None)
    assert failed_run_uuid_case1 is not None
    failed_run_state_case1 = mock_db_session._object_store[failed_run_uuid_case1]
    assert failed_run_state_case1.status == PipelineRunStatus.FAILED
    assert str(flow_exception) in failed_run_state_case1.error_message

    # Reset for Case 2 - Clear relevant parts of the mock store or re-initialize mock_db_session if simpler
    mock_db_session._object_store.clear() # Clear store for next scenario
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log # Re-add needed log
    mock_db_session.reset_mock() # Reset call counts etc.

    # Case 2: Flow function returns error status
    mock_flow_error_result = {"status": "error", "message": "Specific flow error for summarizer"}
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline", return_value=mock_flow_error_result) as mock_flow_err:
        response_err = trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            pipeline_type=PipelineType.PDF_SUMMARIZER
        )
    
    assert response_err.status == PipelineRunStatus.FAILED
    final_run_state_err = mock_db_session._object_store.get(response_err.run_uuid)
    assert final_run_state_err is not None
    assert final_run_state_err.status == PipelineRunStatus.FAILED
    assert final_run_state_err.result is None
    assert mock_flow_error_result["message"] in final_run_state_err.error_message
    mock_flow_err.assert_called_once()

def test_trigger_pipeline_flow_file_log_not_found(mock_db_session: MagicMock):
    """Test pipeline trigger when UploadedFileLog is not found."""
    mock_db_session.get.return_value = None # Simulate UploadedFileLog not found
    
    with pytest.raises(HTTPException) as exc_info:
        trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=999, # Non-existent ID
            pipeline_type=PipelineType.PDF_SUMMARIZER
        )
    assert exc_info.value.status_code == 404
    assert "UploadedFileLog with id 999 not found" in exc_info.value.detail
    # Ensure the get mock was called for UploadedFileLog
    mock_db_session.get.assert_called_with(UploadedFileLog, 999)


# --- Tests for RAG_CHATBOT (Document Ingestion) --- 

def test_trigger_pipeline_flow_rag_ingestion_success(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test successful sync execution of a RAG_CHATBOT (ingestion) pipeline."""
    mock_rag_flow_result = {
        "status": "success", 
        "message": "PDF processed successfully. Created 10 chunks.",
        "doc_id": str(uuid.uuid4()),
        "chunks": 10
    }
    
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log
    
    with patch("app.services.pipeline_service.process_document_rag_flow", return_value=mock_rag_flow_result) as mock_rag_ingest_flow:
        response = trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            pipeline_type=PipelineType.RAG_CHATBOT
        )

    assert response is not None
    assert isinstance(response, PipelineRunCreateResponse)
    assert response.run_uuid is not None
    assert response.status == PipelineRunStatus.COMPLETED
    assert "RAG_CHATBOT flow executed synchronously" in response.message

    final_run_state = mock_db_session._object_store.get(response.run_uuid)
    assert final_run_state is not None
    assert final_run_state.status == PipelineRunStatus.COMPLETED
    assert final_run_state.pipeline_type == PipelineType.RAG_CHATBOT
    assert final_run_state.result == mock_rag_flow_result
    assert final_run_state.error_message is None

    mock_rag_ingest_flow.assert_called_once_with(
        pdf_path=mock_uploaded_file_log.storage_location,
        title=mock_uploaded_file_log.filename
    )

def test_trigger_pipeline_flow_rag_ingestion_failure_modes(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test RAG_CHATBOT (ingestion) pipeline failures: flow raises exception or returns error status."""
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log

    # Case 1: RAG flow function raises an exception
    rag_flow_exception = Exception("RAG ingestion flow processing error")
    with patch("app.services.pipeline_service.process_document_rag_flow", side_effect=rag_flow_exception) as mock_rag_flow_exc:
        with pytest.raises(HTTPException) as exc_info_case1:
            trigger_pipeline_flow(
                db=mock_db_session,
                uploaded_file_log_id=mock_uploaded_file_log.id,
                pipeline_type=PipelineType.RAG_CHATBOT
            )
    
    assert exc_info_case1.value.status_code == 500
    assert str(rag_flow_exception) in exc_info_case1.value.detail
    mock_rag_flow_exc.assert_called_once()
    failed_run_uuid_case1 = next((uid for uid, obj in mock_db_session._object_store.items() if isinstance(obj, PipelineRun) and obj.pipeline_type == PipelineType.RAG_CHATBOT), None)
    assert failed_run_uuid_case1 is not None
    failed_run_state_case1 = mock_db_session._object_store[failed_run_uuid_case1]
    assert failed_run_state_case1.status == PipelineRunStatus.FAILED
    assert str(rag_flow_exception) in failed_run_state_case1.error_message

    # Reset for Case 2
    mock_db_session._object_store.clear()
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log
    mock_db_session.reset_mock()

    # Case 2: RAG flow function returns error status
    mock_rag_flow_error_result = {"status": "error", "message": "Specific RAG ingestion error"}
    with patch("app.services.pipeline_service.process_document_rag_flow", return_value=mock_rag_flow_error_result) as mock_rag_flow_err:
        response_err = trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            pipeline_type=PipelineType.RAG_CHATBOT
        )
    
    assert response_err.status == PipelineRunStatus.FAILED
    final_run_state_err = mock_db_session._object_store.get(response_err.run_uuid)
    assert final_run_state_err is not None
    assert final_run_state_err.status == PipelineRunStatus.FAILED
    assert final_run_state_err.result is None
    assert mock_rag_flow_error_result["message"] in final_run_state_err.error_message
    mock_rag_flow_err.assert_called_once()

# --- Test for Unsupported Pipeline Type ---

def test_trigger_pipeline_flow_unsupported_type(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test triggering with an unsupported pipeline type."""
    mock_db_session._object_store[mock_uploaded_file_log.id] = mock_uploaded_file_log

    class MockUnsupportedPipelineType(str, Enum):
        SOME_NEW_TYPE = "SOME_NEW_TYPE"

    with pytest.raises(HTTPException) as exc_info:
        trigger_pipeline_flow(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            pipeline_type=MockUnsupportedPipelineType.SOME_NEW_TYPE # type: ignore
        )
    assert exc_info.value.status_code == 400
    assert f"Unsupported pipeline type: {MockUnsupportedPipelineType.SOME_NEW_TYPE}" in exc_info.value.detail

# --- Tests for get_pipeline_run_status --- (Keep existing tests, ensure they are compatible)

def test_get_pipeline_run_status_found(
    mock_db_session: MagicMock,
    # mock_pipeline_run: PipelineRun # Using a fixture for PipelineRun can be complex with mock_db_session
):
    """Test retrieving status for an existing pipeline run."""
    run_uuid_val = uuid.uuid4()
    # Create a mock PipelineRun object directly that would be in the store
    expected_run = PipelineRun(
        run_uuid=run_uuid_val,
        pipeline_type=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.COMPLETED,
        uploaded_file_log_id=1,
        result={"summary": ["Test summary"]},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    # Simulate this object being found by `db.exec().one_or_none()`
    # The mock_db_session.exec() needs to return a mock that has one_or_none()
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = expected_run
    mock_db_session.exec.return_value = mock_query_result

    status_response = get_pipeline_run_status(db=mock_db_session, run_uuid=run_uuid_val)

    assert status_response is not None
    assert status_response.run_uuid == run_uuid_val
    assert status_response.status == PipelineRunStatus.COMPLETED
    assert status_response.pipeline_type == PipelineType.PDF_SUMMARIZER
    assert status_response.result == {"summary": ["Test summary"]}
    # Ensure select and where were called appropriately on the session.exec mock if needed
    # For now, just checking the outcome is sufficient if exec is broadly mocked.
    mock_db_session.exec.assert_called_once()
    # Example of how to check call if using a specific statement object:
    # statement = select(PipelineRun).where(PipelineRun.run_uuid == run_uuid_val)
    # mock_db_session.exec.assert_called_with(statement) # This requires comparing SQLModel statements, can be tricky

def test_get_pipeline_run_status_not_found(mock_db_session: MagicMock):
    """Test retrieving status for a non-existent pipeline run."""
    run_uuid_val = uuid.uuid4()
    # Simulate `db.exec().one_or_none()` returning None
    mock_query_result = MagicMock()
    mock_query_result.one_or_none.return_value = None
    mock_db_session.exec.return_value = mock_query_result

    status_response = get_pipeline_run_status(db=mock_db_session, run_uuid=run_uuid_val)
    assert status_response is None
    mock_db_session.exec.assert_called_once()

# All TODO items from previous step are now implemented as tests above.
