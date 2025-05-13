import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime, timezone
from sqlmodel import Session

from app.main import app # Main FastAPI app
from app.models.pipeline_models import PipelineRunCreateResponse, PipelineRunStatusResponse, PipelineRunStatus, PipelineType
from app.db.session import get_session # To override dependency
from fastapi import HTTPException
from app.models.file_models import UploadedFileLog # Import for mocking
from app.models.pipeline_models import PipelineRun

# Fixture for TestClient (can be shared if in a conftest.py or defined per file)
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_db_session_override(client: TestClient): # client fixture to ensure app is configured
    mock_session = MagicMock(spec=Session) # Use regular MagicMock
    
    def get_mock_db():
        try:
            yield mock_session
        finally:
            pass # No cleanup needed for mock
    app.dependency_overrides[get_session] = get_mock_db
    yield mock_session
    # Cleanup dependency override after tests using this fixture are done
    del app.dependency_overrides[get_session]

def test_trigger_pipeline_success(client: TestClient, mock_db_session_override: MagicMock):
    """Test successful triggering and completion of a PDF summarizer pipeline."""
    # Create a real (not mock) response object for the service to return
    mock_run_uuid = uuid.uuid4()
    mock_service_response = PipelineRunCreateResponse(
        run_uuid=mock_run_uuid,
        status=PipelineRunStatus.COMPLETED,
        uploaded_file_log_id=123,
        message="PDF summarization flow executed synchronously."
    )
    
    # Create a real UploadedFileLog object for the DB to return
    mock_file_log = UploadedFileLog(
        id=123, file_uuid=uuid.uuid4(), filename="file.pdf", storage_location="/path/to/file.pdf",
        content_type="application/pdf", size_bytes=100, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )

    # Set up mock db.get() to return our mock file log
    mock_db_session_override.get.return_value = mock_file_log

    # Patch the pipeline service call
    with patch("app.routers.pipelines.pipeline_service.trigger_pipeline_flow", 
              return_value=mock_service_response) as mock_trigger_flow:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "PDF_SUMMARIZER"}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate against the model
    validated_response = PipelineRunCreateResponse(**data)
    assert str(validated_response.run_uuid) == str(mock_run_uuid)
    assert validated_response.status == PipelineRunStatus.COMPLETED
    assert validated_response.uploaded_file_log_id == 123
    assert "synchronously" in validated_response.message
    
    # Assert the db.get call for file log
    mock_db_session_override.get.assert_called_once_with(UploadedFileLog, 123)
    
    # Assert pipeline dispatch call
    mock_trigger_flow.assert_called_once_with(
        db=mock_db_session_override,
        uploaded_file_log_id=123,
        pipeline_type=PipelineType.PDF_SUMMARIZER # Ensure this matches the type sent in request
    )

def test_trigger_pipeline_file_log_not_found(client: TestClient, mock_db_session_override: MagicMock):
    """Test triggering pipeline when the file log ID doesn't exist."""
    # Simulate db.get(UploadedFileLog, ...) returning None
    mock_db_session_override.get.return_value = None

    # Patch the service call (shouldn't be called)
    with patch("app.routers.pipelines.pipeline_service.trigger_pipeline_flow") as mock_trigger_flow:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 999, "pipeline_type": "PDF_SUMMARIZER"}
        )
    
    assert response.status_code == 404
    assert "Uploaded file log with id 999 not found" in response.json()["detail"]
    mock_db_session_override.get.assert_called_once_with(UploadedFileLog, 999)
    mock_trigger_flow.assert_not_called() # Service was not called

def test_trigger_pipeline_unsupported_type(client: TestClient, mock_db_session_override: MagicMock):
    """Test triggering with an unsupported pipeline type."""
    # Mock finding the file log successfully
    mock_file_log = UploadedFileLog(
        id=123, file_uuid=uuid.uuid4(), filename="file.pdf", storage_location="/path/to/file.pdf",
        content_type="application/pdf", size_bytes=100, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )
    mock_db_session_override.get.return_value = mock_file_log

    # Patch the service call (shouldn't be called if router handles type check first)
    # However, the router now passes it to the service, so the service will raise the error.
    # Let's assume the service raises an HTTPException for unsupported type,
    # and the router propagates it.
    with patch("app.routers.pipelines.pipeline_service.trigger_pipeline_flow", 
              side_effect=HTTPException(status_code=400, detail="Unsupported pipeline type: RAG_CHATBOT")) as mock_trigger_flow:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "RAG_CHATBOT"} 
        )
    assert response.status_code == 400 
    assert response.json()["detail"] == "Unsupported pipeline type: RAG_CHATBOT"
    mock_db_session_override.get.assert_called_once_with(UploadedFileLog, 123) 
    mock_trigger_flow.assert_called_once() # Service IS called, and it raises the error

def test_trigger_pipeline_service_http_exception(client: TestClient, mock_db_session_override: MagicMock):
    """Test when the pipeline service layer raises a known HTTPException."""
    # Mock finding the file log successfully
    mock_file_log = UploadedFileLog(
        id=123, file_uuid=uuid.uuid4(), filename="file.pdf", storage_location="/path/to/file.pdf",
        content_type="application/pdf", size_bytes=100, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )
    mock_db_session_override.get.return_value = mock_file_log

    # Mock the service call to raise HTTPException
    with patch("app.routers.pipelines.pipeline_service.trigger_pipeline_flow", 
              side_effect=HTTPException(status_code=500, detail="Flow failed")) as mock_trigger_flow:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "PDF_SUMMARIZER"}
        )
    assert response.status_code == 500
    assert "Flow failed" in response.json()["detail"]
    mock_db_session_override.get.assert_called_once_with(UploadedFileLog, 123) # DB was checked
    mock_trigger_flow.assert_called_once() # Service was called

def test_trigger_pipeline_service_general_exception(client: TestClient, mock_db_session_override: MagicMock):
    """Test when the pipeline service layer raises an unexpected general exception."""
    # Mock finding the file log successfully
    mock_file_log = UploadedFileLog(
        id=123, file_uuid=uuid.uuid4(), filename="file.pdf", storage_location="/path/to/file.pdf",
        content_type="application/pdf", size_bytes=100, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )
    mock_db_session_override.get.return_value = mock_file_log

    # Mock the service call to raise general Exception
    with patch("app.routers.pipelines.pipeline_service.trigger_pipeline_flow", 
              side_effect=Exception("Something broke")) as mock_trigger_flow:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "PDF_SUMMARIZER"}
        )
    assert response.status_code == 500
    # Check the error message includes the exception message
    assert response.json()["detail"] == "Internal server error: Something broke"
    mock_db_session_override.get.assert_called_once_with(UploadedFileLog, 123) # DB was checked
    mock_trigger_flow.assert_called_once() # Service was called

def test_get_pipeline_status_success(client: TestClient, mock_db_session_override: MagicMock):
    """Test successfully retrieving pipeline status."""
    mock_run_uuid = uuid.uuid4()
    now = datetime.now(timezone.utc)
    # Updated response model - result should be a Dict
    mock_service_response = PipelineRunStatusResponse(
        run_uuid=mock_run_uuid,
        pipeline_type=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.COMPLETED,
        uploaded_file_log_id=123,
        result={"summary": "Summary sentence 1."}, # Changed to Dict
        error_message=None,
        created_at=now,
        updated_at=now
    )
    with patch("app.routers.pipelines.pipeline_service.get_pipeline_run_status", return_value=mock_service_response) as mock_get_status:
        response = client.get(f"/api/v1/pipelines/{mock_run_uuid}/status")
    
    assert response.status_code == 200
    data = response.json()
    # Validate structure
    validated_response = PipelineRunStatusResponse(**data)
    assert str(validated_response.run_uuid) == str(mock_run_uuid)
    assert validated_response.status == PipelineRunStatus.COMPLETED
    assert validated_response.result == {"summary": "Summary sentence 1."} # Check Dict
    mock_get_status.assert_called_once_with(run_uuid=mock_run_uuid, db=mock_db_session_override)

def test_get_pipeline_status_not_found(client: TestClient, mock_db_session_override: MagicMock):
    """Test retrieving status for a non-existent pipeline run."""
    non_existent_uuid = uuid.uuid4()
    # Service now returns None when not found
    with patch("app.routers.pipelines.pipeline_service.get_pipeline_run_status", return_value=None) as mock_get_status:
        response = client.get(f"/api/v1/pipelines/{non_existent_uuid}/status")
    
    assert response.status_code == 404 # Just check status code
    assert response.json()["detail"] == "Pipeline run not found" # Check router's message
    mock_get_status.assert_called_once_with(run_uuid=non_existent_uuid, db=mock_db_session_override)

# Edge case: Invalid UUID format for status endpoint
def test_get_pipeline_status_invalid_uuid_format(client: TestClient):
    """Test retrieving status with an invalid UUID format."""
    # No need to mock service or db session as FastAPI validation should catch this.
    response = client.get("/api/v1/pipelines/invalid-uuid-string/status")
    assert response.status_code == 422 # Unprocessable Entity
    # Check for specific Pydantic/FastAPI error message details
    details = response.json()["detail"]
    # Ensure the type check is flexible for minor variations in Pydantic/FastAPI versions
    assert any("uuid" in err.get("type", "").lower() for err in details if err.get("loc") == ["path", "run_uuid"]) 