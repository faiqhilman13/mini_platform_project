import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import uuid
from datetime import datetime, timezone

from app.main import app # Main FastAPI app
from app.models.pipeline_models import PipelineRunRead, PipelineRunStatus, PipelineType
from app.db.session import get_session # To override dependency
from fastapi import HTTPException

# Fixture for TestClient (can be shared if in a conftest.py or defined per file)
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_db_session_override(client: TestClient): # client fixture to ensure app is configured
    mock_session = AsyncMock()
    def get_mock_db():
        return mock_session
    app.dependency_overrides[get_session] = get_mock_db
    yield mock_session
    del app.dependency_overrides[get_session]

def test_trigger_pipeline_success(client: TestClient, mock_db_session_override: AsyncMock):
    """Test successful triggering of a PDF summarizer pipeline."""
    mock_run_uuid = uuid.uuid4()
    mock_service_response = PipelineRunRead(
        id=1,
        run_uuid=mock_run_uuid,
        pipeline_name=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.QUEUED,
        uploaded_file_log_id=123,
        celery_task_id="celery-task-abc",
        error_message=None,
        summary_url=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

    with patch("app.routers.pipelines.service.create_and_dispatch_summary_pipeline", AsyncMock(return_value=mock_service_response)) as mock_create_dispatch:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "PDF_SUMMARIZER"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["run_uuid"] == str(mock_run_uuid)
    assert data["status"] == "QUEUED"
    assert data["uploaded_file_log_id"] == 123
    mock_create_dispatch.assert_called_once_with(uploaded_file_log_id=123, db=mock_db_session_override)

def test_trigger_pipeline_unsupported_type(client: TestClient, mock_db_session_override: AsyncMock):
    """Test triggering with an unsupported pipeline type."""
    with patch("app.routers.pipelines.service.create_and_dispatch_summary_pipeline", AsyncMock()) as mock_create_dispatch:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "unsupported_type"}
        )
    assert response.status_code == 422
    # Check for Pydantic validation error message structure
    details = response.json()["detail"]
    assert any("Input should be 'PDF_SUMMARIZER', 'RAG_CHATBOT' or 'TEXT_CLASSIFIER'" in err["msg"] for err in details if err["loc"] == ["body", "pipeline_type"])
    mock_create_dispatch.assert_not_called()

def test_trigger_pipeline_service_http_exception(client: TestClient, mock_db_session_override: AsyncMock):
    """Test when the service layer raises a known HTTPException (e.g., file not found)."""
    with patch("app.routers.pipelines.service.create_and_dispatch_summary_pipeline", AsyncMock(side_effect=HTTPException(status_code=404, detail="File not found"))) as mock_create_dispatch:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 999, "pipeline_type": "PDF_SUMMARIZER"}
        )
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]
    mock_create_dispatch.assert_called_once()

def test_trigger_pipeline_service_general_exception(client: TestClient, mock_db_session_override: AsyncMock):
    """Test when the service layer raises an unexpected general exception."""
    with patch("app.routers.pipelines.service.create_and_dispatch_summary_pipeline", AsyncMock(side_effect=Exception("Something broke"))) as mock_create_dispatch:
        response = client.post(
            "/api/v1/pipelines/trigger",
            json={"uploaded_file_log_id": 123, "pipeline_type": "PDF_SUMMARIZER"}
        )
    assert response.status_code == 500
    assert "Internal server error triggering pipeline: Something broke" in response.json()["detail"]
    mock_create_dispatch.assert_called_once()


def test_get_pipeline_status_success(client: TestClient, mock_db_session_override: AsyncMock):
    """Test successfully retrieving pipeline status."""
    mock_run_uuid = uuid.uuid4()
    mock_service_response = PipelineRunRead(
        id=1,
        run_uuid=mock_run_uuid,
        pipeline_name=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.COMPLETED,
        uploaded_file_log_id=123,
        celery_task_id="celery-task-abc",
        error_message=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    with patch("app.routers.pipelines.service.get_pipeline_run_status", AsyncMock(return_value=mock_service_response)) as mock_get_status:
        response = client.get(f"/api/v1/pipelines/{mock_run_uuid}/status")
    
    assert response.status_code == 200
    data = response.json()
    assert data["run_uuid"] == str(mock_run_uuid)
    assert data["status"] == "COMPLETED"
    mock_get_status.assert_called_once_with(run_uuid=mock_run_uuid, db=mock_db_session_override)

def test_get_pipeline_status_not_found(client: TestClient, mock_db_session_override: AsyncMock):
    """Test retrieving status for a non-existent pipeline run."""
    non_existent_uuid = uuid.uuid4()
    with patch("app.routers.pipelines.service.get_pipeline_run_status", AsyncMock(return_value=None)) as mock_get_status:
        response = client.get(f"/api/v1/pipelines/{non_existent_uuid}/status")
    
    assert response.status_code == 404
    assert f"Pipeline run with UUID {non_existent_uuid} not found" in response.json()["detail"]
    mock_get_status.assert_called_once_with(run_uuid=non_existent_uuid, db=mock_db_session_override)

# Edge case: Invalid UUID format for status endpoint
def test_get_pipeline_status_invalid_uuid_format(client: TestClient):
    """Test retrieving status with an invalid UUID format."""
    # No need to mock service or db session as FastAPI validation should catch this.
    response = client.get("/api/v1/pipelines/invalid-uuid-string/status")
    assert response.status_code == 422 # Unprocessable Entity
    # Check for specific Pydantic/FastAPI error message details
    details = response.json()["detail"]
    assert any("uuid_parsing" in err["type"].lower() for err in details if "type" in err) # Check for uuid_parsing type 