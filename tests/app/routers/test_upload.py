import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from io import BytesIO
import uuid

from app.main import app # Import the main FastAPI app
from app.models.file_models import FileUploadResponse
from app.db.session import get_session # To override dependency

# Since we are testing routers, we need a TestClient
# No need for pytest-asyncio for TestClient tests if using app.dependency_overrides

@pytest.fixture(scope="module")
def client():
    # Using a module-scoped client can be more efficient if tests don't interfere
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_db_session_override(client: TestClient):
    mock_session = AsyncMock() # Use AsyncMock if get_session is async
    # If get_session is synchronous, use MagicMock
    
    # How to override:
    # 1. Define your mock session.
    # 2. Define a dependency override function that returns your mock session.
    # 3. Apply the override to the app.
    # 4. Clean up the override after the test.

    def get_mock_db():
        return mock_session

    app.dependency_overrides[get_session] = get_mock_db
    yield mock_session # Provide the mock_session to the test if needed
    del app.dependency_overrides[get_session] # Clean up


def test_upload_file_success_pdf(client: TestClient, mock_db_session_override: AsyncMock):
    """Test successful PDF file upload."""
    mock_service_response = FileUploadResponse(
        filename="test.pdf",
        content_type="application/pdf",
        size=1234,
        message="File 'test.pdf' saved and logged with ID 1",
        file_log_id=1,
        file_uuid=uuid.uuid4()
    )
    
    with patch("app.routers.upload.file_service.save_uploaded_file_and_log", AsyncMock(return_value=mock_service_response)) as mock_save:
        file_content = b"dummy pdf content"
        response = client.post(
            "/api/v1/upload/",
            files={"file": ("test.pdf", BytesIO(file_content), "application/pdf")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["content_type"] == "application/pdf"
    assert data["file_log_id"] == 1
    assert "File 'test.pdf' saved and logged with ID 1" in data["message"]
    mock_save.assert_called_once()
    # We can inspect call_args if needed:
    # called_file_arg = mock_save.call_args[1]['file']
    # assert called_file_arg.filename == "test.pdf"


def test_upload_file_success_csv(client: TestClient, mock_db_session_override: AsyncMock):
    """Test successful CSV file upload."""
    mock_service_response = FileUploadResponse(
        filename="test.csv",
        content_type="text/csv",
        size=567,
        message="File 'test.csv' saved and logged with ID 2",
        file_log_id=2,
        file_uuid=uuid.uuid4()
    )
    
    with patch("app.routers.upload.file_service.save_uploaded_file_and_log", AsyncMock(return_value=mock_service_response)) as mock_save:
        file_content = b"col1,col2\ndata1,data2"
        response = client.post(
            "/api/v1/upload/",
            files={"file": ("test.csv", BytesIO(file_content), "text/csv")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.csv"
    assert data["content_type"] == "text/csv"
    assert data["file_log_id"] == 2
    mock_save.assert_called_once()

def test_upload_file_invalid_content_type(client: TestClient, mock_db_session_override: AsyncMock):
    """Test upload with an invalid content type (e.g., image/png)."""
    with patch("app.routers.upload.file_service.save_uploaded_file_and_log", AsyncMock()) as mock_save:
        response = client.post(
            "/api/v1/upload/",
            files={"file": ("test.png", BytesIO(b"dummy png"), "image/png")}
        )
    
    assert response.status_code == 400
    assert "Invalid file type: image/png" in response.json()["detail"]
    mock_save.assert_not_called()

def test_upload_file_service_failure(client: TestClient, mock_db_session_override: AsyncMock):
    """Test scenario where file_service.save_uploaded_file_and_log indicates failure."""
    mock_service_failure_response = FileUploadResponse(
        filename="fail.pdf",
        content_type="application/pdf",
        size=0,
        message="Failed to save file 'fail.pdf'. Error: Disk full",
        file_log_id=None, # Important: service indicates failure via message and no log_id
        file_uuid=None
    )

    with patch("app.routers.upload.file_service.save_uploaded_file_and_log", AsyncMock(return_value=mock_service_failure_response)) as mock_save:
        response = client.post(
            "/api/v1/upload/",
            files={"file": ("fail.pdf", BytesIO(b"dummy content"), "application/pdf")}
        )

    assert response.status_code == 500 # As per router logic
    assert "Failed to save file 'fail.pdf'. Error: Disk full" in response.json()["detail"]
    mock_save.assert_called_once()

def test_upload_no_file(client: TestClient, mock_db_session_override: AsyncMock):
    """Test uploading with no file attached (FastAPI should handle this)."""
    response = client.post("/api/v1/upload/") # No files attached
    assert response.status_code == 422 # Unprocessable Entity for missing 'file'
    # The detail message might vary slightly based on FastAPI version,
    # so checking for key parts is often better.
    # Example: assert "field required" in response.json()["detail"][0]["msg"].lower()
    # For now, status code 422 is a good check.

# Edge case: Empty file (0 bytes) but valid type
def test_upload_empty_file_valid_type(client: TestClient, mock_db_session_override: AsyncMock):
    """Test successful upload of an empty file with a valid content type."""
    mock_service_response = FileUploadResponse(
        filename="empty.pdf",
        content_type="application/pdf",
        size=0,
        message="File 'empty.pdf' saved and logged with ID 3",
        file_log_id=3,
        file_uuid=uuid.uuid4()
    )
    
    with patch("app.routers.upload.file_service.save_uploaded_file_and_log", AsyncMock(return_value=mock_service_response)) as mock_save:
        response = client.post(
            "/api/v1/upload/",
            files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")} # Empty BytesIO
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "empty.pdf"
    assert data["size"] == 0
    assert data["file_log_id"] == 3
    mock_save.assert_called_once() 