import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import os
from uuid import uuid4

from fastapi import UploadFile
from sqlmodel import Session

from app.services.file_service import save_uploaded_file_and_log, UPLOAD_DIRECTORY
from app.models.file_models import UploadedFileLog, FileUploadResponse, UploadedFileLogCreate


@pytest_asyncio.fixture
async def mock_upload_file():
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "test_file.txt"
    mock_file.content_type = "text/plain"
    
    # Mock the read method to return content in chunks
    content = b"This is test content."
    chunks = [content[i:i + 1024] for i in range(0, len(content), 1024)]
    mock_file.read = AsyncMock(side_effect=chunks + [b""]) # Ensure it ends with empty bytes
    mock_file.close = AsyncMock()
    return mock_file

@pytest_asyncio.fixture
def mock_db_session():
    session = MagicMock(spec=Session)
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.rollback = MagicMock()
    return session

@pytest.fixture(autouse=True)
def cleanup_uploaded_files():
    # Ensure the UPLOAD_DIRECTORY exists for tests
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    yield
    # Clean up any files created in UPLOAD_DIRECTORY during tests
    for item in os.listdir(UPLOAD_DIRECTORY):
        item_path = os.path.join(UPLOAD_DIRECTORY, item)
        if os.path.isfile(item_path) and item.startswith("test_"):
            os.remove(item_path)

@pytest.mark.asyncio
async def test_save_uploaded_file_and_log_success(mock_upload_file: AsyncMock, mock_db_session: MagicMock):
    """
    Test successful file upload and logging.
    """
    expected_file_path = os.path.join(UPLOAD_DIRECTORY, mock_upload_file.filename)
    expected_file_size = len(b"This is test content.")
    generated_uuid = uuid4()

    # Patch os.path.getsize and os.makedirs (though makedirs might not be strictly needed here if UPLOAD_DIRECTORY is pre-created)
    with patch("app.services.file_service.os.path.getsize", return_value=expected_file_size), \
         patch("app.services.file_service.os.makedirs"), \
         patch("app.models.file_models.uuid.uuid4", return_value=generated_uuid):


        # Simulate that db.refresh updates the mock object with an ID and uuid
        def refresh_side_effect(obj):
            if isinstance(obj, UploadedFileLog):
                obj.id = 1
                obj.file_uuid = generated_uuid # Use the same UUID
        mock_db_session.refresh.side_effect = refresh_side_effect

        response = await save_uploaded_file_and_log(mock_upload_file, mock_db_session)

    assert os.path.exists(expected_file_path)
    with open(expected_file_path, "rb") as f:
        assert f.read() == b"This is test content."

    mock_upload_file.read.assert_called() # Check if read was called
    mock_upload_file.close.assert_called_once()

    mock_db_session.add.assert_called_once()
    added_obj = mock_db_session.add.call_args[0][0]
    assert isinstance(added_obj, UploadedFileLog)
    assert added_obj.filename == mock_upload_file.filename
    assert added_obj.content_type == mock_upload_file.content_type
    assert added_obj.size_bytes == expected_file_size
    assert added_obj.storage_location == expected_file_path
    
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

    assert response.filename == mock_upload_file.filename
    assert response.content_type == mock_upload_file.content_type
    assert response.size == expected_file_size
    assert response.message == f"File \'{mock_upload_file.filename}\' saved to {expected_file_path} and logged with ID 1"
    assert response.file_log_id == 1
    assert response.file_uuid == generated_uuid


@pytest.mark.asyncio
async def test_save_uploaded_file_failure_during_write(mock_upload_file: AsyncMock, mock_db_session: MagicMock):
    """
    Test failure during file writing (e.g., disk full).
    """
    expected_file_path = os.path.join(UPLOAD_DIRECTORY, mock_upload_file.filename)

    with patch("builtins.open", side_effect=IOError("Disk full")), \
         patch("app.services.file_service.os.path.exists", return_value=False) as mock_exists, \
         patch("app.services.file_service.os.remove") as mock_remove:

        response = await save_uploaded_file_and_log(mock_upload_file, mock_db_session)

    mock_upload_file.close.assert_called_once()
    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()
    mock_db_session.rollback.assert_called_once() # Ensure rollback is called on exception

    assert response.filename == mock_upload_file.filename
    assert response.content_type == mock_upload_file.content_type
    assert response.size == 0
    assert "Failed to save file" in response.message
    assert "Disk full" in response.message
    assert response.file_log_id is None
    assert response.file_uuid is None
    
    # Check if we attempted to remove the file (even if it didn't exist, good to check the logic path)
    # If open fails, the file might not be created, so os.remove might not be called.
    # If the exception happens during write, the file might exist.
    # Given `open` is mocked to raise IOError, `file_location` might not be created.
    # `os.path.exists` is mocked to return False, so remove will not be called.
    mock_exists.assert_called_with(expected_file_path)
    mock_remove.assert_not_called()


@pytest.mark.asyncio
async def test_save_uploaded_file_failure_db_commit(mock_upload_file: AsyncMock, mock_db_session: MagicMock):
    """
    Test failure during database commit.
    """
    expected_file_path = os.path.join(UPLOAD_DIRECTORY, mock_upload_file.filename)
    expected_file_size = len(b"This is test content.")

    mock_db_session.commit.side_effect = Exception("DB commit error")

    with patch("app.services.file_service.os.path.getsize", return_value=expected_file_size), \
         patch("app.services.file_service.os.makedirs"), \
         patch("app.services.file_service.os.path.exists", return_value=True) as mock_exists, \
         patch("app.services.file_service.os.remove") as mock_remove:

        response = await save_uploaded_file_and_log(mock_upload_file, mock_db_session)

    assert os.path.exists(expected_file_path) # File should still be written

    mock_upload_file.close.assert_called_once()
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()
    mock_db_session.refresh.assert_not_called() # Refresh should not be called if commit fails

    # Check if cleanup of the file was attempted
    mock_exists.assert_called_with(expected_file_path)
    mock_remove.assert_called_once_with(expected_file_path)

    assert response.filename == mock_upload_file.filename
    assert response.content_type == mock_upload_file.content_type
    assert response.size == 0 # Size is 0 in the failure response
    assert "Failed to save file" in response.message
    assert "DB commit error" in response.message
    assert response.file_log_id is None
    assert response.file_uuid is None

@pytest.mark.asyncio
async def test_save_uploaded_file_and_log_edge_case_empty_file(mock_db_session: MagicMock):
    """
    Test successful upload and logging of an empty file.
    """
    mock_empty_file = AsyncMock(spec=UploadFile)
    mock_empty_file.filename = "test_empty_file.txt"
    mock_empty_file.content_type = "text/plain"
    mock_empty_file.read = AsyncMock(return_value=b"") # Single call returns empty bytes
    mock_empty_file.close = AsyncMock()
    
    expected_file_path = os.path.join(UPLOAD_DIRECTORY, mock_empty_file.filename)
    expected_file_size = 0 # Empty file
    generated_uuid = uuid4()

    with patch("app.services.file_service.os.path.getsize", return_value=expected_file_size), \
         patch("app.services.file_service.os.makedirs"), \
         patch("app.models.file_models.uuid.uuid4", return_value=generated_uuid):

        def refresh_side_effect(obj):
            if isinstance(obj, UploadedFileLog):
                obj.id = 2 # Different ID for this test
                obj.file_uuid = generated_uuid
        mock_db_session.refresh.side_effect = refresh_side_effect
        
        response = await save_uploaded_file_and_log(mock_empty_file, mock_db_session)

    assert os.path.exists(expected_file_path)
    with open(expected_file_path, "rb") as f:
        assert f.read() == b""

    mock_empty_file.read.assert_called_once() # Should be called once, returns b""
    mock_empty_file.close.assert_called_once()

    mock_db_session.add.assert_called_once()
    added_obj = mock_db_session.add.call_args[0][0]
    assert isinstance(added_obj, UploadedFileLog)
    assert added_obj.filename == mock_empty_file.filename
    assert added_obj.size_bytes == expected_file_size
    
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

    assert response.filename == mock_empty_file.filename
    assert response.size == expected_file_size
    assert response.message == f"File \'{mock_empty_file.filename}\' saved to {expected_file_path} and logged with ID 2"
    assert response.file_log_id == 2
    assert response.file_uuid == generated_uuid

@pytest.mark.asyncio
async def test_save_uploaded_file_failure_os_remove_error(mock_upload_file: AsyncMock, mock_db_session: MagicMock):
    """
    Test failure during database commit and subsequent failure during os.remove cleanup.
    """
    expected_file_path = os.path.join(UPLOAD_DIRECTORY, mock_upload_file.filename)
    expected_file_size = len(b"This is test content.")

    mock_db_session.commit.side_effect = Exception("DB commit error")

    # Mock os.remove to also raise an error
    with patch("app.services.file_service.os.path.getsize", return_value=expected_file_size), \
         patch("app.services.file_service.os.makedirs"), \
         patch("app.services.file_service.os.path.exists", return_value=True) as mock_exists, \
         patch("app.services.file_service.os.remove", side_effect=OSError("Cannot delete file")) as mock_remove, \
         patch("builtins.print") as mock_print: # To capture the print statement

        response = await save_uploaded_file_and_log(mock_upload_file, mock_db_session)

    assert os.path.exists(expected_file_path) # File should still be written

    mock_upload_file.close.assert_called_once()
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()
    
    mock_exists.assert_called_with(expected_file_path)
    mock_remove.assert_called_once_with(expected_file_path)
    
    # Check that the error during removal was printed
    mock_print.assert_any_call(f"Error removing partially saved file \'{expected_file_path}\': Cannot delete file")


    assert response.filename == mock_upload_file.filename
    assert "Failed to save file" in response.message
    assert "DB commit error" in response.message # The original error should be in the response
    assert response.file_log_id is None
    assert response.file_uuid is None
