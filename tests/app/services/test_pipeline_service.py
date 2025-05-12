import pytest
# import pytest_asyncio # No longer needed
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime, timezone

from sqlmodel import Session, select
from fastapi import HTTPException

from app.services.pipeline_service import (
    create_and_dispatch_summary_pipeline,
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
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.get = MagicMock()
    session.exec = MagicMock()
    
    session._added_objects = {}
    session._next_id_counter = 1

    def add_side_effect(obj):
        if isinstance(obj, PipelineRun):
            if obj.run_uuid:
                session._added_objects[obj.run_uuid] = obj
        else: # For other models like UploadedFileLog if they need an id
            if not getattr(obj, 'id', None):
                obj.id = session._next_id_counter
                session._added_objects[obj.id] = obj
                session._next_id_counter += 1
            else: # If ID already exists (e.g. in fixture), store it
                session._added_objects[obj.id] = obj
            
    session.add.side_effect = add_side_effect
    
    def refresh_side_effect(obj):
        # Refresh mainly relevant for objects that might get a DB-assigned ID
        # For PipelineRun, it's identified by run_uuid, already set.
        # If other models need ID simulation upon refresh, handle here.
        if not isinstance(obj, PipelineRun):
            if not getattr(obj, 'id', None):
                obj.id = session._next_id_counter
                session._added_objects[obj.id] = obj
                session._next_id_counter += 1
        elif isinstance(obj, PipelineRun) and obj.run_uuid: # Ensure PipelineRun is in store by run_uuid
            session._added_objects[obj.run_uuid] = obj
             
    session.refresh.side_effect = refresh_side_effect

    def get_side_effect(model, pk):
        # This get is primarily for UploadedFileLog in create_and_dispatch
        if model == UploadedFileLog and isinstance(pk, int) and pk in session._added_objects:
            return session._added_objects[pk]
        # For PipelineRun, the service updates status by getting via run_uuid, then adding/committing
        # So, ensure it can be found after being added.
        if model == PipelineRun and isinstance(pk, uuid.UUID) and pk in session._added_objects:
            return session._added_objects[pk]
        return None
        
    session.get.side_effect = get_side_effect
    return session

@pytest.fixture
def mock_uploaded_file_log():
    log = UploadedFileLog(
        id=1,
        file_uuid=uuid.uuid4(),
        filename="test.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        storage_location="./uploaded_files/test.pdf",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    return log

@pytest.fixture
def mock_pipeline_run():
    run_uuid_val = uuid.uuid4()
    pr = PipelineRun(
        id=1,
        run_uuid=run_uuid_val,
        pipeline_type=PipelineType.PDF_SUMMARIZER,
        status=PipelineRunStatus.PENDING,
        uploaded_file_log_id=1,
        orchestrator_run_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        result=None,
        error_message=None
    )
    return pr

def test_create_and_dispatch_summary_pipeline_success(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test successful creation and sync execution of a summary pipeline using Prefect flow."""
    # Simulated result from the Prefect flow execution
    mock_flow_result = {"status": "success", "summary": ["This is the summary."]}
    
    # Setup get for UploadedFileLog
    mock_db_session.get.return_value = mock_uploaded_file_log
    
    pipeline_run_instance_store = {} # To store added PipelineRun by run_uuid

    def capture_add_side_effect(obj):
        if isinstance(obj, PipelineRun) and obj.run_uuid:
            # Create a copy for storing state at each add, to avoid mutation issues
            pipeline_run_instance_store[obj.run_uuid] = PipelineRun.model_validate(obj) 
    original_add = mock_db_session.add.side_effect
    mock_db_session.add.side_effect = capture_add_side_effect

    # For status updates, service re-fetches PipelineRun by run_uuid using db.get
    def get_pipeline_run_for_update(model, pk):
        if model == UploadedFileLog and pk == mock_uploaded_file_log.id:
            return mock_uploaded_file_log
        if model == PipelineRun and isinstance(pk, uuid.UUID) and pk in pipeline_run_instance_store:
            return PipelineRun.model_validate(pipeline_run_instance_store[pk]) # Return a copy
        return None
    mock_db_session.get.side_effect = get_pipeline_run_for_update

    # Mock the Prefect flow to return our predefined result
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline", return_value=mock_flow_result) as mock_flow:
        response = create_and_dispatch_summary_pipeline(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id, # Use id from fixture
            file_path=mock_uploaded_file_log.storage_location,
            original_filename=mock_uploaded_file_log.filename
        )

    mock_db_session.add.side_effect = original_add # Restore original add mock

    assert response.run_uuid is not None
    final_run_state = pipeline_run_instance_store.get(response.run_uuid)
    assert final_run_state is not None

    # Check number of add calls: PENDING, RUNNING, COMPLETED states
    assert mock_db_session.add.call_count == 3
    assert mock_db_session.commit.call_count == 3 
    # Refresh is called once for the initial PipelineRun creation, then once for RUNNING update, then once for COMPLETE update
    assert mock_db_session.refresh.call_count == 3 

    assert final_run_state.status == PipelineRunStatus.COMPLETED
    assert final_run_state.result == mock_flow_result["summary"]
    assert final_run_state.error_message is None

    # Verify the Prefect flow was called with the correct file path
    mock_flow.assert_called_once_with(pdf_path=mock_uploaded_file_log.storage_location)

    assert isinstance(response, PipelineRunCreateResponse)
    assert response.status == PipelineRunStatus.COMPLETED
    assert "synchronously" in response.message

def test_create_and_dispatch_flow_failure(
    mock_db_session: MagicMock,
    mock_uploaded_file_log: UploadedFileLog
):
    """Test pipeline creation when the Prefect flow fails during execution."""
    flow_exception = Exception("Flow processing error")
    mock_db_session.get.return_value = mock_uploaded_file_log # For initial UploadedFileLog get

    pipeline_run_instance_store = {}
    def capture_add_side_effect(obj):
        if isinstance(obj, PipelineRun) and obj.run_uuid:
            pipeline_run_instance_store[obj.run_uuid] = PipelineRun.model_validate(obj)
    original_add = mock_db_session.add.side_effect
    mock_db_session.add.side_effect = capture_add_side_effect

    def get_pipeline_run_for_update(model, pk):
        if model == UploadedFileLog and pk == mock_uploaded_file_log.id:
            return mock_uploaded_file_log
        # Return a copy to simulate transaction isolation / avoid side effects across gets
        if model == PipelineRun and isinstance(pk, uuid.UUID) and pk in pipeline_run_instance_store:
            return PipelineRun.model_validate(pipeline_run_instance_store[pk])
        return None
    mock_db_session.get.side_effect = get_pipeline_run_for_update

    # Case 1: Flow function raises an exception - Expect HTTPException from service
    with pytest.raises(HTTPException) as exc_info_case1:
        with patch("app.services.pipeline_service.run_pdf_summary_pipeline", side_effect=flow_exception) as mock_flow_exc:
            create_and_dispatch_summary_pipeline(
                db=mock_db_session,
                uploaded_file_log_id=mock_uploaded_file_log.id,
                file_path=mock_uploaded_file_log.storage_location,
                original_filename=mock_uploaded_file_log.filename
            )
    
    assert exc_info_case1.value.status_code == 500
    assert str(flow_exception) in exc_info_case1.value.detail

    # Check state in store (although exception was raised, service tries to update status)
    run_uuid_exc = next(iter(pipeline_run_instance_store)) # Get the UUID captured
    final_run_state_exc = pipeline_run_instance_store.get(run_uuid_exc)
    assert final_run_state_exc is not None
    assert final_run_state_exc.status == PipelineRunStatus.FAILED
    assert str(flow_exception) in final_run_state_exc.error_message
    mock_flow_exc.assert_called_once()

    # Reset mocks for Case 2
    mock_db_session.reset_mock(return_value=True, side_effect=True)
    pipeline_run_instance_store.clear()
    mock_db_session.get.return_value = mock_uploaded_file_log
    mock_db_session.add.side_effect = capture_add_side_effect
    mock_db_session.get.side_effect = get_pipeline_run_for_update
    
    # Case 2: Flow function returns error status - Service should handle gracefully
    mock_flow_error_result = {"status": "error", "message": "Specific flow error message"}
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline", return_value=mock_flow_error_result) as mock_flow_err:
        response_err = create_and_dispatch_summary_pipeline(
            db=mock_db_session,
            uploaded_file_log_id=mock_uploaded_file_log.id,
            file_path=mock_uploaded_file_log.storage_location,
            original_filename=mock_uploaded_file_log.filename
        )
    
    run_uuid_err = response_err.run_uuid
    assert run_uuid_err is not None
    final_run_state_err = pipeline_run_instance_store.get(run_uuid_err)
    assert final_run_state_err is not None

    assert final_run_state_err.status == PipelineRunStatus.FAILED
    assert final_run_state_err.result is None
    assert mock_flow_error_result["message"] in final_run_state_err.error_message
    mock_flow_err.assert_called_once()
    assert response_err.status == PipelineRunStatus.FAILED

def test_create_and_dispatch_file_log_not_found(mock_db_session: MagicMock):
    """Test pipeline creation when uploaded file log is not found."""
    # Simulate db.get(UploadedFileLog, ...) returning None
    mock_db_session.get.return_value = None
    
    # Note: In the pipeline service, there should be a check for UploadedFileLog existence
    # that raises an HTTPException if it's not found, but it's missing in the implementation
    # Let's add a check for this expected behavior
    
    # Patch the flow function so it doesn't run if service proceeds unexpectedly
    with patch("app.services.pipeline_service.run_pdf_summary_pipeline") as mock_flow:
        with pytest.raises(HTTPException) as exc_info:
            create_and_dispatch_summary_pipeline(
                db=mock_db_session,
                uploaded_file_log_id=999,
                file_path="/path/to/file.pdf",  # Provide a reasonable path
                original_filename="file.pdf"    # Provide a reasonable filename
            )
            
    # Check that the expected exception is raised with the right message
    assert exc_info.value.status_code == 404
    assert "UploadedFileLog with id 999 not found" in exc_info.value.detail
    # Ensure no PipelineRun was attempted to be created or run
    mock_flow.assert_not_called()

def test_get_pipeline_run_status_found(mock_db_session: MagicMock, mock_pipeline_run: PipelineRun):
    """Test successful retrieval of pipeline run status."""
    # Create a mock for sqlmodel.select() to help build the query
    mock_select = MagicMock()
    mock_query = MagicMock()
    
    # Setup the chain of methods where one_or_none returns the pipeline run
    mock_query.one_or_none.return_value = mock_pipeline_run
    mock_select_where = MagicMock()
    mock_select_where.where.return_value = mock_query
    mock_select.return_value = mock_select_where
    
    # Patch select to return our prepared mock_select_where 
    # (which will be called with .where() in the service method)
    with patch("app.services.pipeline_service.select", return_value=mock_select_where) as mock_service_select:
        mock_db_session.exec.return_value = mock_query
        result = get_pipeline_run_status(mock_db_session, mock_pipeline_run.run_uuid)
    
    assert result is not None
    assert isinstance(result, PipelineRunStatusResponse)
    assert result.run_uuid == mock_pipeline_run.run_uuid
    assert result.status == mock_pipeline_run.status
    assert result.uploaded_file_log_id == mock_pipeline_run.uploaded_file_log_id
    assert result.result == mock_pipeline_run.result
    
    mock_db_session.exec.assert_called_once_with(mock_select_where.where.return_value)

def test_get_pipeline_run_status_not_found(mock_db_session: MagicMock):
    """Test retrieving status for non-existent pipeline run."""
    # Create a mock for sqlmodel.select()
    mock_select = MagicMock()
    mock_query = MagicMock()
    
    # Setup the chain of methods where one_or_none returns None
    mock_query.one_or_none.return_value = None
    mock_select_where = MagicMock()
    mock_select_where.where.return_value = mock_query
    mock_select.return_value = mock_select_where
    
    test_uuid = uuid.uuid4()
    
    # Patch select to return our prepared mock
    with patch("app.services.pipeline_service.select", return_value=mock_select_where):
        mock_db_session.exec.return_value = mock_query
        result = get_pipeline_run_status(mock_db_session, test_uuid)
    
    assert result is None
    mock_db_session.exec.assert_called_once_with(mock_select_where.where.return_value)
