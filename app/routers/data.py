# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlmodel import Session
from typing import Optional

from app.services.data_profiling_service import DataProfilingService
from app.models.data_models import DatasetPreview, DataProfilingResponse, DataProfilingRequest
from app.core.config import settings
from app.db.session import get_session

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/data",
    tags=["Data Analysis"],
)

# Initialize the data profiling service
profiling_service = DataProfilingService()


@router.get("/{file_id}/preview", response_model=DatasetPreview)
async def get_dataset_preview(
    file_id: int,
    num_rows: int = Query(default=10, ge=1, le=100, description="Number of rows to preview"),
    db: Session = Depends(get_session)
):
    """
    Get a quick preview of the dataset with basic information.
    
    Returns the first N rows of the dataset along with column information
    and basic statistics for quick understanding of the data structure.
    """
    try:
        preview = await profiling_service.get_dataset_preview(file_id, db, num_rows)
        return preview
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating dataset preview: {str(e)}"
        )


@router.get("/{file_id}/profile", response_model=DataProfilingResponse)
async def get_dataset_profile(
    file_id: int,
    sample_size: Optional[int] = Query(
        default=None, 
        ge=100, 
        le=50000, 
        description="Limit analysis to N rows for large datasets"
    ),
    detailed_analysis: bool = Query(
        default=True, 
        description="Whether to perform detailed statistical analysis"
    ),
    db: Session = Depends(get_session)
):
    """
    Generate a comprehensive profile of the dataset.
    
    Performs complete data analysis including:
    - Column-level statistics and data types
    - Missing value analysis
    - Data quality assessment
    - Target variable recommendations
    - Actionable recommendations for data preparation
    """
    try:
        response = await profiling_service.profile_dataset(
            file_id=file_id,
            db=db,
            sample_size=sample_size,
            detailed_analysis=detailed_analysis
        )
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error profiling dataset: {str(e)}"
        )


@router.post("/{file_id}/profile", response_model=DataProfilingResponse)
async def profile_dataset_with_options(
    file_id: int,
    request: DataProfilingRequest,
    db: Session = Depends(get_session)
):
    """
    Generate a dataset profile with custom options.
    
    Allows for more granular control over the profiling process
    through a request body with specific parameters.
    """
    try:
        response = await profiling_service.profile_dataset(
            file_id=request.file_id or file_id,
            db=db,
            sample_size=request.sample_size,
            detailed_analysis=request.detailed_analysis
        )
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error profiling dataset: {str(e)}"
        )


@router.get("/{file_id}/columns", response_model=dict)
async def get_dataset_columns(
    file_id: int,
    db: Session = Depends(get_session)
):
    """
    Get column metadata for the dataset.
    
    Returns basic column information including names, types,
    and categorization (numeric, categorical, text, datetime).
    """
    try:
        preview = await profiling_service.get_dataset_preview(file_id, db, num_rows=1)
        
        return {
            "file_id": file_id,
            "filename": preview.filename,
            "total_columns": preview.total_columns,
            "column_names": preview.column_names,
            "column_types": preview.column_types,
            "numeric_columns": preview.numeric_columns,
            "categorical_columns": preview.categorical_columns,
            "text_columns": preview.text_columns,
            "datetime_columns": preview.datetime_columns
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving column information: {str(e)}"
        ) 