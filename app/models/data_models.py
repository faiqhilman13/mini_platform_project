"""
Data models for DS1.1.2: Data Profiling Service
Defines data structures for dataset profiling and preview responses
"""

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
from enum import Enum


class DataTypeEnum(str, Enum):
    """Enum for data types detected in datasets"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical" 
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnProfile(SQLModel):
    """Profile information for a single column"""
    name: str
    data_type: DataTypeEnum
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric column statistics
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Categorical column statistics
    most_frequent_value: Optional[str] = None
    most_frequent_count: Optional[int] = None
    category_counts: Optional[Dict[str, int]] = None
    
    # Text column statistics
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    
    # Target variable suggestions
    target_suitability_score: float = 0.0
    target_recommendation: str = "Not Recommended"
    
    # Data quality indicators
    has_outliers: bool = False
    outlier_count: Optional[int] = None
    data_quality_score: float = 1.0


class DatasetProfile(SQLModel):
    """Complete profile of a dataset"""
    file_id: int
    filename: str
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    
    # Column profiles
    columns: List[ColumnProfile]
    
    # Dataset-level statistics
    missing_values_total: int
    missing_values_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    
    # Data quality assessment
    overall_quality_score: float
    quality_issues: List[str] = []
    
    # Target variable recommendations
    recommended_targets: List[Dict[str, Any]] = []
    
    # Profiling metadata
    profiled_at: datetime = Field(default_factory=datetime.utcnow)
    profile_version: str = "1.0"


class DatasetPreview(SQLModel):
    """Preview of dataset with sample rows and basic info"""
    file_id: int
    filename: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    column_types: List[str]
    sample_rows: List[Dict[str, Any]]
    preview_row_count: int
    
    # Basic statistics for quick overview
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    text_columns: List[str] = []
    datetime_columns: List[str] = []


class TargetVariableSuggestion(SQLModel):
    """Suggestion for target variable selection"""
    column_name: str
    suitability_score: float
    problem_type: str  # "classification" or "regression"
    reasoning: str
    data_type: DataTypeEnum
    unique_values: int
    missing_percentage: float
    
    # Classification-specific
    class_distribution: Optional[Dict[str, int]] = None
    is_balanced: Optional[bool] = None
    
    # Regression-specific
    is_continuous: Optional[bool] = None
    has_good_variance: Optional[bool] = None


class DataProfilingRequest(SQLModel):
    """Request model for data profiling"""
    file_id: int
    sample_size: Optional[int] = None  # Limit rows for large datasets
    detailed_analysis: bool = True
    include_correlations: bool = False
    target_suggestions: bool = True


class DataProfilingResponse(SQLModel):
    """Response model for data profiling API"""
    success: bool
    message: str
    profile: Optional[DatasetProfile] = None
    preview: Optional[DatasetPreview] = None
    processing_time_seconds: float
    
    # Error information
    error_details: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = []


# Database model for caching profiling results
class DataProfiling(SQLModel, table=True):
    """Database model for storing profiling results"""
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="uploadedfilelog.id", index=True)
    profile_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    column_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Profile summary for quick access
    total_rows: int = 0
    total_columns: int = 0
    quality_score: float = 1.0
    has_target_suggestions: bool = False 