"""
Data Profiling Service for DS1.1.2
Provides automated data analysis, quality assessment, and target variable suggestions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sqlmodel import Session
import os
import time
from datetime import datetime

from app.models.data_models import (
    DatasetProfile, DatasetPreview, ColumnProfile, DataTypeEnum,
    TargetVariableSuggestion, DataProfilingResponse, DataProfiling
)
from app.models.file_models import UploadedFileLog


class DataProfilingService:
    """Service for comprehensive dataset profiling and analysis"""
    
    def __init__(self):
        self.max_categorical_unique = 50  # Max unique values for categorical
        self.max_preview_rows = 100      # Max rows in preview
        self.outlier_threshold = 3       # Z-score threshold for outliers
        
    async def profile_dataset(
        self, 
        file_id: int, 
        db: Session,
        sample_size: Optional[int] = None,
        detailed_analysis: bool = True
    ) -> DataProfilingResponse:
        """
        Complete profiling of a dataset with quality assessment and recommendations
        
        Args:
            file_id: ID of the uploaded file
            db: Database session
            sample_size: Limit analysis to N rows (for large datasets)
            detailed_analysis: Whether to perform detailed statistical analysis
            
        Returns:
            DataProfilingResponse with complete profiling results
        """
        start_time = time.time()
        
        try:
            # Get file information
            file_log = db.get(UploadedFileLog, file_id)
            if not file_log:
                return DataProfilingResponse(
                    success=False,
                    message=f"File with ID {file_id} not found",
                    processing_time_seconds=time.time() - start_time,
                    error_details="File not found in database"
                )
            
            if not file_log.is_dataset:
                return DataProfilingResponse(
                    success=False,
                    message=f"File {file_log.filename} is not a dataset",
                    processing_time_seconds=time.time() - start_time,
                    error_details="File is not marked as a dataset"
                )
            
            # Check if profiling already exists and is recent
            existing_profile = self._get_cached_profile(file_id, db)
            if existing_profile and not detailed_analysis:
                return DataProfilingResponse(
                    success=True,
                    message="Retrieved cached profiling results",
                    profile=self._deserialize_profile(existing_profile),
                    processing_time_seconds=time.time() - start_time
                )
            
            # Load dataset
            df = self._load_dataset(file_log.storage_location, sample_size)
            
            # Generate preview
            preview = self._generate_preview(df, file_id, file_log.filename)
            
            # Generate complete profile
            profile = await self._generate_profile(df, file_id, file_log.filename, detailed_analysis)
            
            # Cache results
            await self._cache_profile(file_id, profile, db)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(profile)
            
            processing_time = time.time() - start_time
            
            return DataProfilingResponse(
                success=True,
                message=f"Successfully profiled dataset '{file_log.filename}'",
                profile=profile,
                preview=preview,
                processing_time_seconds=processing_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return DataProfilingResponse(
                success=False,
                message=f"Error profiling dataset: {str(e)}",
                processing_time_seconds=processing_time,
                error_details=str(e)
            )
    
    async def get_dataset_preview(
        self, 
        file_id: int, 
        db: Session,
        num_rows: int = 10
    ) -> DatasetPreview:
        """Get a quick preview of the dataset with basic information"""
        
        file_log = db.get(UploadedFileLog, file_id)
        if not file_log:
            raise ValueError(f"File with ID {file_id} not found")
        
        df = self._load_dataset(file_log.storage_location, num_rows)
        return self._generate_preview(df, file_id, file_log.filename)
    
    def _load_dataset(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from file with optional sampling"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            # Load CSV with optional sampling
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size)
            else:
                df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            # Load Excel with optional sampling
            if sample_size:
                df = pd.read_excel(file_path, nrows=sample_size)
            else:
                df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return df
    
    def _generate_preview(self, df: pd.DataFrame, file_id: int, filename: str) -> DatasetPreview:
        """Generate a quick preview of the dataset"""
        
        # Basic information
        total_rows, total_columns = df.shape
        column_names = df.columns.tolist()
        column_types = [str(dtype) for dtype in df.dtypes]
        
        # Sample rows (convert to dict for JSON serialization)
        preview_rows = min(self.max_preview_rows, len(df))
        sample_rows = df.head(preview_rows).fillna("").to_dict('records')
        
        # Categorize columns by type
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = []
        text_columns = []
        datetime_columns = []
        
        for col in df.columns:
            if col in numeric_columns:
                continue
            elif df[col].dtype == 'object':
                # Check if it's categorical or text
                unique_count = df[col].nunique()
                if unique_count <= self.max_categorical_unique:
                    categorical_columns.append(col)
                else:
                    text_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
        
        return DatasetPreview(
            file_id=file_id,
            filename=filename,
            total_rows=total_rows,
            total_columns=total_columns,
            column_names=column_names,
            column_types=column_types,
            sample_rows=sample_rows,
            preview_row_count=preview_rows,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            text_columns=text_columns,
            datetime_columns=datetime_columns
        )
    
    async def _generate_profile(
        self, 
        df: pd.DataFrame, 
        file_id: int, 
        filename: str, 
        detailed: bool = True
    ) -> DatasetProfile:
        """Generate complete dataset profile with column analysis"""
        
        total_rows, total_columns = df.shape
        memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))  # Convert to Python float
        
        # Dataset-level statistics - convert numpy types to Python types
        missing_values_total = int(df.isnull().sum().sum())  # Convert numpy.int64 to Python int
        missing_values_percentage = float((missing_values_total / (total_rows * total_columns)) * 100)  # Convert to Python float
        duplicate_rows_count = int(df.duplicated().sum())  # Convert numpy.int64 to Python int
        duplicate_rows_percentage = float((duplicate_rows_count / total_rows) * 100)  # Convert to Python float
        
        # Profile each column
        column_profiles = []
        quality_issues = []
        
        for col in df.columns:
            profile = await self._profile_column(df[col], col, total_rows, detailed)
            column_profiles.append(profile)
            
            # Collect quality issues
            if profile.missing_percentage > 50:
                quality_issues.append(f"Column '{col}' has {profile.missing_percentage:.1f}% missing values")
            if profile.has_outliers and profile.outlier_count > total_rows * 0.1:
                quality_issues.append(f"Column '{col}' has many outliers ({profile.outlier_count})")
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(column_profiles, missing_values_percentage, duplicate_rows_percentage)
        
        # Generate target variable recommendations
        target_recommendations = self._suggest_target_variables(column_profiles, df)
        
        return DatasetProfile(
            file_id=file_id,
            filename=filename,
            total_rows=total_rows,
            total_columns=total_columns,
            memory_usage_mb=memory_usage_mb,
            columns=column_profiles,
            missing_values_total=missing_values_total,
            missing_values_percentage=missing_values_percentage,
            duplicate_rows_count=duplicate_rows_count,
            duplicate_rows_percentage=duplicate_rows_percentage,
            overall_quality_score=quality_score,
            quality_issues=quality_issues,
            recommended_targets=target_recommendations
        )
    
    async def _profile_column(
        self, 
        series: pd.Series, 
        col_name: str, 
        total_rows: int, 
        detailed: bool
    ) -> ColumnProfile:
        """Profile a single column with comprehensive statistics"""
        
        # Basic statistics - convert numpy types to Python types
        missing_count = int(series.isnull().sum())  # Convert numpy.int64 to Python int
        missing_percentage = float((missing_count / total_rows) * 100)  # Convert to Python float
        unique_count = int(series.nunique())  # Convert numpy.int64 to Python int
        unique_percentage = float((unique_count / total_rows) * 100)  # Convert to Python float
        
        # Determine data type
        data_type = self._determine_data_type(series)
        
        # Initialize profile
        profile = ColumnProfile(
            name=col_name,
            data_type=data_type,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage
        )
        
        # Type-specific analysis
        if data_type == DataTypeEnum.NUMERIC and detailed:
            profile = self._profile_numeric_column(series, profile)
        elif data_type == DataTypeEnum.CATEGORICAL and detailed:
            profile = self._profile_categorical_column(series, profile)
        elif data_type == DataTypeEnum.TEXT and detailed:
            profile = self._profile_text_column(series, profile)
        
        # Target variable suitability analysis
        profile.target_suitability_score, profile.target_recommendation = self._assess_target_suitability(series, data_type)
        
        # Data quality assessment
        profile.data_quality_score = self._calculate_column_quality_score(profile)
        
        return profile
    
    def _determine_data_type(self, series: pd.Series) -> DataTypeEnum:
        """Determine the semantic data type of a column"""
        
        # Remove missing values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return DataTypeEnum.UNKNOWN
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return DataTypeEnum.NUMERIC
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataTypeEnum.DATETIME
        
        # Check if boolean
        if series.dtype == bool or set(non_null_series.unique()).issubset({True, False, 'true', 'false', 'True', 'False', 1, 0}):
            return DataTypeEnum.BOOLEAN
        
        # For object dtype, determine if categorical or text
        if series.dtype == 'object':
            unique_count = series.nunique()
            total_count = len(series)
            
            # If less than 50 unique values or less than 10% unique, treat as categorical
            if unique_count <= self.max_categorical_unique or (unique_count / total_count) < 0.1:
                return DataTypeEnum.CATEGORICAL
            else:
                return DataTypeEnum.TEXT
        
        return DataTypeEnum.UNKNOWN
    
    def _profile_numeric_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Add numeric-specific statistics to column profile"""
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) > 0:
            profile.mean = float(numeric_series.mean())
            profile.median = float(numeric_series.median())
            profile.std = float(numeric_series.std())
            profile.min_value = float(numeric_series.min())
            profile.max_value = float(numeric_series.max())
            
            # Outlier detection using Z-score
            if len(numeric_series) > 10:  # Only for sufficient data
                z_scores = np.abs((numeric_series - numeric_series.mean()) / numeric_series.std())
                outliers = z_scores > self.outlier_threshold
                profile.has_outliers = bool(outliers.any())  # Convert numpy.bool to Python bool
                profile.outlier_count = int(outliers.sum())  # Convert numpy.int64 to Python int
        
        return profile
    
    def _profile_categorical_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Add categorical-specific statistics to column profile"""
        
        value_counts = series.value_counts()
        
        if len(value_counts) > 0:
            profile.most_frequent_value = str(value_counts.index[0])
            profile.most_frequent_count = int(value_counts.iloc[0])  # Convert numpy.int64 to Python int
            
            # Store top categories (limit to prevent huge JSON)
            top_categories = value_counts.head(20).to_dict()
            profile.category_counts = {str(k): int(v) for k, v in top_categories.items()}
        
        return profile
    
    def _profile_text_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Add text-specific statistics to column profile"""
        
        text_series = series.dropna().astype(str)
        
        if len(text_series) > 0:
            lengths = text_series.str.len()
            profile.avg_length = float(lengths.mean())  # Convert numpy.float64 to Python float
            profile.max_length = int(lengths.max())     # Convert numpy.int64 to Python int
            profile.min_length = int(lengths.min())     # Convert numpy.int64 to Python int
        
        return profile
    
    def _assess_target_suitability(self, series: pd.Series, data_type: DataTypeEnum) -> Tuple[float, str]:
        """Assess how suitable a column is as a target variable"""
        
        score = 0.0
        recommendation = "Not Recommended"
        
        # Basic checks - convert numpy types to Python types
        missing_pct = float((series.isnull().sum() / len(series)) * 100)  # Convert to Python float
        unique_count = int(series.nunique())  # Convert numpy.int64 to Python int
        unique_pct = float((unique_count / len(series)) * 100)  # Convert to Python float
        
        # Penalize high missing values
        if missing_pct > 30:
            return 0.1, "High missing values"
        
        # Assess by data type
        if data_type == DataTypeEnum.NUMERIC:
            # Good for regression if continuous with good variance
            if unique_count > 10 and series.std() > 0:
                score = 0.8 - (missing_pct / 100)
                recommendation = "Good for Regression"
            elif unique_count <= 10:
                score = 0.7 - (missing_pct / 100)
                recommendation = "Suitable for Classification"
        
        elif data_type == DataTypeEnum.CATEGORICAL:
            # Good for classification if reasonable number of classes
            if 2 <= unique_count <= 20:
                score = 0.9 - (missing_pct / 100)
                recommendation = "Excellent for Classification"
            elif unique_count > 20:
                score = 0.4
                recommendation = "Too many categories"
        
        elif data_type == DataTypeEnum.BOOLEAN:
            score = 0.95 - (missing_pct / 100)
            recommendation = "Perfect for Binary Classification"
        
        return score, recommendation
    
    def _calculate_column_quality_score(self, profile: ColumnProfile) -> float:
        """Calculate a quality score for a column (0-1)"""
        
        score = 1.0
        
        # Penalize missing values
        score -= (profile.missing_percentage / 100) * 0.5
        
        # Penalize outliers
        if profile.has_outliers and profile.outlier_count:
            outlier_ratio = profile.outlier_count / (profile.missing_count + profile.unique_count)
            score -= min(outlier_ratio * 0.3, 0.3)
        
        # Bonus for good uniqueness (not too high, not too low)
        if 1 < profile.unique_percentage < 90:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_quality_score(
        self, 
        columns: List[ColumnProfile], 
        missing_pct: float, 
        duplicate_pct: float
    ) -> float:
        """Calculate overall dataset quality score"""
        
        if not columns:
            return 0.0
        
        # Average column quality scores
        avg_column_quality = sum(col.data_quality_score for col in columns) / len(columns)
        
        # Dataset-level penalties
        score = avg_column_quality
        score -= (missing_pct / 100) * 0.2  # Penalize overall missing values
        score -= (duplicate_pct / 100) * 0.3  # Penalize duplicates more heavily
        
        return max(0.0, min(1.0, score))
    
    def _suggest_target_variables(self, columns: List[ColumnProfile], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate target variable suggestions"""
        
        suggestions = []
        
        # Sort columns by target suitability
        sorted_columns = sorted(columns, key=lambda x: x.target_suitability_score, reverse=True)
        
        # Take top 3 suggestions with score > 0.5
        for col in sorted_columns[:3]:
            if col.target_suitability_score > 0.5:
                problem_type = "regression" if col.data_type == DataTypeEnum.NUMERIC else "classification"
                
                suggestion = {
                    "column_name": col.name,
                    "suitability_score": col.target_suitability_score,
                    "problem_type": problem_type,
                    "reasoning": col.target_recommendation,
                    "data_type": col.data_type.value,
                    "unique_values": col.unique_count,
                    "missing_percentage": col.missing_percentage
                }
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_recommendations(self, profile: DatasetProfile) -> List[str]:
        """Generate actionable recommendations based on profiling results"""
        
        recommendations = []
        
        # Quality-based recommendations
        if profile.overall_quality_score < 0.7:
            recommendations.append("Consider data cleaning - overall quality score is low")
        
        if profile.missing_values_percentage > 20:
            recommendations.append("High percentage of missing values detected - consider imputation strategies")
        
        if profile.duplicate_rows_percentage > 5:
            recommendations.append(f"{profile.duplicate_rows_count} duplicate rows found - consider deduplication")
        
        # Column-specific recommendations
        high_missing_cols = [col.name for col in profile.columns if col.missing_percentage > 50]
        if high_missing_cols:
            recommendations.append(f"Consider removing columns with high missing values: {', '.join(high_missing_cols[:3])}")
        
        # Target variable recommendations
        if profile.recommended_targets:
            best_target = profile.recommended_targets[0]
            recommendations.append(f"'{best_target['column_name']}' appears suitable for {best_target['problem_type']} problems")
        else:
            recommendations.append("No clear target variable candidates found - manual selection may be needed")
        
        # Size recommendations
        if profile.total_rows < 100:
            recommendations.append("Dataset is quite small - consider collecting more data for robust ML models")
        elif profile.total_rows > 100000:
            recommendations.append("Large dataset detected - consider sampling for faster model training")
        
        return recommendations
    
    def _get_cached_profile(self, file_id: int, db: Session) -> Optional[DataProfiling]:
        """Retrieve cached profiling results if available and recent"""
        # Implementation would query the database for cached results
        # For now, return None to always generate fresh profiles
        return None
    
    async def _cache_profile(self, file_id: int, profile: DatasetProfile, db: Session):
        """Cache profiling results for future use"""
        # Implementation would store the profile in the database
        # For now, skip caching
        pass
    
    def _serialize_profile(self, profile: DatasetProfile) -> Dict[str, Any]:
        """Serialize profile for database storage"""
        return profile.model_dump()
    
    def _deserialize_profile(self, cached_profile: DataProfiling) -> DatasetProfile:
        """Deserialize profile from database storage"""
        return DatasetProfile(**cached_profile.profile_data) 