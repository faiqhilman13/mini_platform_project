#!/usr/bin/env python3
"""
Unit tests for DS1.1.2: Data Profiling Service
Tests data profiling functionality, column analysis, and target suggestions
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock

from app.services.data_profiling_service import DataProfilingService
from app.models.data_models import DataTypeEnum


class TestDataProfilingService:
    """Test cases for DataProfilingService"""

    @pytest.fixture
    def service(self):
        """Create a DataProfilingService instance for testing"""
        return DataProfilingService()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        data = {
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'text_col': ['Long text string 1', 'Long text string 2', 'Long text string 3', 'Long text string 4', 'Long text string 5'],
            'boolean_col': [True, False, True, False, True],
            'missing_col': [1, 2, None, 4, None]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_file_log(self):
        """Create a mock file log for testing"""
        mock_file = Mock()
        mock_file.id = 1
        mock_file.filename = "test_dataset.csv"
        mock_file.storage_location = "test_dataset.csv"
        mock_file.is_dataset = True
        return mock_file

    @pytest.fixture
    def mock_db(self, mock_file_log):
        """Create a mock database session"""
        mock_db = Mock()
        mock_db.get = Mock(return_value=mock_file_log)
        return mock_db

    def test_determine_data_type_numeric(self, service):
        """Test data type detection for numeric columns"""
        numeric_series = pd.Series([1, 2, 3, 4, 5])
        data_type = service._determine_data_type(numeric_series)
        assert data_type == DataTypeEnum.NUMERIC

    def test_determine_data_type_categorical(self, service):
        """Test data type detection for categorical columns"""
        categorical_series = pd.Series(['A', 'B', 'A', 'C', 'B'])
        data_type = service._determine_data_type(categorical_series)
        assert data_type == DataTypeEnum.CATEGORICAL

    def test_determine_data_type_text(self, service):
        """Test data type detection for text columns"""
        # Create text series with many unique values
        text_series = pd.Series([f'Text string {i}' for i in range(100)])
        data_type = service._determine_data_type(text_series)
        assert data_type == DataTypeEnum.TEXT

    def test_determine_data_type_boolean(self, service):
        """Test data type detection for boolean columns"""
        boolean_series = pd.Series(['true', 'false', 'true', 'false'])
        data_type = service._determine_data_type(boolean_series)
        assert data_type == DataTypeEnum.BOOLEAN

    def test_determine_data_type_unknown(self, service):
        """Test data type detection for empty/unknown columns"""
        empty_series = pd.Series([None, None, None])
        data_type = service._determine_data_type(empty_series)
        assert data_type == DataTypeEnum.UNKNOWN

    def test_profile_numeric_column(self, service):
        """Test numeric column profiling"""
        numeric_series = pd.Series([1, 2, 3, 4, 5, 100])  # Include outlier
        
        from app.models.data_models import ColumnProfile
        profile = ColumnProfile(
            name="test_col",
            data_type=DataTypeEnum.NUMERIC,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=6,
            unique_percentage=100.0
        )
        
        updated_profile = service._profile_numeric_column(numeric_series, profile)
        
        assert updated_profile.mean is not None
        assert updated_profile.median is not None
        assert updated_profile.std is not None
        assert updated_profile.min_value == 1.0
        assert updated_profile.max_value == 100.0

    def test_profile_categorical_column(self, service):
        """Test categorical column profiling"""
        categorical_series = pd.Series(['A', 'B', 'A', 'A', 'C'])
        
        from app.models.data_models import ColumnProfile
        profile = ColumnProfile(
            name="test_col",
            data_type=DataTypeEnum.CATEGORICAL,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=3,
            unique_percentage=60.0
        )
        
        updated_profile = service._profile_categorical_column(categorical_series, profile)
        
        assert updated_profile.most_frequent_value == 'A'
        assert updated_profile.most_frequent_count == 3
        assert updated_profile.category_counts is not None
        assert 'A' in updated_profile.category_counts

    def test_profile_text_column(self, service):
        """Test text column profiling"""
        text_series = pd.Series(['short', 'medium length', 'very long text string'])
        
        from app.models.data_models import ColumnProfile
        profile = ColumnProfile(
            name="test_col",
            data_type=DataTypeEnum.TEXT,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=3,
            unique_percentage=100.0
        )
        
        updated_profile = service._profile_text_column(text_series, profile)
        
        assert updated_profile.avg_length is not None
        assert updated_profile.max_length == len('very long text string')
        assert updated_profile.min_length == len('short')

    def test_assess_target_suitability_numeric_regression(self, service):
        """Test target suitability assessment for numeric regression"""
        numeric_series = pd.Series(np.random.normal(100, 20, 100))  # Continuous numeric
        score, recommendation = service._assess_target_suitability(numeric_series, DataTypeEnum.NUMERIC)
        
        assert score > 0.7  # Should be high for continuous numeric
        assert "Regression" in recommendation

    def test_assess_target_suitability_categorical_classification(self, service):
        """Test target suitability assessment for categorical classification"""
        categorical_series = pd.Series(['A', 'B', 'C'] * 20)  # 3 balanced classes
        score, recommendation = service._assess_target_suitability(categorical_series, DataTypeEnum.CATEGORICAL)
        
        assert score > 0.8  # Should be very high for balanced categorical
        assert "Classification" in recommendation

    def test_assess_target_suitability_boolean_perfect(self, service):
        """Test target suitability assessment for boolean (perfect for classification)"""
        boolean_series = pd.Series([True, False] * 25)
        score, recommendation = service._assess_target_suitability(boolean_series, DataTypeEnum.BOOLEAN)
        
        assert score > 0.9  # Should be very high for boolean
        assert "Binary Classification" in recommendation

    def test_assess_target_suitability_high_missing_values(self, service):
        """Test target suitability assessment with high missing values"""
        series_with_missing = pd.Series([1, 2, None, None, None] * 20)  # 60% missing
        score, recommendation = service._assess_target_suitability(series_with_missing, DataTypeEnum.NUMERIC)
        
        assert score < 0.2  # Should be very low due to missing values
        assert "missing" in recommendation.lower()

    def test_calculate_column_quality_score_perfect(self, service):
        """Test column quality score calculation for perfect column"""
        from app.models.data_models import ColumnProfile
        profile = ColumnProfile(
            name="perfect_col",
            data_type=DataTypeEnum.NUMERIC,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=50,
            unique_percentage=50.0,  # Good uniqueness
            has_outliers=False,
            outlier_count=0
        )
        
        quality_score = service._calculate_column_quality_score(profile)
        assert quality_score > 0.9  # Should be very high

    def test_calculate_column_quality_score_poor(self, service):
        """Test column quality score calculation for poor quality column"""
        from app.models.data_models import ColumnProfile
        profile = ColumnProfile(
            name="poor_col",
            data_type=DataTypeEnum.NUMERIC,
            missing_count=50,
            missing_percentage=50.0,  # High missing values
            unique_count=100,
            unique_percentage=100.0,
            has_outliers=True,
            outlier_count=20
        )
        
        quality_score = service._calculate_column_quality_score(profile)
        assert quality_score < 0.75  # Should be low due to issues (adjusted threshold)

    def test_calculate_quality_score_dataset_level(self, service):
        """Test overall dataset quality score calculation"""
        from app.models.data_models import ColumnProfile
        
        # Create some column profiles
        good_column = ColumnProfile(
            name="good_col", data_type=DataTypeEnum.NUMERIC,
            missing_count=0, missing_percentage=0.0,
            unique_count=50, unique_percentage=50.0,
            data_quality_score=0.9
        )
        
        poor_column = ColumnProfile(
            name="poor_col", data_type=DataTypeEnum.NUMERIC,
            missing_count=30, missing_percentage=30.0,
            unique_count=100, unique_percentage=100.0,
            data_quality_score=0.5
        )
        
        columns = [good_column, poor_column]
        missing_pct = 15.0
        duplicate_pct = 5.0
        
        quality_score = service._calculate_quality_score(columns, missing_pct, duplicate_pct)
        
        # Should be between the two column scores, penalized for dataset issues
        assert 0.5 < quality_score < 0.9

    def test_suggest_target_variables(self, service):
        """Test target variable suggestion logic"""
        from app.models.data_models import ColumnProfile
        
        # Create columns with different suitability scores
        excellent_target = ColumnProfile(
            name="excellent_target", data_type=DataTypeEnum.BOOLEAN,
            missing_count=0, missing_percentage=0.0,
            unique_count=2, unique_percentage=50.0,
            target_suitability_score=0.95
        )
        
        good_target = ColumnProfile(
            name="good_target", data_type=DataTypeEnum.CATEGORICAL,
            missing_count=0, missing_percentage=0.0,
            unique_count=3, unique_percentage=33.0,
            target_suitability_score=0.85
        )
        
        poor_target = ColumnProfile(
            name="poor_target", data_type=DataTypeEnum.TEXT,
            missing_count=50, missing_percentage=50.0,
            unique_count=100, unique_percentage=100.0,
            target_suitability_score=0.2
        )
        
        columns = [excellent_target, good_target, poor_target]
        df = pd.DataFrame()  # Empty df for this test
        
        suggestions = service._suggest_target_variables(columns, df)
        
        # Should return top suggestions with score > 0.5
        assert len(suggestions) == 2  # excellent and good targets
        assert suggestions[0]['column_name'] == 'excellent_target'
        assert suggestions[1]['column_name'] == 'good_target'

    def test_generate_recommendations_high_quality(self, service):
        """Test recommendation generation for high quality dataset"""
        from app.models.data_models import DatasetProfile, ColumnProfile
        
        profile = DatasetProfile(
            file_id=1,
            filename="test.csv",
            total_rows=1000,
            total_columns=5,
            memory_usage_mb=1.0,
            columns=[],
            missing_values_total=0,
            missing_values_percentage=0.0,
            duplicate_rows_count=0,
            duplicate_rows_percentage=0.0,
            overall_quality_score=0.95,
            quality_issues=[],
            recommended_targets=[{
                'column_name': 'target_col',
                'problem_type': 'classification'
            }]
        )
        
        recommendations = service._generate_recommendations(profile)
        
        # Should have target recommendation
        assert any('target_col' in rec for rec in recommendations)
        # Should not have quality warnings for high quality data
        assert not any('quality score is low' in rec for rec in recommendations)

    def test_generate_recommendations_poor_quality(self, service):
        """Test recommendation generation for poor quality dataset"""
        from app.models.data_models import DatasetProfile
        
        profile = DatasetProfile(
            file_id=1,
            filename="test.csv",
            total_rows=50,  # Small dataset
            total_columns=5,
            memory_usage_mb=1.0,
            columns=[],
            missing_values_total=250,
            missing_values_percentage=50.0,  # High missing values
            duplicate_rows_count=10,
            duplicate_rows_percentage=20.0,  # High duplicates
            overall_quality_score=0.3,  # Poor quality
            quality_issues=['Many missing values'],
            recommended_targets=[]  # No good targets
        )
        
        recommendations = service._generate_recommendations(profile)
        
        # Should have quality warnings
        assert any('quality score is low' in rec for rec in recommendations)
        assert any('missing values' in rec for rec in recommendations)
        assert any('duplicate' in rec for rec in recommendations)
        assert any('small' in rec for rec in recommendations)

    @pytest.mark.asyncio
    async def test_get_dataset_preview(self, service, mock_db, sample_dataset):
        """Test dataset preview generation"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Update mock to point to temp file
            mock_db.get.return_value.storage_location = temp_file
            
            preview = await service.get_dataset_preview(1, mock_db, num_rows=3)
            
            assert preview.file_id == 1
            assert preview.filename == "test_dataset.csv"
            assert preview.total_rows == 3  # Limited by num_rows
            assert preview.total_columns == 5
            assert len(preview.sample_rows) == 3
            assert 'numeric_col' in preview.numeric_columns
            assert 'categorical_col' in preview.categorical_columns
            
        finally:
            os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_profile_dataset_file_not_found(self, service, mock_db):
        """Test profiling with non-existent file"""
        mock_db.get.return_value = None
        
        response = await service.profile_dataset(999, mock_db)
        
        assert not response.success
        assert "not found" in response.message.lower()
        assert response.error_details == "File not found in database"

    @pytest.mark.asyncio
    async def test_profile_dataset_not_a_dataset(self, service, mock_db):
        """Test profiling with non-dataset file"""
        mock_file = mock_db.get.return_value
        mock_file.is_dataset = False
        
        response = await service.profile_dataset(1, mock_db)
        
        assert not response.success
        assert "not a dataset" in response.message.lower()

    def test_load_dataset_csv(self, service, sample_dataset):
        """Test loading CSV dataset"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            df = service._load_dataset(temp_file)
            assert df.shape == sample_dataset.shape
            assert list(df.columns) == list(sample_dataset.columns)
        finally:
            os.unlink(temp_file)

    def test_load_dataset_excel(self, service, sample_dataset):
        """Test loading Excel dataset"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            sample_dataset.to_excel(f.name, index=False)
            temp_file = f.name

        try:
            df = service._load_dataset(temp_file)
            assert df.shape == sample_dataset.shape
            assert list(df.columns) == list(sample_dataset.columns)
        finally:
            os.unlink(temp_file)

    def test_load_dataset_with_sampling(self, service, sample_dataset):
        """Test loading dataset with row sampling"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            df = service._load_dataset(temp_file, sample_size=3)
            assert df.shape[0] == 3  # Should be limited to 3 rows
            assert df.shape[1] == sample_dataset.shape[1]  # Same number of columns
        finally:
            os.unlink(temp_file)

    def test_load_dataset_unsupported_format(self, service):
        """Test loading unsupported file format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some text content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                service._load_dataset(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_dataset_file_not_found(self, service):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            service._load_dataset("/nonexistent/file.csv")


if __name__ == "__main__":
    pytest.main([__file__]) 