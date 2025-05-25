#!/usr/bin/env python3
"""
Unit tests for DS1.2.3: Data Preprocessing Pipeline
Tests preprocessing functionality, Prefect tasks, and integration with algorithm registry
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from preprocessing import (
    PreprocessingConfig, PreprocessingResult, DataPreprocessor,
    load_and_validate_data, create_preprocessing_config,
    preprocess_dataset, validate_preprocessing_result,
    data_preprocessing_flow, get_preprocessing_recommendations,
    validate_preprocessing_config
)

from models.ml_models import ProblemTypeEnum, PreprocessingStepEnum


class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PreprocessingConfig()
        
        assert config.missing_strategy == "mean"
        assert config.missing_threshold == 0.8
        assert config.categorical_strategy == "onehot"
        assert config.max_categories == 20
        assert config.scaling_strategy == "standard"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.stratify == True
        assert config.feature_selection_method is None
        assert config.outlier_method == "none"
        assert config.min_samples_per_class == 5
        assert config.max_memory_mb == 1000
    
    def test_custom_config(self):
        """Test custom configuration creation"""
        config = PreprocessingConfig(
            missing_strategy="median",
            test_size=0.3,
            scaling_strategy="minmax",
            feature_selection_method="selectkbest",
            n_features_to_select=10
        )
        
        assert config.missing_strategy == "median"
        assert config.test_size == 0.3
        assert config.scaling_strategy == "minmax"
        assert config.feature_selection_method == "selectkbest"
        assert config.n_features_to_select == 10


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        np.random.seed(42)
        data = {
            'numerical_1': np.random.normal(0, 1, 100),
            'numerical_2': np.random.normal(5, 2, 100),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y'], 100),
            'target_numeric': np.random.normal(10, 3, 100),
            'target_categorical': np.random.choice(['class1', 'class2', 'class3'], 100)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce some missing values
        df.loc[0:5, 'numerical_1'] = np.nan
        df.loc[10:15, 'categorical_1'] = np.nan
        
        # Add some high cardinality column
        df['high_cardinality'] = [f'cat_{i}' for i in range(100)]
        
        return df
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        config = PreprocessingConfig()
        return DataPreprocessor(config)
    
    def test_analyze_data_quality(self, preprocessor, sample_dataframe):
        """Test data quality analysis"""
        analysis = preprocessor.analyze_data_quality(sample_dataframe, 'target_numeric')
        
        assert analysis['total_rows'] == 100
        assert analysis['total_columns'] == 7
        assert 'memory_usage_mb' in analysis
        assert len(analysis['missing_values']) > 0
        assert 'numerical_1' in analysis['missing_values']
        assert 'categorical_1' in analysis['missing_values']
        assert len(analysis['categorical_columns']) >= 2
        assert len(analysis['numerical_columns']) >= 2
        assert 'high_cardinality' in analysis['high_cardinality_columns']
        assert analysis['duplicate_rows'] == 0
    
    def test_handle_missing_values_mean_strategy(self, sample_dataframe):
        """Test missing value handling with mean strategy"""
        config = PreprocessingConfig(missing_strategy="mean")
        preprocessor = DataPreprocessor(config)
        
        result = preprocessor.handle_missing_values(sample_dataframe, 'target_numeric')
        
        # Should not have missing values in numerical columns
        assert result['numerical_1'].isnull().sum() == 0
        # Should have transformations stored
        assert 'numerical_imputer' in preprocessor.transformations
        assert 'categorical_imputer' in preprocessor.transformations
    
    def test_handle_missing_values_drop_strategy(self, sample_dataframe):
        """Test missing value handling with drop strategy"""
        config = PreprocessingConfig(missing_strategy="drop")
        preprocessor = DataPreprocessor(config)
        
        result = preprocessor.handle_missing_values(sample_dataframe, 'target_numeric')
        
        # Should have fewer rows due to dropping missing values
        assert len(result) < len(sample_dataframe)
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_knn_strategy(self, sample_dataframe):
        """Test missing value handling with KNN strategy"""
        config = PreprocessingConfig(missing_strategy="knn")
        preprocessor = DataPreprocessor(config)
        
        result = preprocessor.handle_missing_values(sample_dataframe, 'target_numeric')
        
        # Should not have missing values
        feature_cols = [col for col in result.columns if col != 'target_numeric']
        assert result[feature_cols].isnull().sum().sum() == 0
        assert 'knn_imputer' in preprocessor.transformations
    
    def test_encode_categorical_variables_onehot(self, sample_dataframe):
        """Test one-hot encoding of categorical variables"""
        config = PreprocessingConfig(categorical_strategy="onehot")
        preprocessor = DataPreprocessor(config)
        
        # First handle missing values
        df_clean = preprocessor.handle_missing_values(sample_dataframe, 'target_numeric')
        result = preprocessor.encode_categorical_variables(df_clean, 'target_numeric', ProblemTypeEnum.REGRESSION)
        
        # Should have more columns due to one-hot encoding
        assert result.shape[1] > sample_dataframe.shape[1]
        # Should have one-hot encoders stored
        onehot_transformers = [key for key in preprocessor.transformations.keys() if key.startswith('onehot_')]
        assert len(onehot_transformers) > 0
        # High cardinality column should be skipped
        assert 'high_cardinality' in result.columns  # Original column should remain
    
    def test_encode_categorical_variables_label(self, sample_dataframe):
        """Test label encoding of categorical variables"""
        config = PreprocessingConfig(categorical_strategy="label")
        preprocessor = DataPreprocessor(config)
        
        # First handle missing values
        df_clean = preprocessor.handle_missing_values(sample_dataframe, 'target_numeric')
        result = preprocessor.encode_categorical_variables(df_clean, 'target_numeric', ProblemTypeEnum.REGRESSION)
        
        # Should have encoded columns
        encoded_cols = [col for col in result.columns if col.endswith('_encoded')]
        assert len(encoded_cols) > 0
        # Should have label encoders stored
        label_transformers = [key for key in preprocessor.transformations.keys() if key.startswith('label_')]
        assert len(label_transformers) > 0
    
    def test_scale_features_standard(self, sample_dataframe):
        """Test standard scaling of features"""
        config = PreprocessingConfig(scaling_strategy="standard")
        preprocessor = DataPreprocessor(config)
        
        # Create train/test split manually
        X_train = sample_dataframe[['numerical_1', 'numerical_2']].iloc[:80]
        X_test = sample_dataframe[['numerical_1', 'numerical_2']].iloc[80:]
        
        # Fill missing values first
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use train mean
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Check that scaling was applied
        assert abs(X_train_scaled['numerical_1'].mean()) < 0.1  # Should be close to 0
        assert abs(X_train_scaled['numerical_1'].std() - 1.0) < 0.1  # Should be close to 1
        assert 'scaler' in preprocessor.transformations
    
    def test_scale_features_none(self, sample_dataframe):
        """Test skipping feature scaling"""
        config = PreprocessingConfig(scaling_strategy="none")
        preprocessor = DataPreprocessor(config)
        
        X_train = sample_dataframe[['numerical_1', 'numerical_2']].iloc[:80]
        X_test = sample_dataframe[['numerical_1', 'numerical_2']].iloc[80:]
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(X_train, X_train_scaled)
        pd.testing.assert_frame_equal(X_test, X_test_scaled)
        assert 'scaler' not in preprocessor.transformations
    
    def test_split_data_classification(self, sample_dataframe):
        """Test data splitting for classification"""
        config = PreprocessingConfig(test_size=0.3, stratify=True)
        preprocessor = DataPreprocessor(config)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            sample_dataframe, 'target_categorical', ProblemTypeEnum.CLASSIFICATION
        )
        
        assert len(X_train) == 70  # 70% of 100
        assert len(X_test) == 30   # 30% of 100
        assert len(y_train) == 70
        assert len(y_test) == 30
        
        # Check that features don't include target
        assert 'target_categorical' not in X_train.columns
        assert 'target_categorical' not in X_test.columns
    
    def test_split_data_regression(self, sample_dataframe):
        """Test data splitting for regression"""
        config = PreprocessingConfig(test_size=0.2)
        preprocessor = DataPreprocessor(config)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            sample_dataframe, 'target_numeric', ProblemTypeEnum.REGRESSION
        )
        
        assert len(X_train) == 80  # 80% of 100
        assert len(X_test) == 20   # 20% of 100
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_feature_selection_selectkbest(self, sample_dataframe):
        """Test feature selection with SelectKBest"""
        config = PreprocessingConfig(feature_selection_method="selectkbest", n_features_to_select=3)
        preprocessor = DataPreprocessor(config)
        
        # Prepare data
        X_train = sample_dataframe.drop(columns=['target_numeric']).fillna(0)
        X_test = X_train.copy()
        y_train = sample_dataframe['target_numeric']
        
        X_train_selected, X_test_selected = preprocessor.select_features(
            X_train, X_test, y_train, ProblemTypeEnum.REGRESSION
        )
        
        assert X_train_selected.shape[1] == 3
        assert X_test_selected.shape[1] == 3
        assert 'feature_selector' in preprocessor.transformations
    
    def test_preprocess_data_complete_pipeline(self, sample_dataframe):
        """Test complete preprocessing pipeline"""
        config = PreprocessingConfig(
            missing_strategy="mean",
            categorical_strategy="onehot",
            scaling_strategy="standard",
            test_size=0.2
        )
        preprocessor = DataPreprocessor(config)
        
        result = preprocessor.preprocess_data(sample_dataframe, 'target_numeric', ProblemTypeEnum.REGRESSION)
        
        assert isinstance(result, PreprocessingResult)
        assert result.X_train.shape[0] == 80  # 80% for training
        assert result.X_test.shape[0] == 20   # 20% for testing
        assert len(result.y_train) == 80
        assert len(result.y_test) == 20
        assert len(result.preprocessing_steps) > 0
        assert len(result.transformations) > 0
        assert len(result.feature_names) > 0
        assert 'total_rows' in result.preprocessing_summary


class TestPrefectTasks:
    """Test cases for Prefect tasks"""
    
    def test_load_and_validate_data_csv(self):
        """Test loading CSV data"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': ['A', 'B', 'C', 'D', 'E'],
                'target': [0, 1, 0, 1, 0]
            })
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Mock Prefect logger
            with patch('preprocessing.get_run_logger') as mock_logger:
                mock_logger.return_value = Mock()
                
                result = load_and_validate_data(csv_path, 'target')
                
                assert len(result) == 5
                assert 'target' in result.columns
                assert 'feature1' in result.columns
                assert 'feature2' in result.columns
        
        finally:
            os.unlink(csv_path)
    
    def test_load_and_validate_data_invalid_target(self):
        """Test loading data with invalid target column"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            with patch('preprocessing.get_run_logger') as mock_logger:
                mock_logger.return_value = Mock()
                
                with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
                    load_and_validate_data(csv_path, 'nonexistent')
        
        finally:
            os.unlink(csv_path)
    
    def test_create_preprocessing_config_with_algorithm_recommendations(self):
        """Test creating preprocessing config based on algorithm recommendations"""
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            # Mock algorithm registry
            with patch('preprocessing.get_algorithm_registry') as mock_registry:
                mock_algo_def = Mock()
                mock_algo_def.recommended_preprocessing = [
                    Mock(value='scale_features'),
                    Mock(value='encode_categorical'),
                    Mock(value='handle_missing')
                ]
                
                mock_registry_instance = Mock()
                mock_registry_instance.get_algorithm.return_value = mock_algo_def
                mock_registry.return_value = mock_registry_instance
                
                config = create_preprocessing_config(
                    ['logistic_regression'], 
                    'classification'
                )
                
                assert config.scaling_strategy == "standard"
                assert config.categorical_strategy == "onehot"
                assert config.missing_strategy == "mean"
    
    def test_create_preprocessing_config_with_custom_overrides(self):
        """Test creating preprocessing config with custom overrides"""
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('preprocessing.get_algorithm_registry') as mock_registry:
                mock_registry_instance = Mock()
                mock_registry_instance.get_algorithm.return_value = None
                mock_registry.return_value = mock_registry_instance
                
                custom_config = {
                    'missing_strategy': 'median',
                    'test_size': 0.3,
                    'scaling_strategy': 'minmax'
                }
                
                config = create_preprocessing_config(
                    ['random_forest'], 
                    'regression',
                    custom_config
                )
                
                assert config.missing_strategy == "median"
                assert config.test_size == 0.3
                assert config.scaling_strategy == "minmax"
    
    def test_preprocess_dataset_task(self):
        """Test preprocessing dataset task"""
        # Create sample data
        df = pd.DataFrame({
            'numerical': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        config = PreprocessingConfig(
            missing_strategy="mean",
            categorical_strategy="onehot",
            scaling_strategy="standard",
            test_size=0.2
        )
        
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            result = preprocess_dataset(df, 'target', 'classification', config)
            
            assert isinstance(result, PreprocessingResult)
            assert len(result.X_train) == 4  # 80% of 5
            assert len(result.X_test) == 1   # 20% of 5
            assert len(result.preprocessing_steps) > 0
    
    def test_validate_preprocessing_result_valid(self):
        """Test validation of valid preprocessing result"""
        # Create valid preprocessing result
        result = PreprocessingResult(
            X_train=pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
            X_test=pd.DataFrame({'feature1': [7], 'feature2': [8]}),
            y_train=pd.Series([0, 1, 0]),
            y_test=pd.Series([1]),
            preprocessing_steps=['missing_values_mean', 'categorical_onehot'],
            transformations={'scaler': Mock()},
            feature_names=['feature1', 'feature2'],
            preprocessing_summary={'total_rows': 4}
        )
        
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            validation = validate_preprocessing_result(result)
            
            assert validation['is_valid'] == True
            assert len(validation['errors']) == 0
            assert validation['metrics']['train_samples'] == 3
            assert validation['metrics']['test_samples'] == 1
            assert validation['metrics']['n_features'] == 2
    
    def test_validate_preprocessing_result_invalid(self):
        """Test validation of invalid preprocessing result"""
        # Create invalid preprocessing result (empty datasets)
        result = PreprocessingResult(
            X_train=pd.DataFrame(),  # Empty training set
            X_test=pd.DataFrame(),   # Empty test set
            y_train=pd.Series([]),
            y_test=pd.Series([]),
            preprocessing_steps=[],
            transformations={},
            feature_names=[],
            preprocessing_summary={}
        )
        
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            validation = validate_preprocessing_result(result)
            
            assert validation['is_valid'] == False
            assert len(validation['errors']) > 0
            assert "empty after preprocessing" in validation['errors'][0]


class TestDataPreprocessingFlow:
    """Test cases for the complete preprocessing flow"""
    
    def test_successful_preprocessing_flow(self):
        """Test successful execution of preprocessing flow"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'numerical': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'categorical': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
                'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            })
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            with patch('preprocessing.get_run_logger') as mock_logger:
                mock_logger.return_value = Mock()
                
                # Mock algorithm registry
                with patch('preprocessing.get_algorithm_registry') as mock_registry:
                    mock_algo_def = Mock()
                    mock_algo_def.recommended_preprocessing = [
                        Mock(value='scale_features'),
                        Mock(value='encode_categorical')
                    ]
                    
                    mock_registry_instance = Mock()
                    mock_registry_instance.get_algorithm.return_value = mock_algo_def
                    mock_registry.return_value = mock_registry_instance
                    
                    result = data_preprocessing_flow(
                        file_path=csv_path,
                        target_column='target',
                        problem_type='classification',
                        algorithm_names=['logistic_regression']
                    )
                    
                    assert result['success'] == True
                    assert result['preprocessing_result'] is not None
                    assert result['validation']['is_valid'] == True
                    assert 'summary' in result
                    assert result['summary']['original_shape'] == (10, 3)
        
        finally:
            os.unlink(csv_path)
    
    def test_failed_preprocessing_flow(self):
        """Test preprocessing flow with invalid input"""
        with patch('preprocessing.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            result = data_preprocessing_flow(
                file_path='nonexistent_file.csv',
                target_column='target',
                problem_type='classification',
                algorithm_names=['logistic_regression']
            )
            
            assert result['success'] == False
            assert 'error' in result
            assert result['preprocessing_result'] is None


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_get_preprocessing_recommendations(self):
        """Test getting preprocessing recommendations"""
        with patch('preprocessing.get_algorithm_registry') as mock_registry:
            # Mock algorithm definitions
            mock_algo_def_1 = Mock()
            mock_algo_def_1.recommended_preprocessing = [
                Mock(value='scale_features'),
                Mock(value='handle_missing')
            ]
            
            mock_algo_def_2 = Mock()
            mock_algo_def_2.recommended_preprocessing = [
                Mock(value='encode_categorical'),
                Mock(value='handle_missing')
            ]
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_algorithm.side_effect = [mock_algo_def_1, mock_algo_def_2]
            mock_registry.return_value = mock_registry_instance
            
            recommendations = get_preprocessing_recommendations(['svm_classifier', 'random_forest_classifier'])
            
            assert recommendations['scaling_needed'] == True
            assert recommendations['categorical_encoding_needed'] == True
            assert recommendations['missing_value_handling_needed'] == True
            assert 'scale_features' in recommendations['recommended_steps']
            assert 'encode_categorical' in recommendations['recommended_steps']
            assert 'handle_missing' in recommendations['recommended_steps']
    
    def test_get_preprocessing_recommendations_invalid_algorithms(self):
        """Test getting recommendations with invalid algorithms"""
        with patch('preprocessing.get_algorithm_registry') as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.get_algorithm.return_value = None
            mock_registry.return_value = mock_registry_instance
            
            recommendations = get_preprocessing_recommendations(['invalid_algorithm'])
            
            assert recommendations['scaling_needed'] == False
            assert recommendations['categorical_encoding_needed'] == False
            assert recommendations['missing_value_handling_needed'] == True
            assert len(recommendations['recommended_steps']) == 0
    
    def test_validate_preprocessing_config_valid(self):
        """Test validation of valid preprocessing configuration"""
        config_dict = {
            'missing_strategy': 'median',
            'categorical_strategy': 'label',
            'scaling_strategy': 'minmax',
            'test_size': 0.25
        }
        
        result = validate_preprocessing_config(config_dict)
        
        assert result['valid'] == True
        assert result['config'] is not None
        assert len(result['errors']) == 0
        assert result['config'].missing_strategy == 'median'
    
    def test_validate_preprocessing_config_invalid(self):
        """Test validation of invalid preprocessing configuration"""
        config_dict = {
            'missing_strategy': 'invalid_strategy',
            'test_size': 'not_a_number'
        }
        
        result = validate_preprocessing_config(config_dict)
        
        assert result['valid'] == False
        assert result['config'] is None
        assert len(result['errors']) > 0


class TestPreprocessingResult:
    """Test cases for PreprocessingResult dataclass"""
    
    def test_preprocessing_result_creation(self):
        """Test creating PreprocessingResult"""
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        X_test = pd.DataFrame({'feature1': [4, 5]})
        y_train = pd.Series([0, 1, 0])
        y_test = pd.Series([1, 0])
        
        result = PreprocessingResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessing_steps=['scaling', 'encoding'],
            transformations={'scaler': Mock()},
            feature_names=['feature1'],
            preprocessing_summary={'rows': 5}
        )
        
        assert result.X_train.shape == (3, 1)
        assert result.X_test.shape == (2, 1)
        assert len(result.y_train) == 3
        assert len(result.y_test) == 2
        assert len(result.preprocessing_steps) == 2
        assert 'scaler' in result.transformations
        assert result.feature_names == ['feature1']
        assert result.preprocessing_summary['rows'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 