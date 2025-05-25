#!/usr/bin/env python3
"""
Unit tests for DS1.3.1: ML Training Workflow
Tests ML training pipeline, model evaluation, and result aggregation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'pipelines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from ml_training import (
    TrainingResult, EvaluationResult, MLTrainingResult, MLModelTrainer,
    validate_ml_config, train_multiple_algorithms, evaluate_trained_models,
    aggregate_training_results, ml_training_flow,
    create_ml_training_config, get_algorithm_suggestions, validate_algorithm_config
)

from models.ml_models import ProblemTypeEnum, AlgorithmNameEnum
from preprocessing import PreprocessingResult


class TestTrainingDataClasses:
    """Test cases for training result data classes"""
    
    def test_training_result_creation(self):
        """Test creating TrainingResult"""
        result = TrainingResult(
            algorithm_name="logistic_regression",
            model=Mock(),
            training_time=1.5,
            hyperparameters={"C": 1.0},
            feature_importance={"feature1": 0.8, "feature2": 0.2},
            model_path="/path/to/model.joblib"
        )
        
        assert result.algorithm_name == "logistic_regression"
        assert result.training_time == 1.5
        assert result.hyperparameters["C"] == 1.0
        assert result.feature_importance["feature1"] == 0.8
        assert result.model_path == "/path/to/model.joblib"
        assert result.error is None
    
    def test_training_result_with_error(self):
        """Test creating TrainingResult with error"""
        result = TrainingResult(
            algorithm_name="svm_classifier",
            model=None,
            training_time=0.1,
            hyperparameters={},
            feature_importance=None,
            model_path=None,
            error="Training failed due to invalid hyperparameters"
        )
        
        assert result.error is not None
        assert result.model is None
        assert result.model_path is None
    
    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult"""
        metrics = {"accuracy": 0.85, "f1_score": 0.82, "precision": 0.80, "recall": 0.84}
        predictions = np.array([0, 1, 1, 0, 1])
        confusion_mat = np.array([[2, 1], [0, 2]])
        
        result = EvaluationResult(
            algorithm_name="random_forest_classifier",
            metrics=metrics,
            predictions=predictions,
            prediction_probabilities=None,
            confusion_matrix=confusion_mat,
            classification_report={"0": {"precision": 0.8}},
            feature_importance={"feature1": 0.6}
        )
        
        assert result.algorithm_name == "random_forest_classifier"
        assert result.metrics["accuracy"] == 0.85
        assert len(result.predictions) == 5
        assert result.confusion_matrix.shape == (2, 2)
        assert result.error is None


class TestMLModelTrainer:
    """Test cases for MLModelTrainer class"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y_train_classification = pd.Series(np.random.choice([0, 1], 100))
        y_train_regression = pd.Series(np.random.normal(10, 2, 100))
        
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 30),
            'feature2': np.random.normal(0, 1, 30),
            'feature3': np.random.normal(0, 1, 30)
        })
        y_test_classification = pd.Series(np.random.choice([0, 1], 30))
        y_test_regression = pd.Series(np.random.normal(10, 2, 30))
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train_classification': y_train_classification,
            'y_test_classification': y_test_classification,
            'y_train_regression': y_train_regression,
            'y_test_regression': y_test_regression
        }
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for saved models"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_trainer_initialization(self, temp_models_dir):
        """Test MLModelTrainer initialization"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        assert trainer.problem_type == ProblemTypeEnum.CLASSIFICATION
        assert trainer.models_save_dir == Path(temp_models_dir)
        assert len(trainer.algorithm_classes) == 10  # 5 classification + 5 regression
        assert trainer.logger is None
    
    def test_get_algorithm_hyperparameters(self, temp_models_dir):
        """Test getting algorithm hyperparameters"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        # Mock algorithm registry
        with patch.object(trainer, 'algorithm_registry') as mock_registry:
            mock_algo_def = Mock()
            mock_algo_def.default_hyperparameters = {"C": 1.0, "max_iter": 1000}
            mock_registry.get_algorithm.return_value = mock_algo_def
            
            # Test without custom parameters
            hyperparams = trainer.get_algorithm_hyperparameters(AlgorithmNameEnum.LOGISTIC_REGRESSION)
            assert hyperparams["C"] == 1.0
            assert hyperparams["max_iter"] == 1000
            
            # Test with custom parameters
            custom_params = {"C": 2.0, "penalty": "l2"}
            hyperparams = trainer.get_algorithm_hyperparameters(
                AlgorithmNameEnum.LOGISTIC_REGRESSION, custom_params
            )
            assert hyperparams["C"] == 2.0  # Custom override
            assert hyperparams["max_iter"] == 1000  # Default
            assert hyperparams["penalty"] == "l2"  # Custom addition
    
    def test_train_single_algorithm_classification_success(self, sample_training_data, temp_models_dir):
        """Test successful training of classification algorithm"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        hyperparams = {"C": 1.0, "max_iter": 100, "random_state": 42}
        
        result = trainer.train_single_algorithm(
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            sample_training_data['X_train'],
            sample_training_data['y_train_classification'],
            hyperparams,
            "test_pipeline_123"
        )
        
        assert result.error is None
        assert result.model is not None
        assert result.algorithm_name == "logistic_regression"
        assert result.training_time > 0
        assert result.hyperparameters == hyperparams
        assert result.feature_importance is not None  # Logistic regression has coefficients
        assert result.model_path is not None
        assert Path(result.model_path).exists()
    
    def test_train_single_algorithm_regression_success(self, sample_training_data, temp_models_dir):
        """Test successful training of regression algorithm"""
        trainer = MLModelTrainer(ProblemTypeEnum.REGRESSION, temp_models_dir)
        
        hyperparams = {"n_estimators": 10, "random_state": 42}
        
        result = trainer.train_single_algorithm(
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            sample_training_data['X_train'],
            sample_training_data['y_train_regression'],
            hyperparams,
            "test_pipeline_456"
        )
        
        assert result.error is None
        assert result.model is not None
        assert result.algorithm_name == "random_forest_regressor"
        assert result.training_time > 0
        assert result.feature_importance is not None  # Random forest has feature importances
        assert Path(result.model_path).exists()
    
    def test_train_single_algorithm_failure(self, sample_training_data, temp_models_dir):
        """Test training failure handling"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        # Invalid hyperparameters that should cause failure
        hyperparams = {"C": "invalid_value"}
        
        result = trainer.train_single_algorithm(
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            sample_training_data['X_train'],
            sample_training_data['y_train_classification'],
            hyperparams,
            "test_pipeline_fail"
        )
        
        assert result.error is not None
        assert result.model is None
        assert result.model_path is None
    
    def test_evaluate_model_classification_success(self, sample_training_data, temp_models_dir):
        """Test successful model evaluation for classification"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        # Create a mock training result with trained model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(sample_training_data['X_train'], sample_training_data['y_train_classification'])
        
        training_result = TrainingResult(
            algorithm_name="logistic_regression",
            model=model,
            training_time=1.0,
            hyperparameters={"C": 1.0},
            feature_importance={"feature1": 0.5, "feature2": 0.3, "feature3": 0.2},
            model_path="/mock/path"
        )
        
        evaluation_result = trainer.evaluate_model(
            training_result,
            sample_training_data['X_test'],
            sample_training_data['y_test_classification']
        )
        
        assert evaluation_result.error is None
        assert evaluation_result.algorithm_name == "logistic_regression"
        assert 'accuracy' in evaluation_result.metrics
        assert 'precision' in evaluation_result.metrics
        assert 'recall' in evaluation_result.metrics
        assert 'f1_score' in evaluation_result.metrics
        assert len(evaluation_result.predictions) == len(sample_training_data['y_test_classification'])
        assert evaluation_result.confusion_matrix is not None
        assert evaluation_result.classification_report is not None
    
    def test_evaluate_model_regression_success(self, sample_training_data, temp_models_dir):
        """Test successful model evaluation for regression"""
        trainer = MLModelTrainer(ProblemTypeEnum.REGRESSION, temp_models_dir)
        
        # Create a mock training result with trained model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(sample_training_data['X_train'], sample_training_data['y_train_regression'])
        
        training_result = TrainingResult(
            algorithm_name="linear_regression",
            model=model,
            training_time=1.0,
            hyperparameters={},
            feature_importance={"feature1": 0.4, "feature2": 0.3, "feature3": 0.3},
            model_path="/mock/path"
        )
        
        evaluation_result = trainer.evaluate_model(
            training_result,
            sample_training_data['X_test'],
            sample_training_data['y_test_regression']
        )
        
        assert evaluation_result.error is None
        assert evaluation_result.algorithm_name == "linear_regression"
        assert 'mae' in evaluation_result.metrics
        assert 'mse' in evaluation_result.metrics
        assert 'rmse' in evaluation_result.metrics
        assert 'r2' in evaluation_result.metrics
        assert 'mean_absolute_percentage_error' in evaluation_result.metrics
        assert len(evaluation_result.predictions) == len(sample_training_data['y_test_regression'])
    
    def test_evaluate_model_with_training_error(self, sample_training_data, temp_models_dir):
        """Test evaluation with training error"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        training_result = TrainingResult(
            algorithm_name="failed_algorithm",
            model=None,
            training_time=0.1,
            hyperparameters={},
            feature_importance=None,
            model_path=None,
            error="Training failed"
        )
        
        evaluation_result = trainer.evaluate_model(
            training_result,
            sample_training_data['X_test'],
            sample_training_data['y_test_classification']
        )
        
        assert evaluation_result.error is not None
        assert evaluation_result.algorithm_name == "failed_algorithm"
        assert len(evaluation_result.metrics) == 0
        assert len(evaluation_result.predictions) == 0
    
    def test_find_best_model_classification(self, temp_models_dir):
        """Test finding best model for classification"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        evaluation_results = [
            EvaluationResult(
                algorithm_name="model_1",
                metrics={"accuracy": 0.8, "f1_score": 0.75},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            ),
            EvaluationResult(
                algorithm_name="model_2",
                metrics={"accuracy": 0.85, "f1_score": 0.82},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            ),
            EvaluationResult(
                algorithm_name="model_3",
                metrics={"accuracy": 0.7, "f1_score": 0.68},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            )
        ]
        
        best_model = trainer.find_best_model(evaluation_results)
        
        assert best_model["algorithm_name"] == "model_2"
        assert best_model["primary_metric"] == "f1_score"
        assert best_model["primary_metric_value"] == 0.82
    
    def test_find_best_model_regression(self, temp_models_dir):
        """Test finding best model for regression"""
        trainer = MLModelTrainer(ProblemTypeEnum.REGRESSION, temp_models_dir)
        
        evaluation_results = [
            EvaluationResult(
                algorithm_name="model_1",
                metrics={"mae": 2.5, "r2": 0.75},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            ),
            EvaluationResult(
                algorithm_name="model_2",
                metrics={"mae": 1.8, "r2": 0.85},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            )
        ]
        
        best_model = trainer.find_best_model(evaluation_results)
        
        assert best_model["algorithm_name"] == "model_2"
        assert best_model["primary_metric"] == "r2"
        assert best_model["primary_metric_value"] == 0.85
    
    def test_find_best_model_no_valid_results(self, temp_models_dir):
        """Test finding best model with no valid results"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        evaluation_results = [
            EvaluationResult(
                algorithm_name="failed_model",
                metrics={},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None,
                error="Training failed"
            )
        ]
        
        best_model = trainer.find_best_model(evaluation_results)
        
        assert best_model["algorithm_name"] is None
        assert "error" in best_model
    
    def test_aggregate_results(self, temp_models_dir):
        """Test aggregating evaluation results"""
        trainer = MLModelTrainer(ProblemTypeEnum.CLASSIFICATION, temp_models_dir)
        
        evaluation_results = [
            EvaluationResult(
                algorithm_name="model_1",
                metrics={"accuracy": 0.8, "f1_score": 0.75},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            ),
            EvaluationResult(
                algorithm_name="model_2",
                metrics={"accuracy": 0.85, "f1_score": 0.82},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None
            ),
            EvaluationResult(
                algorithm_name="failed_model",
                metrics={},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None,
                error="Training failed"
            )
        ]
        
        aggregated = trainer.aggregate_results(evaluation_results)
        
        assert aggregated["models_trained"] == 3
        assert aggregated["models_successful"] == 2
        assert aggregated["models_failed"] == 1
        assert "metrics_summary" in aggregated
        assert "accuracy" in aggregated["metrics_summary"]
        assert "f1_score" in aggregated["metrics_summary"]
        
        # Check accuracy statistics
        accuracy_stats = aggregated["metrics_summary"]["accuracy"]
        assert accuracy_stats["mean"] == 0.825  # (0.8 + 0.85) / 2
        assert accuracy_stats["min"] == 0.8
        assert accuracy_stats["max"] == 0.85
        assert accuracy_stats["count"] == 2


class TestPrefectTasks:
    """Test cases for Prefect tasks"""
    
    def test_validate_ml_config_valid(self):
        """Test ML config validation with valid configuration"""
        config = {
            "file_path": "/path/to/data.csv",
            "target_column": "target",
            "problem_type": "classification",
            "algorithms": [
                {"name": "logistic_regression", "hyperparameters": {"C": 1.0}},
                {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}}
            ]
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.get_algorithm_registry') as mock_registry:
                mock_algo_def = Mock()
                mock_registry_instance = Mock()
                mock_registry_instance.get_algorithm.return_value = mock_algo_def
                mock_registry.return_value = mock_registry_instance
                
                result = validate_ml_config(config)
                
                assert result["is_valid"] == True
                assert len(result["errors"]) == 0
                assert "pipeline_run_id" in result["validated_config"]
                assert len(result["validated_config"]["algorithms"]) == 2
    
    def test_validate_ml_config_missing_fields(self):
        """Test ML config validation with missing required fields"""
        config = {
            "target_column": "target",
            "problem_type": "classification"
            # Missing file_path and algorithms
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            result = validate_ml_config(config)
            
            assert result["is_valid"] == False
            assert "Missing required field: file_path" in result["errors"]
            assert "Missing required field: algorithms" in result["errors"]
    
    def test_validate_ml_config_invalid_problem_type(self):
        """Test ML config validation with invalid problem type"""
        config = {
            "file_path": "/path/to/data.csv",
            "target_column": "target",
            "problem_type": "invalid_type",
            "algorithms": [{"name": "logistic_regression"}]
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            result = validate_ml_config(config)
            
            assert result["is_valid"] == False
            assert any("Invalid problem_type" in error for error in result["errors"])
    
    def test_train_multiple_algorithms_task(self):
        """Test training multiple algorithms task"""
        # Create mock preprocessing result
        preprocessing_result = PreprocessingResult(
            X_train=pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
            X_test=pd.DataFrame({'feature1': [7], 'feature2': [8]}),
            y_train=pd.Series([0, 1, 0]),
            y_test=pd.Series([1]),
            preprocessing_steps=['scaling'],
            transformations={},
            feature_names=['feature1', 'feature2'],
            preprocessing_summary={}
        )
        
        algorithms_config = [
            {"name": "logistic_regression", "hyperparameters": {"C": 1.0, "max_iter": 100}}
        ]
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.MLModelTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_training_result = TrainingResult(
                    algorithm_name="logistic_regression",
                    model=Mock(),
                    training_time=1.0,
                    hyperparameters={"C": 1.0},
                    feature_importance={"feature1": 0.6, "feature2": 0.4},
                    model_path="/mock/path"
                )
                
                mock_trainer.get_algorithm_hyperparameters.return_value = {"C": 1.0, "max_iter": 100}
                mock_trainer.train_single_algorithm.return_value = mock_training_result
                mock_trainer_class.return_value = mock_trainer
                
                results = train_multiple_algorithms(
                    preprocessing_result,
                    algorithms_config,
                    "classification",
                    "test_pipeline_123"
                )
                
                assert len(results) == 1
                assert results[0].algorithm_name == "logistic_regression"
                assert results[0].error is None
    
    def test_evaluate_trained_models_task(self):
        """Test evaluating trained models task"""
        # Create mock preprocessing result
        preprocessing_result = PreprocessingResult(
            X_train=pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
            X_test=pd.DataFrame({'feature1': [7], 'feature2': [8]}),
            y_train=pd.Series([0, 1, 0]),
            y_test=pd.Series([1]),
            preprocessing_steps=['scaling'],
            transformations={},
            feature_names=['feature1', 'feature2'],
            preprocessing_summary={}
        )
        
        training_results = [
            TrainingResult(
                algorithm_name="logistic_regression",
                model=Mock(),
                training_time=1.0,
                hyperparameters={"C": 1.0},
                feature_importance={"feature1": 0.6, "feature2": 0.4},
                model_path="/mock/path"
            )
        ]
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.MLModelTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_evaluation_result = EvaluationResult(
                    algorithm_name="logistic_regression",
                    metrics={"accuracy": 0.8, "f1_score": 0.75},
                    predictions=np.array([1]),
                    prediction_probabilities=None,
                    confusion_matrix=np.array([[0, 0], [0, 1]]),
                    classification_report={"1": {"precision": 1.0}},
                    feature_importance={"feature1": 0.6, "feature2": 0.4}
                )
                
                mock_trainer.evaluate_model.return_value = mock_evaluation_result
                mock_trainer_class.return_value = mock_trainer
                
                results = evaluate_trained_models(
                    training_results,
                    preprocessing_result,
                    "classification"
                )
                
                assert len(results) == 1
                assert results[0].algorithm_name == "logistic_regression"
                assert results[0].error is None
                assert results[0].metrics["accuracy"] == 0.8


class TestMLTrainingFlow:
    """Test cases for the complete ML training flow"""
    
    def test_successful_ml_training_flow(self):
        """Test successful execution of ML training flow"""
        config = {
            "file_path": "/path/to/data.csv",
            "target_column": "target",
            "problem_type": "classification",
            "algorithms": [
                {"name": "logistic_regression", "hyperparameters": {"C": 1.0}}
            ],
            "pipeline_run_id": "test_run_123"
        }
        
        # Mock preprocessing flow result
        mock_preprocessing_result = PreprocessingResult(
            X_train=pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
            X_test=pd.DataFrame({'feature1': [7], 'feature2': [8]}),
            y_train=pd.Series([0, 1, 0]),
            y_test=pd.Series([1]),
            preprocessing_steps=['scaling'],
            transformations={},
            feature_names=['feature1', 'feature2'],
            preprocessing_summary={}
        )
        
        mock_preprocessing_flow_result = {
            "success": True,
            "preprocessing_result": mock_preprocessing_result
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.validate_ml_config') as mock_validate:
                mock_validate.return_value = {
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "validated_config": config
                }
                
                with patch('ml_training.data_preprocessing_flow') as mock_preprocessing:
                    mock_preprocessing.return_value = mock_preprocessing_flow_result
                    
                    with patch('ml_training.train_multiple_algorithms') as mock_train:
                        mock_training_results = [
                            TrainingResult(
                                algorithm_name="logistic_regression",
                                model=Mock(),
                                training_time=1.0,
                                hyperparameters={"C": 1.0},
                                feature_importance={"feature1": 0.6, "feature2": 0.4},
                                model_path="/mock/path"
                            )
                        ]
                        mock_train.return_value = mock_training_results
                        
                        with patch('ml_training.evaluate_trained_models') as mock_evaluate:
                            mock_evaluation_results = [
                                EvaluationResult(
                                    algorithm_name="logistic_regression",
                                    metrics={"accuracy": 0.8, "f1_score": 0.75},
                                    predictions=np.array([1]),
                                    prediction_probabilities=None,
                                    confusion_matrix=np.array([[0, 0], [0, 1]]),
                                    classification_report={"1": {"precision": 1.0}},
                                    feature_importance={"feature1": 0.6, "feature2": 0.4}
                                )
                            ]
                            mock_evaluate.return_value = mock_evaluation_results
                            
                            with patch('ml_training.aggregate_training_results') as mock_aggregate:
                                mock_final_result = MLTrainingResult(
                                    pipeline_run_id="test_run_123",
                                    problem_type="classification",
                                    target_variable="target",
                                    preprocessing_result=mock_preprocessing_result,
                                    training_results=mock_training_results,
                                    evaluation_results=mock_evaluation_results,
                                    best_model={"algorithm_name": "logistic_regression"},
                                    aggregated_metrics={"models_successful": 1},
                                    total_training_time=1.0,
                                    summary={"algorithms_successful": 1}
                                )
                                mock_aggregate.return_value = mock_final_result
                                
                                result = ml_training_flow(config)
                                
                                assert result["success"] == True
                                assert result["result"].pipeline_run_id == "test_run_123"
                                assert result["result"].problem_type == "classification"
                                assert result["config_used"] == config
    
    def test_ml_training_flow_validation_failure(self):
        """Test ML training flow with validation failure"""
        config = {
            "target_column": "target",
            # Missing required fields
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.validate_ml_config') as mock_validate:
                mock_validate.return_value = {
                    "is_valid": False,
                    "errors": ["Missing required field: file_path"],
                    "warnings": [],
                    "validated_config": config
                }
                
                result = ml_training_flow(config)
                
                assert result["success"] == False
                assert "Configuration validation failed" in result["error"]
    
    def test_ml_training_flow_preprocessing_failure(self):
        """Test ML training flow with preprocessing failure"""
        config = {
            "file_path": "/path/to/data.csv",
            "target_column": "target",
            "problem_type": "classification",
            "algorithms": [{"name": "logistic_regression"}]
        }
        
        with patch('ml_training.get_run_logger') as mock_logger:
            mock_logger.return_value = Mock()
            
            with patch('ml_training.validate_ml_config') as mock_validate:
                mock_validate.return_value = {
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "validated_config": config
                }
                
                with patch('ml_training.data_preprocessing_flow') as mock_preprocessing:
                    mock_preprocessing.return_value = {
                        "success": False,
                        "error": "Preprocessing failed"
                    }
                    
                    result = ml_training_flow(config)
                    
                    assert result["success"] == False
                    assert "Preprocessing failed" in result["error"]


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_create_ml_training_config(self):
        """Test creating ML training configuration"""
        algorithms = [
            {"name": "logistic_regression", "hyperparameters": {"C": 1.0}},
            {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}}
        ]
        
        config = create_ml_training_config(
            file_path="/path/to/data.csv",
            target_column="target",
            problem_type="classification",
            algorithms=algorithms,
            preprocessing_config={"scaling_strategy": "standard"},
            pipeline_run_id="custom_run_id"
        )
        
        assert config["file_path"] == "/path/to/data.csv"
        assert config["target_column"] == "target"
        assert config["problem_type"] == "classification"
        assert config["algorithms"] == algorithms
        assert config["preprocessing_config"]["scaling_strategy"] == "standard"
        assert config["pipeline_run_id"] == "custom_run_id"
    
    def test_create_ml_training_config_defaults(self):
        """Test creating ML training configuration with defaults"""
        algorithms = [{"name": "logistic_regression"}]
        
        config = create_ml_training_config(
            file_path="/path/to/data.csv",
            target_column="target",
            problem_type="regression",
            algorithms=algorithms
        )
        
        assert config["preprocessing_config"] == {}
        assert config["pipeline_run_id"].startswith("ml_run_")
    
    def test_get_algorithm_suggestions(self):
        """Test getting algorithm suggestions"""
        with patch('ml_training.get_algorithm_registry') as mock_registry:
            mock_algo_def = Mock()
            mock_algo_def.problem_type = ProblemTypeEnum.CLASSIFICATION
            mock_algo_def.display_name = "Logistic Regression"
            mock_algo_def.description = "Linear classifier"
            mock_algo_def.default_hyperparameters = {"C": 1.0}
            mock_algo_def.complexity = Mock()
            mock_algo_def.complexity.value = "low"
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_algorithm.return_value = mock_algo_def
            mock_registry.return_value = mock_registry_instance
            
            suggestions = get_algorithm_suggestions("classification")
            
            assert len(suggestions) == 10  # All algorithms checked
            # Each suggestion should be validated
            for suggestion in suggestions:
                if suggestion["name"] == "logistic_regression":
                    assert suggestion["display_name"] == "Logistic Regression"
                    assert suggestion["description"] == "Linear classifier"
                    assert suggestion["default_hyperparameters"]["C"] == 1.0
                    assert suggestion["complexity"] == "low"
    
    def test_validate_algorithm_config_valid(self):
        """Test validating valid algorithm configuration"""
        algorithm_config = {
            "name": "logistic_regression",
            "hyperparameters": {"C": 1.0, "max_iter": 1000}
        }
        
        with patch('ml_training.get_algorithm_registry') as mock_registry:
            mock_algo_def = Mock()
            mock_registry_instance = Mock()
            mock_registry_instance.get_algorithm.return_value = mock_algo_def
            mock_registry.return_value = mock_registry_instance
            
            result = validate_algorithm_config(algorithm_config)
            
            assert result["valid"] == True
            assert result["algorithm_def"] == mock_algo_def
    
    def test_validate_algorithm_config_invalid_name(self):
        """Test validating algorithm configuration with invalid name"""
        algorithm_config = {
            "name": "invalid_algorithm",
            "hyperparameters": {"param": "value"}
        }
        
        result = validate_algorithm_config(algorithm_config)
        
        assert result["valid"] == False
        assert "error" in result
    
    def test_validate_algorithm_config_invalid_hyperparams(self):
        """Test validating algorithm configuration with invalid hyperparameters"""
        algorithm_config = {
            "name": "logistic_regression",
            "hyperparameters": "invalid_format"  # Should be dict
        }
        
        with patch('ml_training.get_algorithm_registry') as mock_registry:
            mock_algo_def = Mock()
            mock_registry_instance = Mock()
            mock_registry_instance.get_algorithm.return_value = mock_algo_def
            mock_registry.return_value = mock_registry_instance
            
            result = validate_algorithm_config(algorithm_config)
            
            assert result["valid"] == False
            assert "Hyperparameters must be a dictionary" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 