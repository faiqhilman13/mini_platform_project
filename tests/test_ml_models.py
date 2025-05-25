#!/usr/bin/env python3
"""
Unit tests for DS1.2.1: ML Pipeline Models
Tests all ML model classes, validation, and relationships
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from models.ml_models import (
    # Enums
    ProblemTypeEnum, AlgorithmNameEnum, PreprocessingStepEnum, MetricNameEnum,
    
    # Configuration models
    AlgorithmConfig, PreprocessingConfig, MLPipelineConfig,
    
    # Result models
    ModelMetrics, ModelArtifacts, MLResult,
    
    # Response models
    MLPipelineConfigResponse, MLPipelineRunResponse, MLResultsResponse,
    ModelComparisonResponse,
    
    # Request models
    MLPipelineStartRequest, ModelPredictionRequest, HyperparameterTuningRequest
)


class TestEnums:
    """Test all enum definitions"""
    
    def test_problem_type_enum(self):
        """Test ProblemTypeEnum values"""
        assert ProblemTypeEnum.CLASSIFICATION == "classification"
        assert ProblemTypeEnum.REGRESSION == "regression"
        
        # Test enum can be used in comparisons
        assert ProblemTypeEnum.CLASSIFICATION != ProblemTypeEnum.REGRESSION
        
        # Test enum values list
        problem_types = list(ProblemTypeEnum)
        assert len(problem_types) == 2
        assert ProblemTypeEnum.CLASSIFICATION in problem_types
        assert ProblemTypeEnum.REGRESSION in problem_types
    
    def test_algorithm_name_enum(self):
        """Test AlgorithmNameEnum values"""
        # Classification algorithms
        classification_algos = [
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_CLASSIFIER,
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            AlgorithmNameEnum.SVM_CLASSIFIER,
            AlgorithmNameEnum.KNN_CLASSIFIER
        ]
        
        for algo in classification_algos:
            assert "classifier" in algo.value or "regression" in algo.value
        
        # Regression algorithms
        regression_algos = [
            AlgorithmNameEnum.LINEAR_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_REGRESSOR,
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            AlgorithmNameEnum.SVM_REGRESSOR,
            AlgorithmNameEnum.KNN_REGRESSOR
        ]
        
        for algo in regression_algos:
            assert "regressor" in algo.value or "regression" in algo.value
        
        # Test total count
        all_algos = list(AlgorithmNameEnum)
        assert len(all_algos) == 10  # 5 classification + 5 regression
    
    def test_metric_name_enum(self):
        """Test MetricNameEnum values"""
        classification_metrics = [
            MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION,
            MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC
        ]
        
        regression_metrics = [
            MetricNameEnum.MAE, MetricNameEnum.MSE,
            MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
        ]
        
        assert len(classification_metrics) == 5
        assert len(regression_metrics) == 4
        
        # Test specific values
        assert MetricNameEnum.ACCURACY.value == "accuracy"
        assert MetricNameEnum.R2_SCORE.value == "r2_score"
    
    def test_preprocessing_step_enum(self):
        """Test PreprocessingStepEnum values"""
        steps = [
            PreprocessingStepEnum.HANDLE_MISSING,
            PreprocessingStepEnum.ENCODE_CATEGORICAL,
            PreprocessingStepEnum.SCALE_FEATURES,
            PreprocessingStepEnum.REMOVE_OUTLIERS,
            PreprocessingStepEnum.FEATURE_SELECTION
        ]
        
        assert len(steps) == 5
        assert PreprocessingStepEnum.HANDLE_MISSING.value == "handle_missing"


class TestAlgorithmConfig:
    """Test AlgorithmConfig model"""
    
    def test_basic_creation(self):
        """Test basic algorithm config creation"""
        config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        )
        
        assert config.algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        assert config.hyperparameters == {}
        assert config.is_enabled == True
        assert config.cross_validation_folds == 5
        assert config.random_state == 42
        assert config.max_training_time_minutes == 30
    
    def test_with_hyperparameters(self):
        """Test algorithm config with custom hyperparameters"""
        hyperparams = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "bootstrap": True
        }
        
        config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            hyperparameters=hyperparams,
            cross_validation_folds=10,
            max_training_time_minutes=60
        )
        
        assert config.hyperparameters == hyperparams
        assert config.cross_validation_folds == 10
        assert config.max_training_time_minutes == 60
    
    def test_validation_constraints(self):
        """Test field validation constraints"""
        # Valid values should work
        config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            cross_validation_folds=3,
            max_training_time_minutes=120
        )
        assert config.cross_validation_folds == 3
        
        # Test invalid values raise validation errors
        with pytest.raises(ValueError):
            AlgorithmConfig(
                algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
                cross_validation_folds=1  # Below minimum of 2
            )
        
        with pytest.raises(ValueError):
            AlgorithmConfig(
                algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
                max_training_time_minutes=0  # Below minimum of 1
            )
    
    def test_json_serialization(self):
        """Test JSON serialization"""
        config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.SVM_CLASSIFIER,
            hyperparameters={"C": 1.0, "kernel": "rbf"}
        )
        
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["algorithm_name"] == AlgorithmNameEnum.SVM_CLASSIFIER
        assert config_dict["hyperparameters"]["C"] == 1.0


class TestPreprocessingConfig:
    """Test PreprocessingConfig model"""
    
    def test_default_creation(self):
        """Test default preprocessing config creation"""
        config = PreprocessingConfig()
        
        assert config.steps == []
        assert config.missing_value_strategy == "mean"
        assert config.missing_value_threshold == 0.5
        assert config.categorical_encoding == "onehot"
        assert config.max_categories == 20
        assert config.scaling_method == "standard"
        assert config.outlier_method == "zscore"
        assert config.outlier_threshold == 3.0
        assert config.test_size == 0.2
        assert config.stratify == True
    
    def test_custom_config(self):
        """Test custom preprocessing configuration"""
        steps = [
            PreprocessingStepEnum.HANDLE_MISSING,
            PreprocessingStepEnum.ENCODE_CATEGORICAL,
            PreprocessingStepEnum.SCALE_FEATURES
        ]
        
        config = PreprocessingConfig(
            steps=steps,
            missing_value_strategy="median",
            categorical_encoding="label",
            scaling_method="minmax",
            test_size=0.3,
            outlier_threshold=2.5
        )
        
        assert config.steps == steps
        assert config.missing_value_strategy == "median"
        assert config.categorical_encoding == "label"
        assert config.scaling_method == "minmax"
        assert config.test_size == 0.3
        assert config.outlier_threshold == 2.5
    
    def test_validation_constraints(self):
        """Test preprocessing config validation"""
        # Valid values
        config = PreprocessingConfig(
            missing_value_threshold=0.3,
            test_size=0.25,
            outlier_threshold=2.0
        )
        assert config.missing_value_threshold == 0.3
        
        # Invalid values should raise errors
        with pytest.raises(ValueError):
            PreprocessingConfig(test_size=0.05)  # Below minimum 0.1
        
        with pytest.raises(ValueError):
            PreprocessingConfig(test_size=0.6)   # Above maximum 0.5
        
        with pytest.raises(ValueError):
            PreprocessingConfig(outlier_threshold=0.5)  # Below minimum 1.0


class TestMLPipelineConfig:
    """Test MLPipelineConfig model"""
    
    def test_basic_creation(self):
        """Test basic ML pipeline config creation"""
        config = MLPipelineConfig(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target_col"
        )
        
        assert config.problem_type == ProblemTypeEnum.CLASSIFICATION
        assert config.target_variable == "target_col"
        assert config.feature_variables == []
        assert config.algorithms == []
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert config.evaluation_metrics == []
        assert config.cross_validation == True
        assert config.max_total_training_time_minutes == 120
        assert config.parallel_jobs == 1
    
    def test_complete_config(self):
        """Test complete ML pipeline configuration"""
        # Create algorithm configs
        rf_config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            hyperparameters={"n_estimators": 100}
        )
        
        lr_config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            hyperparameters={"C": 1.0}
        )
        
        # Create preprocessing config
        preprocessing = PreprocessingConfig(
            missing_value_strategy="mean",
            categorical_encoding="onehot"
        )
        
        # Create complete config
        config = MLPipelineConfig(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="churn",
            feature_variables=["age", "tenure", "monthly_charges"],
            algorithms=[rf_config, lr_config],
            preprocessing=preprocessing,
            evaluation_metrics=[
                MetricNameEnum.ACCURACY,
                MetricNameEnum.PRECISION,
                MetricNameEnum.RECALL
            ],
            pipeline_name="Churn Prediction",
            description="Customer churn prediction pipeline",
            max_total_training_time_minutes=180
        )
        
        assert config.target_variable == "churn"
        assert len(config.feature_variables) == 3
        assert len(config.algorithms) == 2
        assert len(config.evaluation_metrics) == 3
        assert config.pipeline_name == "Churn Prediction"
        assert config.max_total_training_time_minutes == 180
    
    def test_validation_constraints(self):
        """Test ML pipeline config validation"""
        # Valid config
        config = MLPipelineConfig(
            problem_type=ProblemTypeEnum.REGRESSION,
            target_variable="price",
            max_total_training_time_minutes=60,
            parallel_jobs=2
        )
        assert config.max_total_training_time_minutes == 60
        assert config.parallel_jobs == 2
        
        # Invalid values
        with pytest.raises(ValueError):
            MLPipelineConfig(
                problem_type=ProblemTypeEnum.CLASSIFICATION,
                target_variable="target",
                max_total_training_time_minutes=5  # Below minimum 10
            )
        
        with pytest.raises(ValueError):
            MLPipelineConfig(
                problem_type=ProblemTypeEnum.CLASSIFICATION,
                target_variable="target",
                parallel_jobs=0  # Below minimum 1
            )


class TestModelResults:
    """Test model result classes"""
    
    def test_model_metrics(self):
        """Test ModelMetrics creation"""
        metric = ModelMetrics(
            metric_name=MetricNameEnum.ACCURACY,
            value=0.85,
            std_dev=0.02,
            confidence_interval=[0.83, 0.87],
            dataset_split="test",
            fold_number=1
        )
        
        assert metric.metric_name == MetricNameEnum.ACCURACY
        assert metric.value == 0.85
        assert metric.std_dev == 0.02
        assert metric.confidence_interval == [0.83, 0.87]
        assert metric.dataset_split == "test"
        assert metric.fold_number == 1
    
    def test_model_artifacts(self):
        """Test ModelArtifacts creation"""
        artifacts = ModelArtifacts(
            model_file_path="/models/model_123.pkl",
            feature_importance_path="/plots/importance_123.png",
            confusion_matrix_path="/plots/confusion_123.png",
            model_size_mb=3.2,
            training_data_shape=[1000, 20],
            feature_names=["feature1", "feature2", "feature3"],
            sklearn_version="1.0.2",
            python_version="3.9.0"
        )
        
        assert artifacts.model_file_path == "/models/model_123.pkl"
        assert artifacts.model_size_mb == 3.2
        assert artifacts.training_data_shape == [1000, 20]
        assert len(artifacts.feature_names) == 3
        assert artifacts.sklearn_version == "1.0.2"
    
    def test_ml_result(self):
        """Test MLResult creation"""
        # Create metrics
        metrics = [
            ModelMetrics(metric_name=MetricNameEnum.ACCURACY, value=0.85),
            ModelMetrics(metric_name=MetricNameEnum.F1_SCORE, value=0.82)
        ]
        
        # Create artifacts
        artifacts = ModelArtifacts(
            model_file_path="/models/rf_model.pkl",
            model_size_mb=2.1
        )
        
        # Create result
        result = MLResult(
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            metrics=metrics,
            primary_metric_value=0.85,
            primary_metric_name="accuracy",
            training_time_seconds=42.5,
            feature_importance={"feature1": 0.4, "feature2": 0.35, "feature3": 0.25},
            artifacts=artifacts
        )
        
        assert result.algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        assert len(result.metrics) == 2
        assert result.primary_metric_value == 0.85
        assert result.training_time_seconds == 42.5
        assert len(result.feature_importance) == 3
        assert result.training_status == "completed"
        assert result.is_best_model == False


class TestResponseModels:
    """Test API response models"""
    
    def test_ml_pipeline_config_response(self):
        """Test MLPipelineConfigResponse"""
        config = MLPipelineConfig(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target"
        )
        
        response = MLPipelineConfigResponse(
            success=True,
            message="Configuration validated successfully",
            config=config,
            validation_errors=[]
        )
        
        assert response.success == True
        assert response.config.problem_type == ProblemTypeEnum.CLASSIFICATION
        assert len(response.validation_errors) == 0
    
    def test_ml_pipeline_run_response(self):
        """Test MLPipelineRunResponse"""
        response = MLPipelineRunResponse(
            success=True,
            message="Pipeline started successfully",
            run_id="12345",
            status="RUNNING",
            progress={"completed_models": 2, "total_models": 5},
            estimated_completion_time=datetime.utcnow()
        )
        
        assert response.success == True
        assert response.run_id == "12345"
        assert response.status == "RUNNING"
        assert response.progress["completed_models"] == 2
    
    def test_ml_results_response(self):
        """Test MLResultsResponse"""
        # Create a sample result
        result = MLResult(
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            hyperparameters={"C": 1.0},
            metrics=[ModelMetrics(metric_name=MetricNameEnum.ACCURACY, value=0.8)],
            primary_metric_value=0.8,
            primary_metric_name="accuracy",
            training_time_seconds=30.0
        )
        
        response = MLResultsResponse(
            success=True,
            message="Training completed successfully",
            total_models=3,
            completed_models=3,
            failed_models=0,
            best_model=result,
            results=[result],
            total_training_time_seconds=150.0,
            data_quality_warnings=["High missing values in column X"]
        )
        
        assert response.success == True
        assert response.total_models == 3
        assert response.completed_models == 3
        assert response.failed_models == 0
        assert response.best_model.algorithm_name == AlgorithmNameEnum.LOGISTIC_REGRESSION
        assert len(response.results) == 1
        assert len(response.data_quality_warnings) == 1


class TestRequestModels:
    """Test API request models"""
    
    def test_ml_pipeline_start_request(self):
        """Test MLPipelineStartRequest"""
        config = MLPipelineConfig(
            problem_type=ProblemTypeEnum.REGRESSION,
            target_variable="price"
        )
        
        request = MLPipelineStartRequest(
            file_id=123,
            config=config,
            experiment_name="House Price Prediction",
            experiment_description="Predict house prices using various algorithms"
        )
        
        assert request.file_id == 123
        assert request.config.problem_type == ProblemTypeEnum.REGRESSION
        assert request.experiment_name == "House Price Prediction"
    
    def test_model_prediction_request(self):
        """Test ModelPredictionRequest"""
        request = ModelPredictionRequest(
            model_id="model_123",
            input_data={"feature1": 1.0, "feature2": 2.0, "feature3": "category_A"},
            return_probabilities=True
        )
        
        assert request.model_id == "model_123"
        assert request.input_data["feature1"] == 1.0
        assert request.return_probabilities == True
    
    def test_hyperparameter_tuning_request(self):
        """Test HyperparameterTuningRequest"""
        request = HyperparameterTuningRequest(
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            parameter_space={
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None]
            },
            optimization_metric=MetricNameEnum.F1_SCORE,
            max_iterations=100,
            optimization_method="grid_search"
        )
        
        assert request.algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        assert len(request.parameter_space["n_estimators"]) == 3
        assert request.optimization_metric == MetricNameEnum.F1_SCORE
        assert request.max_iterations == 100


if __name__ == "__main__":
    pytest.main([__file__]) 