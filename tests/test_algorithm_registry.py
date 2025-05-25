#!/usr/bin/env python3
"""
Unit tests for DS1.2.2: Algorithm Registry
Tests algorithm registry functionality, validation, and configuration management
"""

import pytest
import sys
import os
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from algorithm_registry import (
    AlgorithmRegistry, HyperparameterSpec, AlgorithmDefinition,
    get_algorithm_registry, get_supported_algorithms, 
    create_default_algorithm_configs, validate_algorithm_config
)

from models.ml_models import (
    AlgorithmNameEnum, ProblemTypeEnum, MetricNameEnum,
    PreprocessingStepEnum, AlgorithmConfig
)


class TestAlgorithmRegistry:
    """Test cases for AlgorithmRegistry class"""
    
    def test_registry_initialization(self):
        """Test registry initializes with all expected algorithms"""
        registry = AlgorithmRegistry()
        
        # Should have exactly 10 algorithms
        all_algorithms = registry.get_all_algorithms()
        assert len(all_algorithms) == 10
        
        # Check all expected algorithms are present
        expected_algorithms = [
            # Classification
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_CLASSIFIER,
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            AlgorithmNameEnum.SVM_CLASSIFIER,
            AlgorithmNameEnum.KNN_CLASSIFIER,
            # Regression
            AlgorithmNameEnum.LINEAR_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_REGRESSOR,
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            AlgorithmNameEnum.SVM_REGRESSOR,
            AlgorithmNameEnum.KNN_REGRESSOR,
        ]
        
        for algo in expected_algorithms:
            assert algo in all_algorithms
    
    def test_get_algorithm(self):
        """Test retrieving individual algorithms"""
        registry = AlgorithmRegistry()
        
        # Test valid algorithm
        rf_algo = registry.get_algorithm(AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER)
        assert rf_algo is not None
        assert rf_algo.name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        assert rf_algo.display_name == "Random Forest"
        
        # Test invalid algorithm
        invalid_algo = registry.get_algorithm("invalid_algorithm")
        assert invalid_algo is None
    
    def test_get_algorithms_by_problem_type(self):
        """Test filtering algorithms by problem type"""
        registry = AlgorithmRegistry()
        
        # Test classification algorithms
        classification_algos = registry.get_algorithms_by_problem_type(ProblemTypeEnum.CLASSIFICATION)
        assert len(classification_algos) == 5
        
        expected_classification = [
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_CLASSIFIER,
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            AlgorithmNameEnum.SVM_CLASSIFIER,
            AlgorithmNameEnum.KNN_CLASSIFIER,
        ]
        
        for algo in expected_classification:
            assert algo in classification_algos
            assert ProblemTypeEnum.CLASSIFICATION in classification_algos[algo].problem_types
        
        # Test regression algorithms
        regression_algos = registry.get_algorithms_by_problem_type(ProblemTypeEnum.REGRESSION)
        assert len(regression_algos) == 5
        
        expected_regression = [
            AlgorithmNameEnum.LINEAR_REGRESSION,
            AlgorithmNameEnum.DECISION_TREE_REGRESSOR,
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            AlgorithmNameEnum.SVM_REGRESSOR,
            AlgorithmNameEnum.KNN_REGRESSOR,
        ]
        
        for algo in expected_regression:
            assert algo in regression_algos
            assert ProblemTypeEnum.REGRESSION in regression_algos[algo].problem_types
    
    def test_get_default_algorithms(self):
        """Test default algorithm recommendations"""
        registry = AlgorithmRegistry()
        
        # Test classification defaults
        default_classification = registry.get_default_algorithms(ProblemTypeEnum.CLASSIFICATION)
        assert 1 <= len(default_classification) <= 3
        assert AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER in default_classification
        
        # Test regression defaults
        default_regression = registry.get_default_algorithms(ProblemTypeEnum.REGRESSION)
        assert 1 <= len(default_regression) <= 3
        assert AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR in default_regression
        
        # Test with custom max_count
        limited_defaults = registry.get_default_algorithms(ProblemTypeEnum.CLASSIFICATION, max_count=2)
        assert len(limited_defaults) <= 2
    
    def test_create_algorithm_config(self):
        """Test creating algorithm configurations"""
        registry = AlgorithmRegistry()
        
        # Test creating config with defaults
        config = registry.create_algorithm_config(AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER)
        
        assert config.algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER
        assert config.is_enabled == True
        assert len(config.hyperparameters) > 0
        assert "n_estimators" in config.hyperparameters
        assert config.hyperparameters["n_estimators"] == 100  # Default value
        
        # Test creating config with custom hyperparameters
        custom_params = {"n_estimators": 200, "max_depth": 15}
        custom_config = registry.create_algorithm_config(
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            custom_params
        )
        
        assert custom_config.hyperparameters["n_estimators"] == 200
        assert custom_config.hyperparameters["max_depth"] == 15
        
        # Test invalid algorithm
        with pytest.raises(ValueError):
            registry.create_algorithm_config("invalid_algorithm")


class TestHyperparameterValidation:
    """Test cases for hyperparameter validation"""
    
    def test_valid_hyperparameters(self):
        """Test validation of valid hyperparameters"""
        registry = AlgorithmRegistry()
        
        valid_params = {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 5,
            "bootstrap": True
        }
        
        validated = registry.validate_hyperparameters(
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            valid_params
        )
        
        assert validated["n_estimators"] == 150
        assert validated["max_depth"] == 12
        assert validated["min_samples_split"] == 5
        assert validated["bootstrap"] == True
    
    def test_invalid_hyperparameter_ranges(self):
        """Test validation rejects out-of-range values"""
        registry = AlgorithmRegistry()
        
        # Test values above maximum
        invalid_params = {
            "n_estimators": 2000,  # Above max of 1000
            "max_depth": 100       # Above max of 50
        }
        
        with pytest.raises(ValueError) as exc_info:
            registry.validate_hyperparameters(
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                invalid_params
            )
        
        error_message = str(exc_info.value)
        assert "above maximum" in error_message
        
        # Test values below minimum
        invalid_params = {
            "n_estimators": 5,     # Below min of 10
            "max_depth": 0         # Below min of 1
        }
        
        with pytest.raises(ValueError) as exc_info:
            registry.validate_hyperparameters(
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                invalid_params
            )
        
        error_message = str(exc_info.value)
        assert "below minimum" in error_message
    
    def test_invalid_hyperparameter_types(self):
        """Test validation rejects wrong types"""
        registry = AlgorithmRegistry()
        
        invalid_params = {
            "n_estimators": "not_a_number",
            "bootstrap": "not_a_bool",
            "max_depth": 10.5  # Should be int
        }
        
        with pytest.raises(ValueError) as exc_info:
            registry.validate_hyperparameters(
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                invalid_params
            )
        
        error_message = str(exc_info.value)
        assert "Expected" in error_message
    
    def test_invalid_allowed_values(self):
        """Test validation rejects values not in allowed list"""
        registry = AlgorithmRegistry()
        
        invalid_params = {
            "max_features": "invalid_value"  # Not in allowed values
        }
        
        with pytest.raises(ValueError) as exc_info:
            registry.validate_hyperparameters(
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                invalid_params
            )
        
        error_message = str(exc_info.value)
        assert "not in allowed values" in error_message
    
    def test_unknown_hyperparameters(self):
        """Test validation rejects unknown hyperparameters"""
        registry = AlgorithmRegistry()
        
        invalid_params = {
            "unknown_param": 123,
            "another_unknown": "value"
        }
        
        with pytest.raises(ValueError) as exc_info:
            registry.validate_hyperparameters(
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                invalid_params
            )
        
        error_message = str(exc_info.value)
        assert "Unknown hyperparameter" in error_message
    
    def test_type_conversion(self):
        """Test automatic type conversion when possible"""
        registry = AlgorithmRegistry()
        
        # String numbers should be converted to int/float
        params_with_string_numbers = {
            "n_estimators": "150",
            "C": "1.5"  # For SVM
        }
        
        # Test Random Forest (int conversion)
        validated_rf = registry.validate_hyperparameters(
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            {"n_estimators": "150"}
        )
        assert validated_rf["n_estimators"] == 150
        assert isinstance(validated_rf["n_estimators"], int)
        
        # Test SVM (float conversion)
        validated_svm = registry.validate_hyperparameters(
            AlgorithmNameEnum.SVM_CLASSIFIER,
            {"C": "1.5"}
        )
        assert validated_svm["C"] == 1.5
        assert isinstance(validated_svm["C"], float)


class TestAlgorithmDefinitions:
    """Test cases for individual algorithm definitions"""
    
    def test_random_forest_classifier_definition(self):
        """Test Random Forest Classifier definition"""
        registry = AlgorithmRegistry()
        algo_def = registry.get_algorithm(AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER)
        
        assert algo_def.display_name == "Random Forest"
        assert ProblemTypeEnum.CLASSIFICATION in algo_def.problem_types
        assert algo_def.supports_feature_importance == True
        assert algo_def.supports_probabilities == True
        assert algo_def.training_complexity == "medium"
        
        # Check hyperparameters
        param_names = {param.name for param in algo_def.hyperparameters}
        expected_params = {"n_estimators", "max_depth", "min_samples_split", "bootstrap"}
        assert expected_params.issubset(param_names)
    
    def test_logistic_regression_definition(self):
        """Test Logistic Regression definition"""
        registry = AlgorithmRegistry()
        algo_def = registry.get_algorithm(AlgorithmNameEnum.LOGISTIC_REGRESSION)
        
        assert algo_def.display_name == "Logistic Regression"
        assert ProblemTypeEnum.CLASSIFICATION in algo_def.problem_types
        assert algo_def.supports_feature_importance == True
        assert algo_def.supports_probabilities == True
        assert algo_def.training_complexity == "low"
        
        # Check hyperparameters
        param_names = {param.name for param in algo_def.hyperparameters}
        expected_params = {"C", "max_iter", "penalty", "solver"}
        assert expected_params.issubset(param_names)
    
    def test_svm_classifier_definition(self):
        """Test SVM Classifier definition"""
        registry = AlgorithmRegistry()
        algo_def = registry.get_algorithm(AlgorithmNameEnum.SVM_CLASSIFIER)
        
        assert algo_def.display_name == "Support Vector Machine"
        assert ProblemTypeEnum.CLASSIFICATION in algo_def.problem_types
        assert algo_def.supports_feature_importance == False  # SVM doesn't support feature importance
        assert algo_def.supports_probabilities == True
        assert algo_def.training_complexity == "high"
        
        # Check hyperparameters
        param_names = {param.name for param in algo_def.hyperparameters}
        expected_params = {"C", "kernel", "gamma", "probability"}
        assert expected_params.issubset(param_names)
    
    def test_linear_regression_definition(self):
        """Test Linear Regression definition"""
        registry = AlgorithmRegistry()
        algo_def = registry.get_algorithm(AlgorithmNameEnum.LINEAR_REGRESSION)
        
        assert algo_def.display_name == "Linear Regression"
        assert ProblemTypeEnum.REGRESSION in algo_def.problem_types
        assert algo_def.supports_feature_importance == True
        assert algo_def.supports_probabilities == False  # Regression doesn't predict probabilities
        assert algo_def.training_complexity == "low"
        
        # Check hyperparameters
        param_names = {param.name for param in algo_def.hyperparameters}
        expected_params = {"fit_intercept"}
        assert expected_params.issubset(param_names)


class TestAlgorithmInfo:
    """Test cases for algorithm information retrieval"""
    
    def test_get_algorithm_info(self):
        """Test getting comprehensive algorithm information"""
        registry = AlgorithmRegistry()
        
        info = registry.get_algorithm_info(AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER)
        
        # Check required fields
        required_fields = [
            "name", "display_name", "description", "problem_types",
            "hyperparameters", "default_metrics", "recommended_preprocessing",
            "min_samples", "supports_feature_importance", "training_complexity"
        ]
        
        for field in required_fields:
            assert field in info
        
        # Check specific values
        assert info["name"] == "random_forest_classifier"
        assert info["display_name"] == "Random Forest"
        assert "classification" in info["problem_types"]
        assert len(info["hyperparameters"]) > 0
        assert info["supports_feature_importance"] == True
        
        # Check hyperparameter structure
        hyperparam = info["hyperparameters"][0]
        hyperparam_fields = ["name", "type", "default", "description", "required"]
        for field in hyperparam_fields:
            assert field in hyperparam
    
    def test_get_algorithm_info_invalid(self):
        """Test getting info for invalid algorithm"""
        registry = AlgorithmRegistry()
        
        with pytest.raises(ValueError):
            registry.get_algorithm_info("invalid_algorithm")


class TestConvenienceFunctions:
    """Test cases for convenience functions"""
    
    def test_get_supported_algorithms(self):
        """Test get_supported_algorithms function"""
        # Test all algorithms
        all_algos = get_supported_algorithms()
        assert len(all_algos) == 10
        
        # Test classification algorithms only
        classification_algos = get_supported_algorithms(ProblemTypeEnum.CLASSIFICATION)
        assert len(classification_algos) == 5
        
        # Check structure
        algo = all_algos[0]
        assert "name" in algo
        assert "display_name" in algo
        assert "hyperparameters" in algo
    
    def test_create_default_algorithm_configs(self):
        """Test create_default_algorithm_configs function"""
        # Test classification
        configs = create_default_algorithm_configs(ProblemTypeEnum.CLASSIFICATION)
        assert len(configs) <= 3
        assert all(isinstance(config, AlgorithmConfig) for config in configs)
        
        # Check Random Forest is included (should be first default)
        rf_included = any(
            config.algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER 
            for config in configs
        )
        assert rf_included
        
        # Test regression
        reg_configs = create_default_algorithm_configs(ProblemTypeEnum.REGRESSION)
        assert len(reg_configs) <= 3
        assert all(isinstance(config, AlgorithmConfig) for config in reg_configs)
    
    def test_validate_algorithm_config_function(self):
        """Test validate_algorithm_config convenience function"""
        # Test valid config
        valid_config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            hyperparameters={"C": 1.0, "max_iter": 1000}
        )
        assert validate_algorithm_config(valid_config) == True
        
        # Test invalid config
        invalid_config = AlgorithmConfig(
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            hyperparameters={"C": -1.0}  # Negative C is invalid
        )
        assert validate_algorithm_config(invalid_config) == False


class TestAlgorithmMetrics:
    """Test cases for algorithm metrics and preprocessing"""
    
    def test_classification_metrics(self):
        """Test that classification algorithms have appropriate metrics"""
        registry = AlgorithmRegistry()
        
        classification_algos = registry.get_algorithms_by_problem_type(ProblemTypeEnum.CLASSIFICATION)
        
        for algo_name, algo_def in classification_algos.items():
            # Should have classification metrics
            metric_values = [metric.value for metric in algo_def.default_metrics]
            assert "accuracy" in metric_values
            
            # Should not have regression-only metrics
            regression_metrics = {"mae", "mse", "rmse", "r2_score"}
            assert not any(metric in metric_values for metric in regression_metrics)
    
    def test_regression_metrics(self):
        """Test that regression algorithms have appropriate metrics"""
        registry = AlgorithmRegistry()
        
        regression_algos = registry.get_algorithms_by_problem_type(ProblemTypeEnum.REGRESSION)
        
        for algo_name, algo_def in regression_algos.items():
            # Should have regression metrics
            metric_values = [metric.value for metric in algo_def.default_metrics]
            common_regression_metrics = {"mae", "mse", "rmse", "r2_score"}
            assert any(metric in metric_values for metric in common_regression_metrics)
            
            # Should not have classification-only metrics
            classification_metrics = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
            assert not any(metric in metric_values for metric in classification_metrics)
    
    def test_preprocessing_recommendations(self):
        """Test preprocessing recommendations are appropriate"""
        registry = AlgorithmRegistry()
        
        # Algorithms that need scaling
        scaling_algos = [
            AlgorithmNameEnum.LOGISTIC_REGRESSION,
            AlgorithmNameEnum.SVM_CLASSIFIER,
            AlgorithmNameEnum.KNN_CLASSIFIER,
            AlgorithmNameEnum.SVM_REGRESSOR,
            AlgorithmNameEnum.KNN_REGRESSOR
        ]
        
        for algo_name in scaling_algos:
            algo_def = registry.get_algorithm(algo_name)
            preprocessing_steps = [step.value for step in algo_def.recommended_preprocessing]
            assert "scale_features" in preprocessing_steps, f"{algo_name} should recommend scaling"
        
        # Tree-based algorithms that don't need scaling
        tree_algos = [
            AlgorithmNameEnum.DECISION_TREE_CLASSIFIER,
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            AlgorithmNameEnum.DECISION_TREE_REGRESSOR,
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR
        ]
        
        for algo_name in tree_algos:
            algo_def = registry.get_algorithm(algo_name)
            preprocessing_steps = [step.value for step in algo_def.recommended_preprocessing]
            assert "scale_features" not in preprocessing_steps, f"{algo_name} should not recommend scaling"


class TestGlobalRegistry:
    """Test cases for global registry instance"""
    
    def test_singleton_behavior(self):
        """Test that get_algorithm_registry returns same instance"""
        registry1 = get_algorithm_registry()
        registry2 = get_algorithm_registry()
        
        # Should be the same instance
        assert registry1 is registry2
        
        # Should have same algorithms
        algos1 = registry1.get_all_algorithms()
        algos2 = registry2.get_all_algorithms()
        assert algos1.keys() == algos2.keys()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 