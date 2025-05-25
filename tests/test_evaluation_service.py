#!/usr/bin/env python3
"""
Unit tests for DS1.3.2: Model Evaluation Service
Tests advanced model evaluation, visualization generation, and analysis features
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from evaluation import (
    ModelEvaluationService, MetricResult, VisualizationData, ModelAnalysis,
    evaluate_single_model, compare_multiple_models
)
from models.ml_models import ProblemTypeEnum


class TestModelEvaluationService:
    """Test cases for ModelEvaluationService class"""
    
    @pytest.fixture
    def temp_viz_dir(self):
        """Create temporary visualization directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 2, n_samples), name='target')
        
        return X, y
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression dataset"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randn(n_samples) * 10 + 50, name='target')
        
        return X, y
    
    @pytest.fixture
    def mock_classification_model(self):
        """Create mock classification model"""
        model = Mock()
        
        def predict_side_effect(X):
            n_samples = len(X)
            np.random.seed(42)  # For consistent results
            return np.random.randint(0, 2, n_samples)
        
        def predict_proba_side_effect(X):
            n_samples = len(X)
            np.random.seed(42)  # For consistent results
            proba = np.random.rand(n_samples, 2)
            # Normalize to sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
        
        model.predict.side_effect = predict_side_effect
        model.predict_proba.side_effect = predict_proba_side_effect
        model.feature_importances_ = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        return model
    
    @pytest.fixture
    def mock_regression_model(self):
        """Create mock regression model"""
        model = Mock()
        
        def predict_side_effect(X):
            n_samples = len(X)
            np.random.seed(42)  # For consistent results
            return np.random.randn(n_samples) * 10 + 50
        
        model.predict.side_effect = predict_side_effect
        model.feature_importances_ = np.array([0.25, 0.2, 0.3, 0.15, 0.1])
        return model
    
    def test_service_initialization(self, temp_viz_dir):
        """Test ModelEvaluationService initialization"""
        service = ModelEvaluationService(save_visualizations=True, viz_dir=temp_viz_dir)
        
        # Import the HAS_MATPLOTLIB flag from the evaluation module
        from evaluation import HAS_MATPLOTLIB
        
        if HAS_MATPLOTLIB:
            # Matplotlib is available
            assert service.save_visualizations == True
            assert os.path.exists(temp_viz_dir)
        else:
            # Matplotlib not available, visualizations should be disabled
            assert service.save_visualizations == False
        
        assert service.viz_dir == temp_viz_dir
    
    def test_service_initialization_no_viz(self):
        """Test ModelEvaluationService initialization without visualization saving"""
        service = ModelEvaluationService(save_visualizations=False)
        
        assert service.save_visualizations == False
        assert service.viz_dir == "evaluation_plots"
    
    def test_classification_metrics_calculation(self, temp_viz_dir):
        """Test classification metrics calculation"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Create test data
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8],
                                [0.4, 0.6], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]])
        
        metrics = service._calculate_metrics(y_true, y_pred, y_pred_proba, ProblemTypeEnum.CLASSIFICATION)
        
        # Check that required metrics are present
        metric_names = [m.name for m in metrics]
        assert "accuracy" in metric_names
        assert "precision" in metric_names
        assert "recall" in metric_names
        assert "f1_score" in metric_names
        assert "roc_auc" in metric_names
        
        # Check metric properties
        for metric in metrics:
            assert isinstance(metric, MetricResult)
            assert metric.value >= 0
            assert metric.display_name is not None
            assert metric.description is not None
            assert isinstance(metric.higher_is_better, bool)
            assert metric.category in ["performance", "reliability", "efficiency"]
    
    def test_regression_metrics_calculation(self, temp_viz_dir):
        """Test regression metrics calculation"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Create test data
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0, 48.0])
        
        metrics = service._calculate_metrics(y_true, y_pred, None, ProblemTypeEnum.REGRESSION)
        
        # Check that required metrics are present
        metric_names = [m.name for m in metrics]
        assert "mae" in metric_names
        assert "mse" in metric_names
        assert "rmse" in metric_names
        assert "r2" in metric_names
        assert "mape" in metric_names
        
        # Check metric values are reasonable
        mae_metric = next(m for m in metrics if m.name == "mae")
        assert mae_metric.value > 0
        assert not mae_metric.higher_is_better
        
        r2_metric = next(m for m in metrics if m.name == "r2")
        assert r2_metric.higher_is_better
    
    def test_primary_metric_selection(self, temp_viz_dir):
        """Test primary metric selection for different problem types"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Test classification primary metric
        classification_metrics = [
            MetricResult("accuracy", 0.8, "Accuracy", "", True, "performance"),
            MetricResult("f1_score", 0.75, "F1", "", True, "performance"),
            MetricResult("precision", 0.7, "Precision", "", True, "performance")
        ]
        
        primary = service._get_primary_metric(classification_metrics, ProblemTypeEnum.CLASSIFICATION)
        assert primary.name == "f1_score"
        
        # Test regression primary metric
        regression_metrics = [
            MetricResult("mae", 5.0, "MAE", "", False, "performance"),
            MetricResult("rmse", 7.0, "RMSE", "", False, "performance"),
            MetricResult("r2", 0.85, "R²", "", True, "performance")
        ]
        
        primary = service._get_primary_metric(regression_metrics, ProblemTypeEnum.REGRESSION)
        assert primary.name == "r2"
    
    def test_confusion_matrix_analysis(self, temp_viz_dir):
        """Test confusion matrix analysis"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
        
        confusion_data = service._analyze_confusion_matrix(y_true, y_pred)
        
        assert "matrix" in confusion_data
        assert "labels" in confusion_data
        assert "per_class_metrics" in confusion_data
        assert "total_samples" in confusion_data
        assert "correct_predictions" in confusion_data
        assert "accuracy" in confusion_data
        
        # Check matrix dimensions
        matrix = confusion_data["matrix"]
        assert len(matrix) == 2  # Binary classification
        assert len(matrix[0]) == 2
        
        # Check per-class metrics
        per_class = confusion_data["per_class_metrics"]
        assert "0" in per_class
        assert "1" in per_class
        
        for class_metrics in per_class.values():
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1_score" in class_metrics
            assert "support" in class_metrics
    
    def test_feature_importance_extraction(self, temp_viz_dir):
        """Test feature importance extraction from different model types"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        feature_names = ['feature_0', 'feature_1', 'feature_2']
        
        # Test tree-based model (feature_importances_)
        tree_model = Mock()
        tree_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        importance = service._extract_feature_importance(tree_model, feature_names)
        assert importance is not None
        assert len(importance) == 3
        assert importance['feature_0'] == 0.5
        assert importance['feature_1'] == 0.3
        assert importance['feature_2'] == 0.2
        
        # Test linear model (coef_)
        linear_model = Mock()
        linear_model.coef_ = np.array([0.4, -0.3, 0.2])
        del linear_model.feature_importances_  # Ensure it doesn't have this attribute
        
        importance = service._extract_feature_importance(linear_model, feature_names)
        assert importance is not None
        assert len(importance) == 3
        assert importance['feature_0'] == 0.4  # abs value
        assert importance['feature_1'] == 0.3  # abs value
        assert importance['feature_2'] == 0.2
        
        # Test model without importance
        no_importance_model = Mock()
        del no_importance_model.feature_importances_
        del no_importance_model.coef_
        
        importance = service._extract_feature_importance(no_importance_model, feature_names)
        assert importance is None
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_feature_importance_visualization(self, mock_close, mock_savefig, temp_viz_dir):
        """Test feature importance visualization creation"""
        service = ModelEvaluationService(save_visualizations=True, viz_dir=temp_viz_dir)
        
        feature_importance = {
            'feature_0': 0.5,
            'feature_1': 0.3,
            'feature_2': 0.2,
            'feature_3': 0.1
        }
        
        viz_data = service._create_feature_importance_plot(feature_importance, "test_algorithm")
        
        assert viz_data is not None
        assert isinstance(viz_data, VisualizationData)
        assert viz_data.chart_type == "feature_importance"
        assert viz_data.title == "Feature Importance - test_algorithm"
        assert "features" in viz_data.data
        assert "importances" in viz_data.data
        assert viz_data.data["top_feature"] == "feature_0"
        assert viz_data.file_path is not None
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_confusion_matrix_visualization(self, mock_close, mock_savefig, temp_viz_dir):
        """Test confusion matrix visualization creation"""
        service = ModelEvaluationService(save_visualizations=True, viz_dir=temp_viz_dir)
        
        confusion_data = {
            "matrix": [[10, 2], [3, 15]],
            "labels": ["0", "1"],
            "total_samples": 30,
            "correct_predictions": 25,
            "accuracy": 0.833
        }
        
        viz_data = service._create_confusion_matrix_plot(confusion_data, "test_algorithm")
        
        assert viz_data is not None
        assert isinstance(viz_data, VisualizationData)
        assert viz_data.chart_type == "confusion_matrix"
        assert viz_data.title == "Confusion Matrix - test_algorithm"
        assert viz_data.data == confusion_data
        assert viz_data.file_path is not None
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_insight_generation_classification(self, temp_viz_dir):
        """Test insight generation for classification models"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # High performance metrics
        high_metrics = [
            MetricResult("f1_score", 0.95, "F1", "", True, "performance"),
            MetricResult("accuracy", 0.94, "Accuracy", "", True, "performance")
        ]
        
        feature_importance = {"important_feature": 0.8, "less_important": 0.2}
        
        insights = service._generate_insights(
            high_metrics, feature_importance, None, ProblemTypeEnum.CLASSIFICATION
        )
        
        assert len(insights) >= 2
        assert "excellent" in insights[0].lower() or "very strong" in insights[0].lower()
        assert "important_feature" in insights[1]
        
        # Low performance metrics
        low_metrics = [
            MetricResult("f1_score", 0.3, "F1", "", True, "performance")
        ]
        
        insights = service._generate_insights(
            low_metrics, None, None, ProblemTypeEnum.CLASSIFICATION
        )
        
        assert len(insights) >= 1
        assert "poor" in insights[0].lower() or "low" in insights[0].lower()
    
    def test_insight_generation_regression(self, temp_viz_dir):
        """Test insight generation for regression models"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Test with negative R²
        negative_r2_metrics = [
            MetricResult("r2", -0.1, "R²", "", True, "performance"),
            MetricResult("mape", 15.0, "MAPE", "", False, "performance")
        ]
        
        insights = service._generate_insights(
            negative_r2_metrics, None, None, ProblemTypeEnum.REGRESSION
        )
        
        assert len(insights) >= 2
        assert "negative" in insights[0].lower()
        assert "worse than" in insights[0].lower()
        
        # Test with low MAPE
        good_metrics = [
            MetricResult("r2", 0.85, "R²", "", True, "performance"),
            MetricResult("mape", 5.0, "MAPE", "", False, "performance")
        ]
        
        insights = service._generate_insights(
            good_metrics, None, None, ProblemTypeEnum.REGRESSION
        )
        
        assert any("low mape" in insight.lower() for insight in insights)
    
    def test_recommendation_generation(self, temp_viz_dir):
        """Test recommendation generation for different algorithms"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Test tree-based algorithm recommendations
        tree_metrics = [
            MetricResult("f1_score", 0.6, "F1", "", True, "performance")
        ]
        
        tree_recommendations = service._generate_recommendations(
            tree_metrics, "decision_tree_classifier", ProblemTypeEnum.CLASSIFICATION
        )
        
        assert len(tree_recommendations) >= 2
        assert any("ensemble" in rec.lower() for rec in tree_recommendations)
        assert any("pruning" in rec.lower() for rec in tree_recommendations)
        
        # Test linear algorithm recommendations
        linear_metrics = [
            MetricResult("f1_score", 0.5, "F1", "", True, "performance")
        ]
        
        linear_recommendations = service._generate_recommendations(
            linear_metrics, "logistic_regression", ProblemTypeEnum.CLASSIFICATION
        )
        
        assert len(linear_recommendations) >= 2
        assert any("scaled" in rec.lower() or "normalized" in rec.lower() for rec in linear_recommendations)
    
    def test_model_complexity_estimation(self, temp_viz_dir):
        """Test model complexity estimation"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Test tree model complexity
        tree_model = Mock()
        tree_model.tree_ = Mock()
        tree_model.tree_.node_count = 15
        
        complexity = service._estimate_model_complexity(tree_model, "decision_tree")
        assert complexity == "low"
        
        tree_model.tree_.node_count = 150
        complexity = service._estimate_model_complexity(tree_model, "decision_tree")
        assert complexity == "high"
        
        # Test forest model complexity
        forest_model = Mock()
        forest_model.n_estimators = 30
        
        complexity = service._estimate_model_complexity(forest_model, "random_forest")
        assert complexity == "medium"
        
        forest_model.n_estimators = 300
        complexity = service._estimate_model_complexity(forest_model, "random_forest")
        assert complexity == "very_high"
        
        # Test linear model complexity
        linear_complexity = service._estimate_model_complexity(Mock(), "linear_regression")
        assert linear_complexity == "low"
        
        # Test unknown algorithm
        unknown_complexity = service._estimate_model_complexity(Mock(), "unknown_algorithm")
        assert unknown_complexity == "medium"
    
    def test_complete_classification_evaluation(self, sample_classification_data, mock_classification_model, temp_viz_dir):
        """Test complete model evaluation for classification"""
        X, y = sample_classification_data
        X_test, y_test = X[:50], y[:50]  # Use first 50 samples as test
        
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Import HAS_MATPLOTLIB to check if matplotlib is available
        from evaluation import HAS_MATPLOTLIB
        
        if HAS_MATPLOTLIB:
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                analysis = service.evaluate_model(
                    mock_classification_model,
                    X_test,
                    y_test,
                    "random_forest_classifier",
                    ProblemTypeEnum.CLASSIFICATION,
                    training_time=1.5
                )
        else:
            # No matplotlib, just run without patching
            analysis = service.evaluate_model(
                mock_classification_model,
                X_test,
                y_test,
                "random_forest_classifier",
                ProblemTypeEnum.CLASSIFICATION,
                training_time=1.5
            )
        
        # Check analysis structure
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.algorithm_name == "random_forest_classifier"
        assert analysis.problem_type == "classification"
        assert analysis.training_time == 1.5
        
        # Check metrics
        assert len(analysis.metrics) >= 4  # At least accuracy, precision, recall, f1
        assert analysis.primary_metric.name in ["f1_score", "accuracy"]
        
        # Check predictions
        assert len(analysis.predictions) == 50
        assert analysis.prediction_probabilities is not None
        assert analysis.prediction_probabilities.shape == (50, 2)
        
        # Check analysis components
        assert analysis.confusion_matrix_data is not None
        assert analysis.classification_report_data is not None
        assert analysis.feature_importance is not None
        
        # Check insights and recommendations
        assert len(analysis.insights) > 0
        assert len(analysis.recommendations) > 0
        
        # Check metadata
        assert analysis.evaluation_timestamp is not None
        assert analysis.model_complexity in ["low", "medium", "high", "very_high"]
    
    def test_complete_regression_evaluation(self, sample_regression_data, mock_regression_model, temp_viz_dir):
        """Test complete model evaluation for regression"""
        X, y = sample_regression_data
        X_test, y_test = X[:50], y[:50]  # Use first 50 samples as test
        
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Import HAS_MATPLOTLIB to check if matplotlib is available
        from evaluation import HAS_MATPLOTLIB
        
        if HAS_MATPLOTLIB:
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                analysis = service.evaluate_model(
                    mock_regression_model,
                    X_test,
                    y_test,
                    "random_forest_regressor",
                    ProblemTypeEnum.REGRESSION,
                    training_time=2.1
                )
        else:
            # No matplotlib, just run without patching
            analysis = service.evaluate_model(
                mock_regression_model,
                X_test,
                y_test,
                "random_forest_regressor",
                ProblemTypeEnum.REGRESSION,
                training_time=2.1
            )
        
        # Check analysis structure
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.algorithm_name == "random_forest_regressor"
        assert analysis.problem_type == "regression"
        assert analysis.training_time == 2.1
        
        # Check metrics
        assert len(analysis.metrics) >= 4  # At least MAE, MSE, RMSE, R²
        assert analysis.primary_metric.name in ["r2", "rmse"]
        
        # Check predictions
        assert len(analysis.predictions) == 50
        assert analysis.prediction_probabilities is None  # No probabilities for regression
        
        # Check analysis components
        assert analysis.confusion_matrix_data is None  # No confusion matrix for regression
        assert analysis.classification_report_data is None
        assert analysis.feature_importance is not None
        
        # Check insights and recommendations
        assert len(analysis.insights) > 0
        assert len(analysis.recommendations) > 0
    
    def test_model_comparison(self, temp_viz_dir):
        """Test model comparison functionality"""
        service = ModelEvaluationService(save_visualizations=False, viz_dir=temp_viz_dir)
        
        # Create mock analyses
        analysis1 = ModelAnalysis(
            algorithm_name="algorithm_1",
            problem_type="classification",
            metrics=[],
            primary_metric=MetricResult("f1_score", 0.85, "F1", "", True, "performance"),
            predictions=np.array([]),
            prediction_probabilities=None,
            confusion_matrix_data=None,
            classification_report_data=None,
            feature_importance=None,
            training_time=1.5,
            prediction_time=None,
            memory_usage=None,
            visualizations=[],
            insights=["insight1", "insight2"],
            recommendations=["rec1"],
            evaluation_timestamp="2024-01-01T12:00:00",
            model_complexity="medium"
        )
        
        analysis2 = ModelAnalysis(
            algorithm_name="algorithm_2",
            problem_type="classification",
            metrics=[],
            primary_metric=MetricResult("f1_score", 0.78, "F1", "", True, "performance"),
            predictions=np.array([]),
            prediction_probabilities=None,
            confusion_matrix_data=None,
            classification_report_data=None,
            feature_importance=None,
            training_time=0.8,
            prediction_time=None,
            memory_usage=None,
            visualizations=[],
            insights=["insight1"],
            recommendations=["rec1", "rec2"],
            evaluation_timestamp="2024-01-01T12:00:00",
            model_complexity="low"
        )
        
        comparison = service.compare_models([analysis1, analysis2])
        
        # Check comparison structure
        assert "best_model" in comparison
        assert "model_ranking" in comparison
        assert "comparison_summary" in comparison
        
        # Check best model (should be algorithm_1 with higher F1)
        best_model = comparison["best_model"]
        assert best_model["algorithm_name"] == "algorithm_1"
        assert best_model["primary_metric_value"] == 0.85
        
        # Check ranking
        ranking = comparison["model_ranking"]
        assert len(ranking) == 2
        assert ranking[0]["algorithm_name"] == "algorithm_1"  # Best first
        assert ranking[1]["algorithm_name"] == "algorithm_2"
        
        # Check summary
        summary = comparison["comparison_summary"]
        assert summary["total_models"] == 2
        assert summary["problem_type"] == "classification"


class TestConvenienceFunctions:
    """Test convenience functions for external use"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(20, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.randint(0, 2, 20))
        return X, y
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        model = Mock()
        model.predict.return_value = np.random.randint(0, 2, 20)
        model.predict_proba.return_value = np.random.rand(20, 2)
        model.feature_importances_ = np.array([0.4, 0.35, 0.25])
        return model
    
    def test_evaluate_single_model_function(self, sample_data, mock_model):
        """Test evaluate_single_model convenience function"""
        X, y = sample_data
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            analysis = evaluate_single_model(
                mock_model,
                X,
                y,
                "test_algorithm",
                ProblemTypeEnum.CLASSIFICATION
            )
        
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.algorithm_name == "test_algorithm"
        assert analysis.problem_type == "classification"
        assert len(analysis.metrics) > 0
    
    def test_compare_multiple_models_function(self):
        """Test compare_multiple_models convenience function"""
        # Create mock analyses
        analysis1 = ModelAnalysis(
            algorithm_name="algo1",
            problem_type="classification",
            metrics=[],
            primary_metric=MetricResult("f1_score", 0.9, "F1", "", True, "performance"),
            predictions=np.array([]),
            prediction_probabilities=None,
            confusion_matrix_data=None,
            classification_report_data=None,
            feature_importance=None,
            training_time=1.0,
            prediction_time=None,
            memory_usage=None,
            visualizations=[],
            insights=[],
            recommendations=[],
            evaluation_timestamp="2024-01-01T12:00:00",
            model_complexity="medium"
        )
        
        analysis2 = ModelAnalysis(
            algorithm_name="algo2",
            problem_type="classification",
            metrics=[],
            primary_metric=MetricResult("f1_score", 0.8, "F1", "", True, "performance"),
            predictions=np.array([]),
            prediction_probabilities=None,
            confusion_matrix_data=None,
            classification_report_data=None,
            feature_importance=None,
            training_time=1.5,
            prediction_time=None,
            memory_usage=None,
            visualizations=[],
            insights=[],
            recommendations=[],
            evaluation_timestamp="2024-01-01T12:00:00",
            model_complexity="high"
        )
        
        comparison = compare_multiple_models([analysis1, analysis2])
        
        assert "best_model" in comparison
        assert comparison["best_model"]["algorithm_name"] == "algo1"
        assert len(comparison["model_ranking"]) == 2


class TestDataClasses:
    """Test data classes used in evaluation service"""
    
    def test_metric_result_creation(self):
        """Test MetricResult data class"""
        metric = MetricResult(
            name="accuracy",
            value=0.85,
            display_name="Accuracy",
            description="Fraction of correct predictions",
            higher_is_better=True,
            category="performance"
        )
        
        assert metric.name == "accuracy"
        assert metric.value == 0.85
        assert metric.display_name == "Accuracy"
        assert metric.description == "Fraction of correct predictions"
        assert metric.higher_is_better == True
        assert metric.category == "performance"
    
    def test_visualization_data_creation(self):
        """Test VisualizationData data class"""
        viz_data = VisualizationData(
            chart_type="confusion_matrix",
            data={"matrix": [[10, 2], [3, 15]]},
            title="Confusion Matrix",
            description="Classification results matrix",
            file_path="/path/to/plot.png"
        )
        
        assert viz_data.chart_type == "confusion_matrix"
        assert viz_data.data["matrix"] == [[10, 2], [3, 15]]
        assert viz_data.title == "Confusion Matrix"
        assert viz_data.description == "Classification results matrix"
        assert viz_data.file_path == "/path/to/plot.png"
    
    def test_model_analysis_creation(self):
        """Test ModelAnalysis data class"""
        metric = MetricResult("f1", 0.8, "F1", "", True, "performance")
        viz = VisualizationData("plot", {}, "Title", "Desc")
        
        analysis = ModelAnalysis(
            algorithm_name="test_algo",
            problem_type="classification",
            metrics=[metric],
            primary_metric=metric,
            predictions=np.array([0, 1, 0]),
            prediction_probabilities=None,
            confusion_matrix_data=None,
            classification_report_data=None,
            feature_importance={"f1": 0.5},
            training_time=1.2,
            prediction_time=0.1,
            memory_usage=100.0,
            visualizations=[viz],
            insights=["insight"],
            recommendations=["recommendation"],
            evaluation_timestamp="2024-01-01T12:00:00",
            model_complexity="medium"
        )
        
        assert analysis.algorithm_name == "test_algo"
        assert analysis.problem_type == "classification"
        assert len(analysis.metrics) == 1
        assert analysis.primary_metric.name == "f1"
        assert len(analysis.predictions) == 3
        assert analysis.feature_importance["f1"] == 0.5
        assert analysis.training_time == 1.2
        assert len(analysis.visualizations) == 1
        assert len(analysis.insights) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.model_complexity == "medium"


if __name__ == "__main__":
    pytest.main([__file__]) 