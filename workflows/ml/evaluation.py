"""
DS1.3.2: Model Evaluation Service
Advanced model evaluation with comprehensive metrics, visualizations, and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import sys
import os

# Scientific computing and ML libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve

# Handle matplotlib and seaborn imports gracefully
HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.ioff()  # Turn off interactive mode
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

# Add app to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from models.ml_models import ProblemTypeEnum, MetricNameEnum


@dataclass
class MetricResult:
    """Individual metric calculation result"""
    name: str
    value: float
    display_name: str
    description: str
    higher_is_better: bool
    category: str  # performance, reliability, efficiency


@dataclass
class VisualizationData:
    """Data for creating visualizations"""
    chart_type: str  # confusion_matrix, roc_curve, feature_importance, etc.
    data: Dict[str, Any]
    title: str
    description: str
    file_path: Optional[str] = None


@dataclass
class ModelAnalysis:
    """Comprehensive model analysis results"""
    algorithm_name: str
    problem_type: str
    
    # Core metrics
    metrics: List[MetricResult]
    primary_metric: MetricResult
    
    # Predictions and probabilities
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray]
    
    # Analysis components
    confusion_matrix_data: Optional[Dict[str, Any]]
    classification_report_data: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    
    # Performance analysis
    training_time: float
    prediction_time: Optional[float]
    memory_usage: Optional[float]
    
    # Visualizations
    visualizations: List[VisualizationData]
    
    # Insights and recommendations
    insights: List[str]
    recommendations: List[str]
    
    # Metadata
    evaluation_timestamp: str
    model_complexity: str


class ModelEvaluationService:
    """
    Advanced model evaluation service for comprehensive model analysis
    """
    
    def __init__(self, save_visualizations: bool = True, viz_dir: str = "evaluation_plots"):
        self.save_visualizations = save_visualizations and HAS_MATPLOTLIB
        self.viz_dir = viz_dir
        
        # Create visualization directory only if matplotlib is available
        if self.save_visualizations:
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Configure matplotlib for better plots only if available
        if HAS_MATPLOTLIB:
            try:
                plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
                sns.set_palette("husl")
            except Exception:
                pass  # Fallback to default style
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        algorithm_name: str,
        problem_type: ProblemTypeEnum,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        training_time: float = 0.0,
        feature_names: Optional[List[str]] = None
    ) -> ModelAnalysis:
        """
        Comprehensive model evaluation with advanced analysis
        
        Args:
            model: Trained scikit-learn model
            X_test: Test features
            y_test: Test targets
            algorithm_name: Name of the algorithm
            problem_type: Classification or regression
            X_train: Training features (optional, for advanced analysis)
            y_train: Training targets (optional, for advanced analysis)
            training_time: Model training time in seconds
            feature_names: Names of features for interpretation
        
        Returns:
            ModelAnalysis with comprehensive evaluation results
        """
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba') and problem_type == ProblemTypeEnum.CLASSIFICATION:
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate core metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, problem_type)
        
        # Determine primary metric
        primary_metric = self._get_primary_metric(metrics, problem_type)
        
        # Generate confusion matrix and classification report for classification
        confusion_data = None
        classification_data = None
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            confusion_data = self._analyze_confusion_matrix(y_test, y_pred)
            classification_data = self._generate_classification_report(y_test, y_pred)
        
        # Extract feature importance
        feature_importance = self._extract_feature_importance(model, feature_names or X_test.columns.tolist())
        
        # Create visualizations
        visualizations = self._create_visualizations(
            y_test, y_pred, y_pred_proba, model, algorithm_name, 
            problem_type, feature_importance, confusion_data
        )
        
        # Generate insights and recommendations
        insights = self._generate_insights(metrics, feature_importance, confusion_data, problem_type)
        recommendations = self._generate_recommendations(metrics, algorithm_name, problem_type)
        
        # Determine model complexity
        complexity = self._estimate_model_complexity(model, algorithm_name)
        
        return ModelAnalysis(
            algorithm_name=algorithm_name,
            problem_type=problem_type.value,
            metrics=metrics,
            primary_metric=primary_metric,
            predictions=y_pred,
            prediction_probabilities=y_pred_proba,
            confusion_matrix_data=confusion_data,
            classification_report_data=classification_data,
            feature_importance=feature_importance,
            training_time=training_time,
            prediction_time=None,  # Could be measured if needed
            memory_usage=None,     # Could be measured if needed
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations,
            evaluation_timestamp=datetime.now().isoformat(),
            model_complexity=complexity
        )
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray],
        problem_type: ProblemTypeEnum
    ) -> List[MetricResult]:
        """Calculate comprehensive metrics based on problem type"""
        metrics = []
        
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            # Core classification metrics
            metrics.extend([
                MetricResult(
                    name="accuracy",
                    value=float(accuracy_score(y_true, y_pred)),
                    display_name="Accuracy",
                    description="Fraction of predictions that match the true labels",
                    higher_is_better=True,
                    category="performance"
                ),
                MetricResult(
                    name="precision",
                    value=float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    display_name="Precision (Weighted)",
                    description="Weighted average precision across all classes",
                    higher_is_better=True,
                    category="performance"
                ),
                MetricResult(
                    name="recall",
                    value=float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    display_name="Recall (Weighted)",
                    description="Weighted average recall across all classes",
                    higher_is_better=True,
                    category="performance"
                ),
                MetricResult(
                    name="f1_score",
                    value=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                    display_name="F1-Score (Weighted)",
                    description="Weighted harmonic mean of precision and recall",
                    higher_is_better=True,
                    category="performance"
                )
            ])
            
            # ROC-AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:  # Multi-class
                        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    
                    metrics.append(MetricResult(
                        name="roc_auc",
                        value=float(roc_auc),
                        display_name="ROC-AUC",
                        description="Area under the ROC curve",
                        higher_is_better=True,
                        category="performance"
                    ))
                except Exception:
                    pass  # Skip if ROC-AUC can't be calculated
        
        else:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            metrics.extend([
                MetricResult(
                    name="mae",
                    value=float(mean_absolute_error(y_true, y_pred)),
                    display_name="Mean Absolute Error",
                    description="Average absolute difference between predictions and true values",
                    higher_is_better=False,
                    category="performance"
                ),
                MetricResult(
                    name="mse",
                    value=float(mse),
                    display_name="Mean Squared Error",
                    description="Average squared difference between predictions and true values",
                    higher_is_better=False,
                    category="performance"
                ),
                MetricResult(
                    name="rmse",
                    value=float(np.sqrt(mse)),
                    display_name="Root Mean Squared Error",
                    description="Square root of the mean squared error",
                    higher_is_better=False,
                    category="performance"
                ),
                MetricResult(
                    name="r2",
                    value=float(r2_score(y_true, y_pred)),
                    display_name="R² Score",
                    description="Coefficient of determination (explained variance)",
                    higher_is_better=True,
                    category="performance"
                )
            ])
        
        return metrics
    
    def _get_primary_metric(self, metrics: List[MetricResult], problem_type: ProblemTypeEnum) -> MetricResult:
        """Determine the primary metric for model comparison"""
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            # Look for F1-score first, then accuracy
            for metric in metrics:
                if metric.name == "f1_score":
                    return metric
            for metric in metrics:
                if metric.name == "accuracy":
                    return metric
        else:  # Regression
            # Look for R² first, then RMSE
            for metric in metrics:
                if metric.name == "r2":
                    return metric
            for metric in metrics:
                if metric.name == "rmse":
                    return metric
        
        # Fallback to first metric
        return metrics[0] if metrics else MetricResult("unknown", 0.0, "Unknown", "", True, "performance")
    
    def _analyze_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix for classification problems"""
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true.unique()) | set(np.unique(y_pred))))
        
        return {
            "matrix": cm.tolist(),
            "labels": [str(label) for label in labels],
            "total_samples": len(y_true),
            "correct_predictions": int(np.trace(cm)),
            "accuracy": float(np.trace(cm) / cm.sum())
        }
    
    def _generate_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate detailed classification report"""
        try:
            return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            return {}
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from the model"""
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.mean(np.abs(model.coef_), axis=0)
        
        if importance is not None and len(importance) == len(feature_names):
            return dict(zip(feature_names, importance.astype(float)))
        
        return None
    
    def _create_visualizations(self, *args, **kwargs) -> List[VisualizationData]:
        """Create visualizations for model analysis (placeholder)"""
        return []  # Simplified for now
    
    def _generate_insights(self, metrics: List[MetricResult], *args, **kwargs) -> List[str]:
        """Generate insights about model performance"""
        insights = []
        
        # Performance insights
        primary_metric = next((m for m in metrics if m.name in ["f1_score", "r2"]), metrics[0])
        
        if primary_metric.value > 0.9:
            insights.append(f"Excellent {primary_metric.display_name.lower()} of {primary_metric.value:.3f}")
        elif primary_metric.value > 0.7:
            insights.append(f"Good {primary_metric.display_name.lower()} of {primary_metric.value:.3f}")
        else:
            insights.append(f"Room for improvement with {primary_metric.display_name.lower()} of {primary_metric.value:.3f}")
        
        return insights
    
    def _generate_recommendations(self, metrics: List[MetricResult], algorithm_name: str, problem_type: ProblemTypeEnum) -> List[str]:
        """Generate recommendations for model improvement"""
        recommendations = []
        
        primary_metric = next((m for m in metrics if m.name in ["f1_score", "r2"]), metrics[0])
        
        if primary_metric.value < 0.6:
            recommendations.append("Consider trying different algorithms or hyperparameter tuning")
            recommendations.append("Review feature engineering and data preprocessing steps")
        
        return recommendations
    
    def _estimate_model_complexity(self, model: Any, algorithm_name: str) -> str:
        """Estimate model complexity based on algorithm and parameters"""
        if "tree" in algorithm_name.lower():
            return "medium"
        elif "forest" in algorithm_name.lower():
            return "high"
        elif "linear" in algorithm_name.lower() or "logistic" in algorithm_name.lower():
            return "low"
        return "medium"
    
    def compare_models(self, analyses: List[ModelAnalysis]) -> Dict[str, Any]:
        """Compare multiple model analyses"""
        if not analyses:
            return {}
        
        # Extract primary metrics for comparison
        model_comparison = []
        for analysis in analyses:
            model_comparison.append({
                "algorithm_name": analysis.algorithm_name,
                "primary_metric_name": analysis.primary_metric.name,
                "primary_metric_value": analysis.primary_metric.value,
                "training_time": analysis.training_time,
                "complexity": analysis.model_complexity
            })
        
        # Sort by primary metric (assuming higher is better for most metrics)
        problem_type = analyses[0].problem_type
        if problem_type == "classification":
            model_comparison.sort(key=lambda x: x["primary_metric_value"], reverse=True)
        else:  # regression
            # For regression, R² is higher-better, but RMSE/MAE are lower-better
            first_metric = analyses[0].primary_metric.name
            reverse_sort = first_metric in ["r2"]  # Metrics where higher is better
            model_comparison.sort(key=lambda x: x["primary_metric_value"], reverse=reverse_sort)
        
        return {
            "best_model": model_comparison[0] if model_comparison else None,
            "model_ranking": model_comparison,
            "comparison_summary": {
                "total_models": len(analyses),
                "problem_type": problem_type,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }


# Convenience functions for external use

def evaluate_single_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    algorithm_name: str,
    problem_type: ProblemTypeEnum,
    **kwargs
) -> ModelAnalysis:
    """Convenience function for evaluating a single model"""
    evaluator = ModelEvaluationService()
    return evaluator.evaluate_model(
        model, X_test, y_test, algorithm_name, problem_type, **kwargs
    )


def compare_multiple_models(
    evaluations: List[ModelAnalysis]
) -> Dict[str, Any]:
    """Convenience function for comparing multiple models"""
    evaluator = ModelEvaluationService()
    return evaluator.compare_models(evaluations)


# Legacy function names for backward compatibility
def evaluate_model_comprehensive(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    algorithm_name: str,
    problem_type: ProblemTypeEnum,
    **kwargs
) -> ModelAnalysis:
    """Legacy function name - use evaluate_single_model instead"""
    return evaluate_single_model(model, X_test, y_test, algorithm_name, problem_type, **kwargs)


def compare_models(evaluations: List[ModelAnalysis]) -> Dict[str, Any]:
    """Legacy function name - use compare_multiple_models instead"""
    return compare_multiple_models(evaluations)


def export_evaluation_report(analysis: ModelAnalysis, format: str = "dict") -> Dict[str, Any]:
    """Export evaluation report (simplified implementation)"""
    return {
        "algorithm_name": analysis.algorithm_name,
        "problem_type": analysis.problem_type,
        "primary_metric": {
            "name": analysis.primary_metric.name,
            "value": analysis.primary_metric.value,
            "display_name": analysis.primary_metric.display_name
        },
        "all_metrics": [
            {
                "name": m.name,
                "value": m.value,
                "display_name": m.display_name,
                "description": m.description
            }
            for m in analysis.metrics
        ],
        "insights": analysis.insights,
        "recommendations": analysis.recommendations,
        "training_time": analysis.training_time,
        "evaluation_timestamp": analysis.evaluation_timestamp
    }


# Type alias for backward compatibility
ComprehensiveEvaluation = ModelAnalysis 