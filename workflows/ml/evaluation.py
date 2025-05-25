"""
<<<<<<< HEAD
Model Evaluation Service for DS1.3.2
Advanced model evaluation capabilities with comprehensive metrics, visualizations, and analysis
"""

import pandas as pd
import numpy as np
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
=======
DS1.3.2: Model Evaluation Service
Advanced model evaluation with comprehensive metrics, visualizations, and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime
>>>>>>> d5136a25c2ce03de6747e77d05de13579bf5c0e4
import sys
import os

# Scientific computing and ML libraries
from sklearn.metrics import (
<<<<<<< HEAD
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, log_loss,
    
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, mean_squared_log_error, median_absolute_error,
    
    # General metrics
    mean_poisson_deviance, mean_gamma_deviance
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
=======
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
>>>>>>> d5136a25c2ce03de6747e77d05de13579bf5c0e4

# Handle matplotlib and seaborn imports gracefully
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Create mock matplotlib for testing
    class MockPlt:
        def figure(self, *args, **kwargs): return None
        def plot(self, *args, **kwargs): return None
        def barh(self, *args, **kwargs): return []
        def scatter(self, *args, **kwargs): return None
        def axhline(self, *args, **kwargs): return None
        def xlabel(self, *args, **kwargs): return None
        def ylabel(self, *args, **kwargs): return None
        def title(self, *args, **kwargs): return None
        def legend(self, *args, **kwargs): return None
        def grid(self, *args, **kwargs): return None
        def text(self, *args, **kwargs): return None
        def tight_layout(self, *args, **kwargs): return None
        def savefig(self, *args, **kwargs): return None
        def close(self, *args, **kwargs): return None
        def gca(self): return MockAxes()
        def yticks(self, *args, **kwargs): return None
        
        class style:
            available = []
            def use(self, style): pass
    
    class MockAxes:
        def invert_yaxis(self): return None
        def set_color(self, *args, **kwargs): return None
        @property 
        def transAxes(self): return None
    
    class MockSeaborn:
        def heatmap(self, *args, **kwargs): return None
        def set_palette(self, *args, **kwargs): return None
    
    plt = MockPlt()
    sns = MockSeaborn()

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

<<<<<<< HEAD
from models.ml_models import ProblemTypeEnum, AlgorithmNameEnum


@dataclass
class DetailedMetrics:
    """Detailed metrics for model evaluation"""
    primary_metrics: Dict[str, float]
    secondary_metrics: Dict[str, float]
    cross_validation_scores: Optional[Dict[str, List[float]]]
    feature_importance: Optional[Dict[str, float]]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]]


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics"""
    # Primary metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    
    # Secondary metrics  
    balanced_accuracy: float
    average_precision: Optional[float]
    cohen_kappa: float
    matthews_corrcoef: float
    log_loss_score: Optional[float]
    
    # Per-class metrics
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_score_per_class: Dict[str, float]
    
    # Curve data
    roc_curve_data: Optional[Dict[str, np.ndarray]]
    pr_curve_data: Optional[Dict[str, np.ndarray]]
    calibration_curve_data: Optional[Dict[str, np.ndarray]]
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    
    # Classification report
    classification_report: Dict[str, Any]


@dataclass
class RegressionMetrics:
    """Comprehensive regression metrics"""
    # Primary metrics
    r2_score: float
    mae: float
    mse: float
    rmse: float
    
    # Secondary metrics
    mape: float  # Mean Absolute Percentage Error
    explained_variance: float
    max_error: float
    median_absolute_error: float
    mean_squared_log_error: Optional[float]  # Only for positive targets
    
    # Additional metrics
    mean_poisson_deviance: Optional[float]
    mean_gamma_deviance: Optional[float]
    
    # Residual analysis
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    residual_statistics: Dict[str, float]


@dataclass
class ModelDiagnostics:
    """Model diagnostic information"""
    training_time: float
    prediction_time: float
    model_complexity: Dict[str, Any]  # Model-specific complexity metrics
    memory_usage: Optional[float]
    stability_metrics: Optional[Dict[str, float]]  # Cross-validation stability
    feature_importance_stability: Optional[Dict[str, float]]


@dataclass
class ComprehensiveEvaluation:
    """Complete evaluation result for a model"""
    algorithm_name: str
    problem_type: str
    metrics: Union[ClassificationMetrics, RegressionMetrics]
    diagnostics: ModelDiagnostics
    feature_importance: Optional[Dict[str, float]]
    permutation_importance: Optional[Dict[str, float]]
    cross_validation_results: Optional[Dict[str, Any]]
    error: Optional[str] = None


class ModelEvaluator:
    """
    Advanced model evaluation service with comprehensive metrics and analysis
    """
    
    def __init__(self, problem_type: ProblemTypeEnum, cv_folds: int = 5):
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        
        # Suppress sklearn warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    def evaluate_classification_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                    X_train: Optional[pd.DataFrame] = None, 
                                    y_train: Optional[pd.Series] = None) -> ClassificationMetrics:
        """Comprehensive classification model evaluation"""
=======
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
>>>>>>> d5136a25c2ce03de6747e77d05de13579bf5c0e4
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
<<<<<<< HEAD
        if hasattr(model, 'predict_proba'):
=======
        if hasattr(model, 'predict_proba') and problem_type == ProblemTypeEnum.CLASSIFICATION:
>>>>>>> d5136a25c2ce03de6747e77d05de13579bf5c0e4
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
<<<<<<< HEAD
        # Primary metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (handle binary and multi-class)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        # Secondary metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        matthews_cc = matthews_corrcoef(y_test, y_pred)
        
        # Average precision score
        avg_precision = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
                else:
                    avg_precision = average_precision_score(y_test, y_pred_proba, average='weighted')
            except:
                pass
        
        # Log loss
        log_loss_val = None
        if y_pred_proba is not None:
            try:
                log_loss_val = log_loss(y_test, y_pred_proba)
            except:
                pass
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        
        for label in class_report:
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                precision_per_class[str(label)] = class_report[label]['precision']
                recall_per_class[str(label)] = class_report[label]['recall']
                f1_per_class[str(label)] = class_report[label]['f1-score']
        
        # Confusion matrices
        conf_matrix = confusion_matrix(y_test, y_pred)
        norm_conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
        
        # ROC curve data
        roc_curve_data = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
                roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            except:
                pass
        
        # Precision-Recall curve data
        pr_curve_data = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
                pr_curve_data = {'precision': precision_curve, 'recall': recall_curve, 'thresholds': thresholds}
            except:
                pass
        
        # Calibration curve data
        calibration_data = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                fraction_pos, mean_pred_value = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
                calibration_data = {'fraction_positive': fraction_pos, 'mean_predicted_value': mean_pred_value}
            except:
                pass
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            balanced_accuracy=balanced_acc,
            average_precision=avg_precision,
            cohen_kappa=cohen_kappa,
            matthews_corrcoef=matthews_cc,
            log_loss_score=log_loss_val,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_score_per_class=f1_per_class,
            roc_curve_data=roc_curve_data,
            pr_curve_data=pr_curve_data,
            calibration_curve_data=calibration_data,
            confusion_matrix=conf_matrix,
            normalized_confusion_matrix=norm_conf_matrix,
            classification_report=class_report
        )
    
    def evaluate_regression_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                X_train: Optional[pd.DataFrame] = None,
                                y_train: Optional[pd.Series] = None) -> RegressionMetrics:
        """Comprehensive regression model evaluation"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Primary metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Secondary metrics
        mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-8, None))) * 100
        explained_var = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)
        
        # Mean squared log error (only for positive targets)
        msle = None
        if np.all(y_test >= 0) and np.all(y_pred >= 0):
            try:
                msle = mean_squared_log_error(y_test, np.maximum(y_pred, 0))
            except:
                pass
        
        # Additional metrics (for positive targets)
        poisson_dev = None
        gamma_dev = None
        if np.all(y_test > 0) and np.all(y_pred > 0):
            try:
                poisson_dev = mean_poisson_deviance(y_test, y_pred)
                gamma_dev = mean_gamma_deviance(y_test, y_pred)
            except:
                pass
        
        # Residual analysis
        residuals = y_test - y_pred
        
        # Standardized residuals
        residual_std = np.std(residuals)
        standardized_residuals = residuals / (residual_std + 1e-8)
        
        # Residual statistics
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(self._calculate_skewness(residuals)),
            'kurtosis': float(self._calculate_kurtosis(residuals)),
            'jarque_bera_pvalue': float(self._jarque_bera_test(residuals)),
            'autocorrelation': float(self._calculate_autocorrelation(residuals))
        }
        
        return RegressionMetrics(
            r2_score=r2,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            explained_variance=explained_var,
            max_error=max_err,
            median_absolute_error=median_ae,
            mean_squared_log_error=msle,
            mean_poisson_deviance=poisson_dev,
            mean_gamma_deviance=gamma_dev,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            residual_statistics=residual_stats
        )
    
    def calculate_feature_importance(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                   method: str = 'permutation') -> Dict[str, float]:
        """Calculate feature importance using various methods"""
        
        feature_importance = {}
        
        # Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) == 1:
                feature_importance = dict(zip(X_test.columns, np.abs(model.coef_)))
            else:
                # Multi-class: average across classes
                feature_importance = dict(zip(X_test.columns, np.mean(np.abs(model.coef_), axis=0)))
        
        # Permutation importance (more robust but computationally expensive)
        if method == 'permutation':
            try:
                perm_importance = permutation_importance(
                    model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
                )
                perm_importance_dict = dict(zip(X_test.columns, perm_importance.importances_mean))
                
                # Combine or use permutation importance if no built-in available
                if not feature_importance:
                    feature_importance = perm_importance_dict
                
                return feature_importance, perm_importance_dict
            except:
                pass
        
        return feature_importance, None
    
    def perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation analysis"""
        
        cv_results = {}
        
        try:
            # Primary metric cross-validation
            if self.problem_type == ProblemTypeEnum.CLASSIFICATION:
                scoring_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
            else:
                scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=metric)
                    cv_results[metric] = {
                        'scores': scores.tolist(),
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    }
                except:
                    continue
            
            # Learning curve analysis
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=self.cv_folds, n_jobs=1,
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )
                
                cv_results['learning_curve'] = {
                    'train_sizes': train_sizes.tolist(),
                    'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                    'train_scores_std': np.std(train_scores, axis=1).tolist(),
                    'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                    'val_scores_std': np.std(val_scores, axis=1).tolist()
                }
            except:
                pass
            
        except Exception as e:
            cv_results['error'] = str(e)
        
        return cv_results
    
    def calculate_model_diagnostics(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  training_time: float) -> ModelDiagnostics:
        """Calculate model diagnostic metrics"""
        
        import time
        
        # Prediction time
        start_time = time.time()
        _ = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Model complexity metrics
        complexity = {}
        
        # Get model-specific complexity metrics
        if hasattr(model, 'n_features_in_'):
            complexity['n_features'] = model.n_features_in_
        
        if hasattr(model, 'tree_'):  # Decision Tree
            complexity['n_nodes'] = model.tree_.node_count
            complexity['max_depth'] = model.tree_.max_depth
        
        if hasattr(model, 'estimators_'):  # Random Forest
            complexity['n_estimators'] = len(model.estimators_)
            if hasattr(model.estimators_[0], 'tree_'):
                complexity['avg_tree_depth'] = np.mean([est.tree_.max_depth for est in model.estimators_])
        
        if hasattr(model, 'coef_'):  # Linear models
            complexity['n_coefficients'] = np.prod(model.coef_.shape)
            complexity['l1_norm'] = float(np.sum(np.abs(model.coef_)))
            complexity['l2_norm'] = float(np.sqrt(np.sum(model.coef_ ** 2)))
        
        if hasattr(model, 'support_vectors_'):  # SVM
            complexity['n_support_vectors'] = len(model.support_vectors_)
        
        # Memory usage (approximate)
        memory_usage = None
        try:
            import pickle
            memory_usage = len(pickle.dumps(model)) / 1024  # KB
        except:
            pass
        
        return ModelDiagnostics(
            training_time=training_time,
            prediction_time=prediction_time,
            model_complexity=complexity,
            memory_usage=memory_usage,
            stability_metrics=None,  # Could be filled with CV stability analysis
            feature_importance_stability=None
        )
    
    def comprehensive_evaluate(self, model, algorithm_name: str,
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             training_time: float = 0.0) -> ComprehensiveEvaluation:
        """Perform comprehensive model evaluation"""
        
        try:
            # Core metrics evaluation
            if self.problem_type == ProblemTypeEnum.CLASSIFICATION:
                metrics = self.evaluate_classification_model(model, X_test, y_test, X_train, y_train)
            else:
                metrics = self.evaluate_regression_model(model, X_test, y_test, X_train, y_train)
            
            # Feature importance
            feature_importance, perm_importance = self.calculate_feature_importance(model, X_test, y_test)
            
            # Cross-validation (using full dataset)
            X_full = pd.concat([X_train, X_test])
            y_full = pd.concat([y_train, y_test])
            cv_results = self.perform_cross_validation(model, X_full, y_full)
            
            # Model diagnostics
            diagnostics = self.calculate_model_diagnostics(model, X_train, y_train, X_test, y_test, training_time)
            
            return ComprehensiveEvaluation(
                algorithm_name=algorithm_name,
                problem_type=self.problem_type.value,
                metrics=metrics,
                diagnostics=diagnostics,
                feature_importance=feature_importance,
                permutation_importance=perm_importance,
                cross_validation_results=cv_results
            )
            
        except Exception as e:
            return ComprehensiveEvaluation(
                algorithm_name=algorithm_name,
                problem_type=self.problem_type.value,
                metrics=None,
                diagnostics=None,
                feature_importance=None,
                permutation_importance=None,
                cross_validation_results=None,
                error=str(e)
            )
    
    # Helper methods for statistical calculations
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        if std == 0:
            return 0.0
        
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        if std == 0:
            return 0.0
        
        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurtosis
    
    def _jarque_bera_test(self, data: np.ndarray) -> float:
        """Jarque-Bera test for normality (returns p-value)"""
        n = len(data)
        if n < 8:
            return 1.0
        
        try:
            from scipy import stats
            _, p_value = stats.jarque_bera(data)
            return p_value
        except:
            # Fallback calculation
            skewness = self._calculate_skewness(data)
            kurtosis = self._calculate_kurtosis(data)
            jb_stat = n * (skewness**2 / 6 + (kurtosis)**2 / 24)
            
            # Approximate p-value using chi-square distribution
            try:
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(jb_stat, 2)
                return p_value
            except:
                return 0.5  # Default neutral p-value
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        n = len(data)
        if n <= lag:
            return 0.0
        
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / n
        
        if c0 == 0:
            return 0.0
        
        c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / (n - lag)
        
        return c_lag / c0


# Utility functions for external integration

def evaluate_model_comprehensive(model, algorithm_name: str, problem_type: str,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               training_time: float = 0.0) -> ComprehensiveEvaluation:
    """
    Convenience function for comprehensive model evaluation
    
    Args:
        model: Trained scikit-learn model
        algorithm_name: Name of the algorithm
        problem_type: 'classification' or 'regression'
        X_train: Training features
        y_train: Training target
        X_test: Test features  
        y_test: Test target
        training_time: Time taken to train the model
    
    Returns:
        ComprehensiveEvaluation object with all metrics and analysis
    """
    problem_enum = ProblemTypeEnum(problem_type.upper())
    evaluator = ModelEvaluator(problem_enum)
    
    return evaluator.comprehensive_evaluate(
        model, algorithm_name, X_train, y_train, X_test, y_test, training_time
    )


def compare_models(evaluations: List[ComprehensiveEvaluation]) -> Dict[str, Any]:
    """
    Compare multiple model evaluations and provide insights
    
    Args:
        evaluations: List of ComprehensiveEvaluation objects
    
    Returns:
        Dictionary with comparison metrics and recommendations
    """
    valid_evaluations = [eval_result for eval_result in evaluations if not eval_result.error]
    
    if not valid_evaluations:
        return {"error": "No valid evaluations to compare"}
    
    problem_type = valid_evaluations[0].problem_type
    
    comparison = {
        "problem_type": problem_type,
        "models_compared": len(valid_evaluations),
        "model_rankings": {},
        "best_models": {},
        "performance_summary": {}
    }
    
    # Extract primary metrics for comparison
    if problem_type == "CLASSIFICATION":
        primary_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    else:
        primary_metrics = ['r2_score', 'mae', 'mse', 'rmse']
    
    # Rank models by each metric
    for metric in primary_metrics:
        rankings = []
        for eval_result in valid_evaluations:
            if hasattr(eval_result.metrics, metric):
                value = getattr(eval_result.metrics, metric)
                rankings.append((eval_result.algorithm_name, value))
        
        # Sort (higher is better for most metrics, except error metrics)
        reverse = metric not in ['mae', 'mse', 'rmse', 'log_loss_score']
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        comparison["model_rankings"][metric] = rankings
        if rankings:
            comparison["best_models"][metric] = rankings[0][0]
    
    # Overall performance summary
    for eval_result in valid_evaluations:
        algo_name = eval_result.algorithm_name
        comparison["performance_summary"][algo_name] = {
            "training_time": eval_result.diagnostics.training_time if eval_result.diagnostics else 0,
            "prediction_time": eval_result.diagnostics.prediction_time if eval_result.diagnostics else 0,
            "model_complexity": eval_result.diagnostics.model_complexity if eval_result.diagnostics else {},
            "cross_validation_stability": None  # Could add CV std analysis here
        }
        
        # Add primary metrics
        if problem_type == "CLASSIFICATION" and eval_result.metrics:
            comparison["performance_summary"][algo_name].update({
                "accuracy": eval_result.metrics.accuracy,
                "f1_score": eval_result.metrics.f1_score,
                "roc_auc": eval_result.metrics.roc_auc
            })
        elif problem_type == "REGRESSION" and eval_result.metrics:
            comparison["performance_summary"][algo_name].update({
                "r2_score": eval_result.metrics.r2_score,
                "mae": eval_result.metrics.mae,
                "rmse": eval_result.metrics.rmse
            })
    
    return comparison


def export_evaluation_report(evaluation: ComprehensiveEvaluation, 
                           output_path: str = None) -> Dict[str, Any]:
    """
    Export comprehensive evaluation report to JSON
    
    Args:
        evaluation: ComprehensiveEvaluation object
        output_path: Optional path to save JSON report
    
    Returns:
        Dictionary representation of the evaluation
    """
    # Convert evaluation to dictionary (handling numpy arrays)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Create comprehensive report
    report = {
        "algorithm_name": evaluation.algorithm_name,
        "problem_type": evaluation.problem_type,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "error": evaluation.error
    }
    
    if not evaluation.error:
        # Metrics
        if evaluation.metrics:
            metrics_dict = asdict(evaluation.metrics)
            # Convert numpy arrays
            for key, value in metrics_dict.items():
                metrics_dict[key] = convert_numpy(value)
            report["metrics"] = metrics_dict
        
        # Diagnostics
        if evaluation.diagnostics:
            report["diagnostics"] = asdict(evaluation.diagnostics)
        
        # Feature importance
        if evaluation.feature_importance:
            report["feature_importance"] = evaluation.feature_importance
        
        if evaluation.permutation_importance:
            report["permutation_importance"] = evaluation.permutation_importance
        
        # Cross-validation results
        if evaluation.cross_validation_results:
            report["cross_validation"] = evaluation.cross_validation_results
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report 
=======
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
            
            # Average precision if binary classification with probabilities
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
                    metrics.append(MetricResult(
                        name="avg_precision",
                        value=float(avg_precision),
                        display_name="Average Precision",
                        description="Area under the precision-recall curve",
                        higher_is_better=True,
                        category="performance"
                    ))
                except Exception:
                    pass
        
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
            
            # Mean Absolute Percentage Error
            try:
                mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
                metrics.append(MetricResult(
                    name="mape",
                    value=float(mape),
                    display_name="Mean Absolute Percentage Error",
                    description="Average absolute percentage error",
                    higher_is_better=False,
                    category="performance"
                ))
            except Exception:
                pass
        
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
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(labels):
            if i < cm.shape[0] and i < cm.shape[1]:
                tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
                fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                tn = cm.sum() - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[str(label)] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "support": int(np.sum(y_true == label))
                }
        
        return {
            "matrix": cm.tolist(),
            "labels": [str(label) for label in labels],
            "per_class_metrics": per_class_metrics,
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
    
    def _create_visualizations(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        model: Any,
        algorithm_name: str,
        problem_type: ProblemTypeEnum,
        feature_importance: Optional[Dict[str, float]],
        confusion_data: Optional[Dict[str, Any]]
    ) -> List[VisualizationData]:
        """Create visualizations for model analysis"""
        visualizations = []
        
        # Feature importance plot
        if feature_importance:
            viz_data = self._create_feature_importance_plot(feature_importance, algorithm_name)
            if viz_data:
                visualizations.append(viz_data)
        
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            # Confusion matrix heatmap
            if confusion_data:
                viz_data = self._create_confusion_matrix_plot(confusion_data, algorithm_name)
                if viz_data:
                    visualizations.append(viz_data)
            
            # ROC curve for binary classification
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                viz_data = self._create_roc_curve_plot(y_true, y_pred_proba[:, 1], algorithm_name)
                if viz_data:
                    visualizations.append(viz_data)
                
                # Precision-Recall curve
                viz_data = self._create_pr_curve_plot(y_true, y_pred_proba[:, 1], algorithm_name)
                if viz_data:
                    visualizations.append(viz_data)
        
        else:  # Regression
            # Residual plot
            viz_data = self._create_residual_plot(y_true, y_pred, algorithm_name)
            if viz_data:
                visualizations.append(viz_data)
            
            # Prediction vs actual plot
            viz_data = self._create_prediction_plot(y_true, y_pred, algorithm_name)
            if viz_data:
                visualizations.append(viz_data)
        
        return visualizations
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float], algorithm_name: str) -> Optional[VisualizationData]:
        """Create feature importance visualization"""
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:min(15, len(sorted_features))]  # Top 15 features
            
            if not top_features:
                return None
            
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {algorithm_name}')
            plt.gca().invert_yaxis()
            
            # Color bars by importance magnitude
            for bar, importance in zip(bars, importances):
                if importance > 0:
                    bar.set_color('darkgreen' if importance > np.mean(importances) else 'lightgreen')
                else:
                    bar.set_color('darkred' if abs(importance) > np.mean(np.abs(importances)) else 'lightcoral')
            
            plt.tight_layout()
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_feature_importance.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="feature_importance",
                data={
                    "features": list(features),
                    "importances": list(importances),
                    "top_feature": features[0] if features else None,
                    "max_importance": max(importances) if importances else 0
                },
                title=f"Feature Importance - {algorithm_name}",
                description="Relative importance of features in the trained model",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _create_confusion_matrix_plot(self, confusion_data: Dict[str, Any], algorithm_name: str) -> Optional[VisualizationData]:
        """Create confusion matrix heatmap"""
        try:
            cm = np.array(confusion_data["matrix"])
            labels = confusion_data["labels"]
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - {algorithm_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_confusion_matrix.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="confusion_matrix",
                data=confusion_data,
                title=f"Confusion Matrix - {algorithm_name}",
                description="Matrix showing correct and incorrect predictions by class",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _create_roc_curve_plot(self, y_true: pd.Series, y_scores: np.ndarray, algorithm_name: str) -> Optional[VisualizationData]:
        """Create ROC curve plot for binary classification"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'{algorithm_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {algorithm_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_roc_curve.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="roc_curve",
                data={
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                    "auc": float(auc_score)
                },
                title=f"ROC Curve - {algorithm_name}",
                description="Receiver Operating Characteristic curve showing model performance",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _create_pr_curve_plot(self, y_true: pd.Series, y_scores: np.ndarray, algorithm_name: str) -> Optional[VisualizationData]:
        """Create Precision-Recall curve plot"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label=f'{algorithm_name} (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {algorithm_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_pr_curve.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="precision_recall_curve",
                data={
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                    "average_precision": float(avg_precision)
                },
                title=f"Precision-Recall Curve - {algorithm_name}",
                description="Precision-Recall curve showing trade-off between precision and recall",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _create_residual_plot(self, y_true: pd.Series, y_pred: np.ndarray, algorithm_name: str) -> Optional[VisualizationData]:
        """Create residual plot for regression"""
        try:
            residuals = y_true - y_pred
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {algorithm_name}')
            plt.grid(True, alpha=0.3)
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_residuals.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="residual_plot",
                data={
                    "predicted": y_pred.tolist(),
                    "residuals": residuals.tolist(),
                    "rmse": float(np.sqrt(np.mean(residuals**2)))
                },
                title=f"Residual Plot - {algorithm_name}",
                description="Plot of residuals vs predicted values to check for patterns",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _create_prediction_plot(self, y_true: pd.Series, y_pred: np.ndarray, algorithm_name: str) -> Optional[VisualizationData]:
        """Create prediction vs actual plot for regression"""
        try:
            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, y_pred, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=2)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Prediction vs Actual - {algorithm_name}')
            plt.grid(True, alpha=0.3)
            
            # Add R² annotation
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
            
            file_path = None
            if self.save_visualizations:
                file_path = os.path.join(self.viz_dir, f"{algorithm_name}_predictions.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            return VisualizationData(
                chart_type="prediction_plot",
                data={
                    "actual": y_true.tolist(),
                    "predicted": y_pred.tolist(),
                    "r2": float(r2)
                },
                title=f"Prediction vs Actual - {algorithm_name}",
                description="Scatter plot comparing predicted vs actual values",
                file_path=file_path
            )
        
        except Exception as e:
            return None
    
    def _generate_insights(
        self,
        metrics: List[MetricResult],
        feature_importance: Optional[Dict[str, float]],
        confusion_data: Optional[Dict[str, Any]],
        problem_type: ProblemTypeEnum
    ) -> List[str]:
        """Generate insights about model performance"""
        insights = []
        
        # Performance insights
        primary_metric = next((m for m in metrics if m.name in ["f1_score", "r2"]), metrics[0])
        
        if primary_metric.value > 0.9:
            insights.append(f"Excellent {primary_metric.display_name.lower()} of {primary_metric.value:.3f} indicates very strong model performance")
        elif primary_metric.value > 0.7:
            insights.append(f"Good {primary_metric.display_name.lower()} of {primary_metric.value:.3f} indicates solid model performance")
        elif primary_metric.value > 0.5:
            insights.append(f"Moderate {primary_metric.display_name.lower()} of {primary_metric.value:.3f} suggests room for improvement")
        else:
            insights.append(f"Low {primary_metric.display_name.lower()} of {primary_metric.value:.3f} indicates poor model performance")
        
        # Feature importance insights
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feature = sorted_features[0]
            insights.append(f"Most important feature: '{top_feature[0]}' (importance: {top_feature[1]:.3f})")
            
            # Feature distribution insights
            total_importance = sum(abs(imp) for imp in feature_importance.values())
            top_3_importance = sum(abs(imp) for _, imp in sorted_features[:3])
            if len(sorted_features) > 3:
                concentration = top_3_importance / total_importance
                if concentration > 0.8:
                    insights.append("Model heavily relies on top 3 features - consider feature engineering")
                elif concentration < 0.3:
                    insights.append("Feature importance is well distributed across many features")
        
        # Classification-specific insights
        if problem_type == ProblemTypeEnum.CLASSIFICATION and confusion_data:
            accuracy = confusion_data.get("accuracy", 0)
            if "per_class_metrics" in confusion_data:
                class_f1s = [metrics["f1_score"] for metrics in confusion_data["per_class_metrics"].values()]
                if class_f1s:
                    f1_std = np.std(class_f1s)
                    if f1_std > 0.2:
                        insights.append("Significant class imbalance detected - consider balancing techniques")
                    elif f1_std < 0.05:
                        insights.append("Balanced performance across all classes")
        
        # Regression-specific insights
        if problem_type == ProblemTypeEnum.REGRESSION:
            r2_metric = next((m for m in metrics if m.name == "r2"), None)
            mape_metric = next((m for m in metrics if m.name == "mape"), None)
            
            if r2_metric and r2_metric.value < 0:
                insights.append("Negative R² indicates model performs worse than simple mean prediction")
            
            if mape_metric:
                if mape_metric.value < 10:
                    insights.append("Low MAPE indicates highly accurate predictions")
                elif mape_metric.value > 50:
                    insights.append("High MAPE suggests significant prediction errors")
        
        return insights
    
    def _generate_recommendations(
        self,
        metrics: List[MetricResult],
        algorithm_name: str,
        problem_type: ProblemTypeEnum
    ) -> List[str]:
        """Generate recommendations for model improvement"""
        recommendations = []
        
        # Performance-based recommendations
        primary_metric = next((m for m in metrics if m.name in ["f1_score", "r2"]), metrics[0])
        
        if primary_metric.value < 0.6:
            recommendations.append("Consider trying different algorithms or hyperparameter tuning")
            recommendations.append("Review feature engineering and data preprocessing steps")
        
        # Algorithm-specific recommendations
        if "tree" in algorithm_name.lower():
            if primary_metric.value < 0.7:
                recommendations.append("Try ensemble methods like Random Forest or Gradient Boosting")
            recommendations.append("Consider pruning parameters to prevent overfitting")
        
        elif "logistic" in algorithm_name.lower() or "linear" in algorithm_name.lower():
            recommendations.append("Ensure features are properly scaled and normalized")
            if primary_metric.value < 0.7:
                recommendations.append("Consider polynomial features or interaction terms")
        
        elif "forest" in algorithm_name.lower():
            recommendations.append("Tune n_estimators and max_depth for optimal performance")
            recommendations.append("Consider feature selection to reduce model complexity")
        
        elif "svm" in algorithm_name.lower() or "svc" in algorithm_name.lower():
            recommendations.append("Experiment with different kernel types (rbf, linear, polynomial)")
            recommendations.append("Tune C and gamma parameters for better performance")
        
        elif "knn" in algorithm_name.lower():
            recommendations.append("Optimize k value using cross-validation")
            recommendations.append("Ensure features are scaled due to distance-based nature")
        
        # Problem-specific recommendations
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            precision_metric = next((m for m in metrics if m.name == "precision"), None)
            recall_metric = next((m for m in metrics if m.name == "recall"), None)
            
            if precision_metric and recall_metric:
                if precision_metric.value > recall_metric.value + 0.1:
                    recommendations.append("Consider adjusting decision threshold to improve recall")
                elif recall_metric.value > precision_metric.value + 0.1:
                    recommendations.append("Consider adjusting decision threshold to improve precision")
        
        return recommendations
    
    def _estimate_model_complexity(self, model: Any, algorithm_name: str) -> str:
        """Estimate model complexity based on algorithm and parameters"""
        
        if "tree" in algorithm_name.lower():
            if hasattr(model, 'tree_'):
                n_nodes = model.tree_.node_count
                if n_nodes < 20:
                    return "low"
                elif n_nodes < 100:
                    return "medium"
                else:
                    return "high"
        
        elif "forest" in algorithm_name.lower():
            if hasattr(model, 'n_estimators'):
                if model.n_estimators < 50:
                    return "medium"
                elif model.n_estimators < 200:
                    return "high"
                else:
                    return "very_high"
        
        elif "svm" in algorithm_name.lower():
            if hasattr(model, 'kernel'):
                if model.kernel == 'linear':
                    return "low"
                elif model.kernel in ['rbf', 'polynomial']:
                    return "high"
        
        elif "linear" in algorithm_name.lower() or "logistic" in algorithm_name.lower():
            return "low"
        
        elif "knn" in algorithm_name.lower():
            return "medium"
        
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
                "complexity": analysis.model_complexity,
                "num_insights": len(analysis.insights),
                "num_recommendations": len(analysis.recommendations)
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
>>>>>>> d5136a25c2ce03de6747e77d05de13579bf5c0e4
