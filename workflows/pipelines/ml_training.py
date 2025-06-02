"""
ML Training Pipeline for DS1.3.1
Main Prefect workflow for training multiple ML algorithms, evaluating performance, and aggregating results
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os
import joblib
import traceback

# Scientific computing and ML libraries
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

# Prefect for workflow orchestration
try:
    from prefect import task, flow, get_run_logger
    from prefect.task_runners import SequentialTaskRunner
    PREFECT_AVAILABLE = True
except ImportError:
    # Fallback when Prefect is not available
    PREFECT_AVAILABLE = False
    
    def task(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def flow(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    
    def get_run_logger():
        return MockLogger()
    
    class SequentialTaskRunner:
        def __init__(self):
            pass

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

# Import models and utilities - use try/except for IDE compatibility
try:
    from models.ml_models import (  # type: ignore
        ProblemTypeEnum, AlgorithmNameEnum, MLPipelineConfig,
        MLPipelineRun, MLResult, ModelMetrics
    )
except ImportError:
    # Fallback for when running from different contexts
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
    from models.ml_models import (  # type: ignore
        ProblemTypeEnum, AlgorithmNameEnum, MLPipelineConfig,
        MLPipelineRun, MLResult, ModelMetrics
    )

try:
    from algorithm_registry import get_algorithm_registry  # type: ignore
except ImportError:
    # Fallback for IDE/different execution contexts
    from ..ml.algorithm_registry import get_algorithm_registry

try:
    from preprocessing import data_preprocessing_flow, PreprocessingResult  # type: ignore
except ImportError:
    # Fallback for IDE/different execution contexts
    from ..ml.preprocessing import data_preprocessing_flow, PreprocessingResult


@dataclass
class TrainingResult:
    """Result of training a single algorithm"""
    algorithm_name: str
    model: Any  # Trained scikit-learn model
    training_time: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    model_path: Optional[str]  # Path where model is saved
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a trained model"""
    algorithm_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    error: Optional[str] = None


@dataclass
class MLTrainingResult:
    """Complete result of ML training pipeline"""
    pipeline_run_id: str
    problem_type: str
    target_variable: str
    preprocessing_result: PreprocessingResult
    training_results: List[TrainingResult]
    evaluation_results: List[EvaluationResult]
    best_model: Dict[str, Any]
    aggregated_metrics: Dict[str, Any]
    total_training_time: float
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        try:
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types"""
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):  # Handle numpy boolean
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.complexfloating):  # Handle complex numbers
                    return str(obj)  # Convert complex to string since JSON doesn't support complex
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, tuple):
                    return [convert_numpy_types(v) for v in obj]  # Convert tuples to lists
                elif hasattr(obj, '__module__') and 'pandas' in str(obj.__module__) and hasattr(obj, 'to_list'):
                    return obj.to_list()  # Handle pandas Series
                elif hasattr(obj, '__module__') and 'pandas' in str(obj.__module__) and hasattr(obj, 'to_dict'):
                    return obj.to_dict('records')  # Handle pandas DataFrame
                return obj
            
            result = {
                "pipeline_run_id": self.pipeline_run_id,
                "problem_type": self.problem_type,
                "target_variable": self.target_variable,
                "preprocessing_result": self.preprocessing_result.to_dict(),
                "training_results": [
                    {
                        "algorithm_name": tr.algorithm_name,
                        "training_time": float(tr.training_time),
                        "hyperparameters": convert_numpy_types(tr.hyperparameters),
                        "feature_importance": convert_numpy_types(tr.feature_importance),
                        "model_path": tr.model_path,
                        "error": tr.error
                    } for tr in self.training_results
                ],
                "evaluation_results": [
                    {
                        "algorithm_name": er.algorithm_name,
                        "metrics": convert_numpy_types(er.metrics),
                        "feature_importance": convert_numpy_types(er.feature_importance),
                        "error": er.error
                    } for er in self.evaluation_results
                ],
                "best_model": convert_numpy_types(self.best_model),
                "aggregated_metrics": convert_numpy_types(self.aggregated_metrics),
                "total_training_time": float(self.total_training_time),
                "summary": convert_numpy_types(self.summary)
            }
            
            return convert_numpy_types(result)
            
        except Exception as e:
            # Fallback to basic representation if conversion fails
            return {
                "pipeline_run_id": self.pipeline_run_id,
                "problem_type": self.problem_type,
                "target_variable": self.target_variable,
                "summary": {"conversion_error": str(e)},
                "conversion_error": str(e)
            }


class MLModelTrainer:
    """
    Main class for training ML models with multiple algorithms
    """
    
    def __init__(self, problem_type: ProblemTypeEnum, models_save_dir: str = "trained_models"):
        self.problem_type = problem_type
        self.models_save_dir = Path(models_save_dir)
        self.models_save_dir.mkdir(exist_ok=True)
        self.algorithm_registry = get_algorithm_registry()
        self.logger = None
        
        # Algorithm mapping
        self.algorithm_classes = {
            # Classification algorithms
            AlgorithmNameEnum.LOGISTIC_REGRESSION: LogisticRegression,
            AlgorithmNameEnum.DECISION_TREE_CLASSIFIER: DecisionTreeClassifier,
            AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
            AlgorithmNameEnum.SVM_CLASSIFIER: SVC,
            AlgorithmNameEnum.KNN_CLASSIFIER: KNeighborsClassifier,
            
            # Regression algorithms
            AlgorithmNameEnum.LINEAR_REGRESSION: LinearRegression,
            AlgorithmNameEnum.DECISION_TREE_REGRESSOR: DecisionTreeRegressor,
            AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
            AlgorithmNameEnum.SVM_REGRESSOR: SVR,
            AlgorithmNameEnum.KNN_REGRESSOR: KNeighborsRegressor,
        }
    
    def set_logger(self, logger):
        """Set Prefect logger"""
        self.logger = logger
    
    def log(self, message: str, level: str = "info"):
        """Log message with fallback to print"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def get_algorithm_hyperparameters(self, algorithm_name: AlgorithmNameEnum, 
                                    custom_params: Optional[Dict[str, Any]] = None,
                                    pipeline_run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get hyperparameters for an algorithm with custom overrides"""
        # Use the algorithm registry to get hyperparameters with unique random seeds
        try:
            algo_config = self.algorithm_registry.create_algorithm_config(
                algorithm_name, 
                custom_params, 
                pipeline_run_id
            )
            return algo_config.hyperparameters
        except Exception as e:
            self.log(f"Error creating algorithm config for {algorithm_name.value}: {e}", "warning")
            
            # Fallback to original method if registry fails
            algorithm_def = self.algorithm_registry.get_algorithm(algorithm_name)
            
            if not algorithm_def:
                self.log(f"Algorithm {algorithm_name.value} not found in registry", "warning")
                return {}
            
            # Start with default hyperparameters
            hyperparams = {
                param.name: param.default for param in algorithm_def.hyperparameters
            }
            
            # Apply custom parameters
            if custom_params:
                hyperparams.update(custom_params)
                self.log(f"Applied custom hyperparameters for {algorithm_name.value}: {custom_params}")
            
            return hyperparams
    
    def train_single_algorithm(self, algorithm_name: AlgorithmNameEnum, 
                             X_train: pd.DataFrame, y_train: pd.Series,
                             hyperparameters: Dict[str, Any],
                             pipeline_run_id: str) -> TrainingResult:
        """Train a single ML algorithm"""
        self.log(f"Training {algorithm_name.value} with hyperparameters: {hyperparameters}")
        
        start_time = time.time()
        
        try:
            # Get algorithm class
            if algorithm_name not in self.algorithm_classes:
                raise ValueError(f"Algorithm {algorithm_name.value} not supported")
            
            AlgorithmClass = self.algorithm_classes[algorithm_name]
            
            # Create and train model
            model = AlgorithmClass(**hyperparameters)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Extract feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients as importance
                if model.coef_.ndim == 1:
                    feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
                else:
                    # For multi-class, take mean of absolute coefficients
                    feature_importance = dict(zip(X_train.columns, np.mean(np.abs(model.coef_), axis=0)))
            
            # Save model
            model_filename = f"{pipeline_run_id}_{algorithm_name.value}_{int(start_time)}.joblib"
            model_path = self.models_save_dir / model_filename
            joblib.dump(model, model_path)
            
            self.log(f"Successfully trained {algorithm_name.value} in {training_time:.2f}s")
            
            return TrainingResult(
                algorithm_name=algorithm_name.value,
                model=model,
                training_time=training_time,
                hyperparameters=hyperparameters,
                feature_importance=feature_importance,
                model_path=str(model_path)
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Training failed for {algorithm_name.value}: {str(e)}"
            self.log(error_msg, "error")
            
            return TrainingResult(
                algorithm_name=algorithm_name.value,
                model=None,
                training_time=training_time,
                hyperparameters=hyperparameters,
                feature_importance=None,
                model_path=None,
                error=error_msg
            )
    
    def evaluate_model(self, training_result: TrainingResult,
                      X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
        """Evaluate a trained model"""
        if training_result.error or training_result.model is None:
            return EvaluationResult(
                algorithm_name=training_result.algorithm_name,
                metrics={},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=training_result.feature_importance,
                error=training_result.error or "Model training failed"
            )
        
        self.log(f"Evaluating {training_result.algorithm_name}")
        
        try:
            model = training_result.model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass  # Some models might not support predict_proba in all cases
            
            # Calculate metrics based on problem type
            metrics = {}
            conf_matrix = None
            class_report = None
            
            if self.problem_type == ProblemTypeEnum.CLASSIFICATION:
                # Classification metrics
                metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                
                # ROC-AUC for binary classification or with probabilities
                if y_pred_proba is not None:
                    try:
                        if len(np.unique(y_test)) == 2:  # Binary classification
                            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                        else:  # Multi-class
                            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba, 
                                                                   multi_class='ovr', average='weighted'))
                    except Exception as e:
                        self.log(f"Could not calculate ROC-AUC: {e}", "warning")
                
                # Confusion matrix and classification report
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
            else:  # Regression
                metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
                metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['r2'] = float(r2_score(y_test, y_pred))
                
                # Additional regression metrics
                metrics['mean_absolute_percentage_error'] = float(
                    np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
                )
            
            self.log(f"Successfully evaluated {training_result.algorithm_name}")
            
            return EvaluationResult(
                algorithm_name=training_result.algorithm_name,
                metrics=metrics,
                predictions=y_pred,
                prediction_probabilities=y_pred_proba,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                feature_importance=training_result.feature_importance
            )
            
        except Exception as e:
            error_msg = f"Evaluation failed for {training_result.algorithm_name}: {str(e)}"
            self.log(error_msg, "error")
            
            return EvaluationResult(
                algorithm_name=training_result.algorithm_name,
                metrics={},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=training_result.feature_importance,
                error=error_msg
            )
    
    def find_best_model(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Find the best performing model based on primary metric"""
        valid_results = [result for result in evaluation_results if not result.error and result.metrics]
        
        if not valid_results:
            return {"algorithm_name": None, "metrics": {}, "error": "No valid models found"}
        
        # Define primary metric for comparison
        if self.problem_type == ProblemTypeEnum.CLASSIFICATION:
            primary_metric = 'f1_score'
            higher_is_better = True
        else:
            primary_metric = 'r2'
            higher_is_better = True
        
        # Find best model
        best_result = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for result in valid_results:
            if primary_metric in result.metrics:
                value = result.metrics[primary_metric]
                if (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                    best_value = value
                    best_result = result
        
        if best_result:
            self.log(f"Best model: {best_result.algorithm_name} with {primary_metric}={best_value:.4f}")
            return {
                "algorithm_name": best_result.algorithm_name,
                "metrics": best_result.metrics,
                "primary_metric": primary_metric,
                "primary_metric_value": best_value
            }
        else:
            return {"algorithm_name": None, "metrics": {}, "error": f"No models with {primary_metric} metric"}
    
    def aggregate_results(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate results across all models"""
        valid_results = [result for result in evaluation_results if not result.error and result.metrics]
        
        if not valid_results:
            return {"error": "No valid results to aggregate"}
        
        aggregated = {
            "models_trained": len(evaluation_results),
            "models_successful": len(valid_results),
            "models_failed": len(evaluation_results) - len(valid_results),
            "metrics_summary": {}
        }
        
        # Aggregate metrics
        all_metrics = {}
        for result in valid_results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate summary statistics for each metric
        for metric, values in all_metrics.items():
            aggregated["metrics_summary"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }
        
        return aggregated


# Prefect Tasks for ML Training Pipeline

@task(name="validate_ml_config")
def validate_ml_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ML training configuration"""
    logger = get_run_logger()
    logger.info("Validating ML training configuration")
    
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "validated_config": config.copy()
    }
    
    try:
        # Required fields
        required_fields = ["problem_type", "target_column", "algorithms", "file_path"]
        for field in required_fields:
            if field not in config:
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["is_valid"] = False
        
        # Validate problem type
        if "problem_type" in config:
            try:
                ProblemTypeEnum(config["problem_type"])
            except ValueError:
                validation_result["errors"].append(f"Invalid problem_type: {config['problem_type']}")
                validation_result["is_valid"] = False
        
        # Validate algorithms
        if "algorithms" in config and isinstance(config["algorithms"], list):
            registry = get_algorithm_registry()
            valid_algorithms = []
            
            for algo_config in config["algorithms"]:
                if isinstance(algo_config, dict) and "name" in algo_config:
                    try:
                        algo_enum = AlgorithmNameEnum(algo_config["name"])
                        algo_def = registry.get_algorithm(algo_enum)
                        if algo_def:
                            valid_algorithms.append(algo_config)
                        else:
                            validation_result["warnings"].append(f"Algorithm {algo_config['name']} not found in registry")
                    except ValueError:
                        validation_result["warnings"].append(f"Invalid algorithm name: {algo_config['name']}")
                else:
                    validation_result["warnings"].append(f"Invalid algorithm configuration: {algo_config}")
            
            validation_result["validated_config"]["algorithms"] = valid_algorithms
            
            if not valid_algorithms:
                validation_result["errors"].append("No valid algorithms specified")
                validation_result["is_valid"] = False
        
        # Set defaults
        if "pipeline_run_id" not in config:
            validation_result["validated_config"]["pipeline_run_id"] = f"ml_run_{int(time.time())}"
        
        if validation_result["is_valid"]:
            logger.info("ML configuration validation passed")
        else:
            logger.error(f"ML configuration validation failed: {validation_result['errors']}")
        
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(warning)
    
    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Validation error: {str(e)}")
        logger.error(f"ML configuration validation failed: {str(e)}")
    
    return validation_result


@task(name="train_multiple_algorithms")
def train_multiple_algorithms(
    preprocessing_result: PreprocessingResult,
    algorithms_config: List[Dict[str, Any]],
    problem_type: str,
    pipeline_run_id: str
) -> List[TrainingResult]:
    """Train multiple ML algorithms"""
    logger = get_run_logger()
    logger.info(f"Training {len(algorithms_config)} algorithms for {problem_type}")
    
    problem_type_enum = ProblemTypeEnum(problem_type)
    trainer = MLModelTrainer(problem_type_enum)
    trainer.set_logger(logger)
    
    training_results = []
    
    for i, algo_config in enumerate(algorithms_config):
        try:
            algorithm_name = AlgorithmNameEnum(algo_config["name"])
            custom_params = algo_config.get("hyperparameters", {})
            
            # Get hyperparameters
            hyperparams = trainer.get_algorithm_hyperparameters(algorithm_name, custom_params, pipeline_run_id)
            
            # Train algorithm
            result = trainer.train_single_algorithm(
                algorithm_name,
                preprocessing_result.X_train,
                preprocessing_result.y_train,
                hyperparams,
                pipeline_run_id
            )
            
            training_results.append(result)
            
            if result.error:
                logger.error(f"Training failed for {algorithm_name.value}: {result.error}")
            else:
                logger.info(f"Successfully trained {algorithm_name.value} ({i+1}/{len(algorithms_config)})")
        
        except Exception as e:
            error_msg = f"Failed to train algorithm {algo_config.get('name', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            
            training_results.append(TrainingResult(
                algorithm_name=algo_config.get('name', 'unknown'),
                model=None,
                training_time=0.0,
                hyperparameters={},
                feature_importance=None,
                model_path=None,
                error=error_msg
            ))
    
    successful_trainings = len([r for r in training_results if not r.error])
    logger.info(f"Training completed: {successful_trainings}/{len(training_results)} algorithms successful")
    
    return training_results


@task(name="evaluate_trained_models")
def evaluate_trained_models(
    training_results: List[TrainingResult],
    preprocessing_result: PreprocessingResult,
    problem_type: str
) -> List[EvaluationResult]:
    """Evaluate all trained models"""
    logger = get_run_logger()
    logger.info(f"Evaluating {len(training_results)} trained models")
    
    problem_type_enum = ProblemTypeEnum(problem_type)
    trainer = MLModelTrainer(problem_type_enum)
    trainer.set_logger(logger)
    
    evaluation_results = []
    
    for i, training_result in enumerate(training_results):
        try:
            evaluation_result = trainer.evaluate_model(
                training_result,
                preprocessing_result.X_test,
                preprocessing_result.y_test
            )
            
            evaluation_results.append(evaluation_result)
            
            if evaluation_result.error:
                logger.error(f"Evaluation failed for {training_result.algorithm_name}: {evaluation_result.error}")
            else:
                logger.info(f"Successfully evaluated {training_result.algorithm_name} ({i+1}/{len(training_results)})")
        
        except Exception as e:
            error_msg = f"Failed to evaluate {training_result.algorithm_name}: {str(e)}"
            logger.error(error_msg)
            
            evaluation_results.append(EvaluationResult(
                algorithm_name=training_result.algorithm_name,
                metrics={},
                predictions=np.array([]),
                prediction_probabilities=None,
                confusion_matrix=None,
                classification_report=None,
                feature_importance=training_result.feature_importance,
                error=error_msg
            ))
    
    successful_evaluations = len([r for r in evaluation_results if not r.error])
    logger.info(f"Evaluation completed: {successful_evaluations}/{len(evaluation_results)} models successful")
    
    return evaluation_results


@task(name="aggregate_training_results")
def aggregate_training_results(
    training_results: List[TrainingResult],
    evaluation_results: List[EvaluationResult],
    problem_type: str,
    pipeline_run_id: str,
    preprocessing_result: PreprocessingResult,
    config: Dict[str, Any]
) -> MLTrainingResult:
    """Aggregate all training and evaluation results"""
    logger = get_run_logger()
    logger.info("Aggregating ML training results")
    
    problem_type_enum = ProblemTypeEnum(problem_type)
    trainer = MLModelTrainer(problem_type_enum)
    trainer.set_logger(logger)
    
    # Calculate total training time
    total_training_time = sum(result.training_time for result in training_results)
    
    # Find best model
    best_model = trainer.find_best_model(evaluation_results)
    
    # Aggregate metrics
    aggregated_metrics = trainer.aggregate_results(evaluation_results)
    
    # Create summary
    summary = {
        "pipeline_run_id": pipeline_run_id,
        "problem_type": problem_type,
        "target_variable": config.get("target_column", "unknown"),
        "algorithms_attempted": len(training_results),
        "algorithms_successful": len([r for r in training_results if not r.error]),
        "total_training_time": total_training_time,
        "best_algorithm": best_model.get("algorithm_name"),
        "best_metric_value": best_model.get("primary_metric_value"),
        "dataset_info": {
            "train_samples": len(preprocessing_result.X_train),
            "test_samples": len(preprocessing_result.X_test),
            "features": len(preprocessing_result.feature_names),
            "preprocessing_steps": preprocessing_result.preprocessing_steps
        },
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Training summary: {summary['algorithms_successful']}/{summary['algorithms_attempted']} successful")
    if summary['best_metric_value'] is not None:
        logger.info(f"Best model: {summary['best_algorithm']} with {best_model.get('primary_metric')}={summary['best_metric_value']:.4f}")
    else:
        logger.info("No successful models found")
    
    return MLTrainingResult(
        pipeline_run_id=pipeline_run_id,
        problem_type=problem_type,
        target_variable=config.get("target_column", "unknown"),
        preprocessing_result=preprocessing_result,
        training_results=training_results,
        evaluation_results=evaluation_results,
        best_model=best_model,
        aggregated_metrics=aggregated_metrics,
        total_training_time=total_training_time,
        summary=summary
    )


# Main ML Training Flow

@flow(name="ml_training_flow", task_runner=SequentialTaskRunner())
def ml_training_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete ML training flow
    
    Args:
        config: Dictionary containing ML training configuration:
            - file_path: Path to dataset
            - target_column: Target variable name
            - problem_type: 'classification' or 'regression'
            - algorithms: List of algorithm configurations
            - preprocessing_config: Optional preprocessing configuration
            - pipeline_run_id: Optional run identifier
    
    Returns:
        Dictionary containing complete training results
    """
    logger = get_run_logger()
    logger.info("Starting ML training flow")
    
    try:
        # Step 1: Validate configuration
        validation_result = validate_ml_config(config)
        
        if not validation_result["is_valid"]:
            return {
                "success": False,
                "error": f"Configuration validation failed: {validation_result['errors']}",
                "validation": validation_result
            }
        
        validated_config = validation_result["validated_config"]
        
        # Step 2: Run data preprocessing
        logger.info("Starting data preprocessing")
        
        algorithm_names = [algo["name"] for algo in validated_config["algorithms"]]
        preprocessing_config = validated_config.get("preprocessing_config", {})
        
        preprocessing_flow_result = data_preprocessing_flow(
            file_path=validated_config["file_path"],
            target_column=validated_config["target_column"],
            problem_type=validated_config["problem_type"],
            algorithm_names=algorithm_names,
            custom_config=preprocessing_config,
            pipeline_run_id=validated_config["pipeline_run_id"]
        )
        
        if not preprocessing_flow_result["success"]:
            return {
                "success": False,
                "error": f"Preprocessing failed: {preprocessing_flow_result.get('error', 'Unknown error')}",
                "preprocessing_result": preprocessing_flow_result
            }
        
        preprocessing_result = preprocessing_flow_result["preprocessing_result"]
        
        # Step 3: Train multiple algorithms
        training_results = train_multiple_algorithms(
            preprocessing_result,
            validated_config["algorithms"],
            validated_config["problem_type"],
            validated_config["pipeline_run_id"]
        )
        
        # Step 4: Evaluate trained models
        evaluation_results = evaluate_trained_models(
            training_results,
            preprocessing_result,
            validated_config["problem_type"]
        )
        
        # Step 5: Aggregate results
        final_result = aggregate_training_results(
            training_results,
            evaluation_results,
            validated_config["problem_type"],
            validated_config["pipeline_run_id"],
            preprocessing_result,
            validated_config
        )
        
        logger.info("ML training flow completed successfully")
        
        # Convert preprocessing_result to dict for JSON serialization
        preprocessing_result_dict = preprocessing_flow_result.copy()
        if "preprocessing_result" in preprocessing_result_dict and hasattr(preprocessing_result_dict["preprocessing_result"], "to_dict"):
            preprocessing_result_dict["preprocessing_result"] = preprocessing_result_dict["preprocessing_result"].to_dict()
        
        # Convert config_used to ensure JSON serialization
        config_serializable = validated_config.copy()
        if "preprocessing_config" in config_serializable and hasattr(config_serializable["preprocessing_config"], "to_dict"):
            config_serializable["preprocessing_config"] = config_serializable["preprocessing_config"].to_dict()
        
        return {
            "success": True,
            "result": final_result.to_dict(),  # Convert to dict for JSON serialization
            "preprocessing_result": preprocessing_result_dict,  # Convert preprocessing result to dict
            "validation": validation_result,
            "config_used": config_serializable  # Convert config to dict
        }
        
    except Exception as e:
        error_msg = f"ML training flow failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


# Utility functions for external integration

def create_ml_training_config(
    file_path: str,
    target_column: str,
    problem_type: str,
    algorithms: List[Dict[str, Any]],
    preprocessing_config: Optional[Dict[str, Any]] = None,
    pipeline_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create ML training configuration dictionary"""
    return {
        "file_path": file_path,
        "target_column": target_column,
        "problem_type": problem_type,
        "algorithms": algorithms,
        "preprocessing_config": preprocessing_config or {},
        "pipeline_run_id": pipeline_run_id or f"ml_run_{int(time.time())}"
    }


def get_algorithm_suggestions(problem_type: str) -> List[Dict[str, Any]]:
    """Get suggested algorithms for a problem type"""
    registry = get_algorithm_registry()
    problem_enum = ProblemTypeEnum(problem_type)
    
    suggestions = []
    for algorithm_name in AlgorithmNameEnum:
        algorithm_def = registry.get_algorithm(algorithm_name)
        if algorithm_def and problem_enum in algorithm_def.problem_types:
            suggestions.append({
                "name": algorithm_name.value,
                "display_name": algorithm_def.display_name,
                "description": algorithm_def.description,
                "default_hyperparameters": {
                    param.name: param.default for param in algorithm_def.hyperparameters
                },
                "complexity": algorithm_def.training_complexity
            })
    
    return suggestions


def validate_algorithm_config(algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate individual algorithm configuration"""
    try:
        algorithm_name = AlgorithmNameEnum(algorithm_config["name"])
        registry = get_algorithm_registry()
        algorithm_def = registry.get_algorithm(algorithm_name)
        
        if not algorithm_def:
            return {"valid": False, "error": f"Algorithm {algorithm_name.value} not found"}
        
        # Validate hyperparameters if provided
        if "hyperparameters" in algorithm_config:
            # Basic validation - check if parameters are in expected format
            hyperparams = algorithm_config["hyperparameters"]
            if not isinstance(hyperparams, dict):
                return {"valid": False, "error": "Hyperparameters must be a dictionary"}
        
        return {"valid": True, "algorithm_def": algorithm_def}
        
    except Exception as e:
        return {"valid": False, "error": str(e)} 