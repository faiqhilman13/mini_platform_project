"""
ML Pipeline Models for DS1.2.1
Defines data structures for ML pipeline configuration, execution, and results
"""

from sqlmodel import SQLModel, Field, Column, Relationship
from sqlalchemy import JSON, Text
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
from enum import Enum

# Import existing models for relationships
from app.models.file_models import UploadedFileLog
from app.models.pipeline_models import PipelineRunBase


class ProblemTypeEnum(str, Enum):
    """Enum for machine learning problem types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AlgorithmNameEnum(str, Enum):
    """Enum for supported ML algorithms"""
    # Classification algorithms
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    SVM_CLASSIFIER = "svm_classifier"
    KNN_CLASSIFIER = "knn_classifier"
    
    # Regression algorithms
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    SVM_REGRESSOR = "svm_regressor"
    KNN_REGRESSOR = "knn_regressor"


class PreprocessingStepEnum(str, Enum):
    """Enum for preprocessing steps"""
    HANDLE_MISSING = "handle_missing"
    ENCODE_CATEGORICAL = "encode_categorical"
    SCALE_FEATURES = "scale_features"
    REMOVE_OUTLIERS = "remove_outliers"
    FEATURE_SELECTION = "feature_selection"


class MetricNameEnum(str, Enum):
    """Enum for evaluation metrics"""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    
    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"


class AlgorithmConfig(SQLModel):
    """Configuration for a single ML algorithm"""
    algorithm_name: AlgorithmNameEnum
    hyperparameters: Dict[str, Any] = Field(default={})
    is_enabled: bool = True
    
    # Algorithm-specific settings
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    random_state: Optional[int] = 42
    
    # Resource constraints
    max_training_time_minutes: Optional[int] = Field(default=30, ge=1, le=120)
    
    class Config:
        json_schema_extra = {
            "example": {
                "algorithm_name": "random_forest_classifier",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2
                },
                "is_enabled": True,
                "cross_validation_folds": 5,
                "random_state": 42
            }
        }


class PreprocessingConfig(SQLModel):
    """Configuration for data preprocessing"""
    steps: List[PreprocessingStepEnum] = Field(default=[])
    
    # Missing value handling
    missing_value_strategy: str = Field(default="mean")  # mean, median, mode, drop
    missing_value_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # Drop columns with >50% missing
    
    # Categorical encoding
    categorical_encoding: str = Field(default="onehot")  # onehot, label, target
    max_categories: int = Field(default=20, ge=2, le=100)
    
    # Feature scaling
    scaling_method: str = Field(default="standard")  # standard, minmax, robust, none
    
    # Outlier handling
    outlier_method: str = Field(default="zscore")  # zscore, iqr, isolation_forest
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    
    # Train/test split
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    stratify: bool = True  # For classification problems
    
    # Feature selection
    feature_selection_method: Optional[str] = None  # selectkbest, rfe, lasso
    max_features: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "steps": ["handle_missing", "encode_categorical", "scale_features"],
                "missing_value_strategy": "mean",
                "categorical_encoding": "onehot",
                "scaling_method": "standard",
                "test_size": 0.2,
                "stratify": True
            }
        }


class MLPipelineConfig(SQLModel):
    """Complete configuration for an ML pipeline"""
    # Problem definition
    problem_type: ProblemTypeEnum
    target_variable: str
    feature_variables: List[str] = Field(default=[])  # Empty means use all except target
    
    # Algorithms to train
    algorithms: List[AlgorithmConfig] = Field(default=[])
    
    # Preprocessing configuration
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    
    # Evaluation settings
    evaluation_metrics: List[MetricNameEnum] = Field(default=[])
    cross_validation: bool = True
    
    # Pipeline settings
    pipeline_name: Optional[str] = None
    description: Optional[str] = None
    
    # Resource constraints
    max_total_training_time_minutes: int = Field(default=120, ge=10, le=480)
    parallel_jobs: int = Field(default=1, ge=1, le=4)
    
    class Config:
        json_schema_extra = {
            "example": {
                "problem_type": "classification",
                "target_variable": "target_column",
                "feature_variables": ["feature1", "feature2", "feature3"],
                "algorithms": [],
                "preprocessing": {},
                "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
                "pipeline_name": "Customer Churn Prediction",
                "description": "Predict customer churn using various ML algorithms"
            }
        }


class ModelMetrics(SQLModel):
    """Performance metrics for a trained model"""
    metric_name: MetricNameEnum
    value: float
    
    # Additional metric details
    std_dev: Optional[float] = None  # From cross-validation
    confidence_interval: Optional[List[float]] = None  # [lower, upper]
    
    # Context information
    dataset_split: str = "test"  # train, validation, test
    fold_number: Optional[int] = None  # For cross-validation


class ModelArtifacts(SQLModel):
    """Artifacts generated from model training"""
    model_file_path: Optional[str] = None  # Serialized model location
    feature_importance_path: Optional[str] = None  # Feature importance plot/data
    confusion_matrix_path: Optional[str] = None  # Confusion matrix (classification)
    learning_curve_path: Optional[str] = None  # Learning curve plot
    residual_plot_path: Optional[str] = None  # Residual plot (regression)
    
    # Model metadata
    model_size_mb: Optional[float] = None
    training_data_shape: Optional[List[int]] = None  # [n_rows, n_features]
    feature_names: List[str] = Field(default=[])
    
    # Serialization info
    sklearn_version: Optional[str] = None
    python_version: Optional[str] = None


class MLResult(SQLModel):
    """Results from training a single ML model"""
    # Model identification
    model_id: uuid.UUID = Field(default_factory=uuid.uuid4, unique=True)
    algorithm_name: AlgorithmNameEnum
    hyperparameters: Dict[str, Any] = Field(default={})
    
    # Performance metrics
    metrics: List[ModelMetrics] = Field(default=[])
    primary_metric_value: float  # Main metric for ranking (e.g., accuracy, r2)
    primary_metric_name: str
    
    # Training information
    training_time_seconds: float
    cross_validation_scores: Optional[Dict[str, List[float]]] = None
    
    # Feature importance (top 10)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model artifacts
    artifacts: Optional[ModelArtifacts] = None
    
    # Status and errors
    training_status: str = "completed"  # training, completed, failed
    error_message: Optional[str] = None
    
    # Validation information
    overfitting_score: Optional[float] = None  # Train vs validation performance gap
    is_best_model: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "algorithm_name": "random_forest_classifier",
                "hyperparameters": {"n_estimators": 100, "max_depth": 10},
                "primary_metric_value": 0.85,
                "primary_metric_name": "accuracy",
                "training_time_seconds": 45.2,
                "training_status": "completed"
            }
        }


# Database Models (extend existing pipeline models)

class MLPipelineRun(SQLModel, table=True):
    """ML-specific pipeline run tracking"""
    __tablename__ = "ml_pipeline_run"
    __table_args__ = {'extend_existing': True}
    
    # Primary key and basic info
    id: Optional[int] = Field(default=None, primary_key=True)
    run_uuid: str = Field(unique=True, index=True, default_factory=lambda: str(uuid.uuid4()))
    
    # Link to uploaded file
    uploaded_file_log_id: int = Field(foreign_key="uploadedfilelog.id", index=True)
    
    # Basic pipeline fields
    pipeline_type: str = Field(default="ML_TRAINING")
    status: str = Field(default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED
    
    # ML-specific fields
    problem_type: ProblemTypeEnum
    target_variable: str
    feature_count: int = 0
    
    # Configuration (stored as JSON)
    ml_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    algorithms_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    preprocessing_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Results summary
    total_models_trained: int = 0
    best_model_id: Optional[str] = None
    best_model_score: Optional[float] = None
    best_model_metric: Optional[str] = None
    
    # Training metadata
    total_training_time_seconds: Optional[float] = None
    dataset_rows_used: Optional[int] = None
    dataset_features_used: Optional[int] = None
    
    # Quality indicators
    data_quality_score: Optional[float] = None
    preprocessing_warnings: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    
    # Error handling
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MLModel(SQLModel, table=True):
    """Individual ML model within a pipeline run"""
    __tablename__ = "ml_model"
    __table_args__ = {'extend_existing': True}
    
    # Primary identification
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True, index=True)  # UUID as string
    pipeline_run_id: Optional[int] = Field(foreign_key="ml_pipeline_run.id", index=True)
    
    # Model information
    algorithm_name: str
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Performance metrics (stored as JSON)
    performance_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    primary_metric_value: Optional[float] = None
    primary_metric_name: Optional[str] = None
    
    # Training information
    training_time_seconds: Optional[float] = None
    cross_validation_scores: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Feature analysis
    feature_importance: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    feature_count: Optional[int] = None
    
    # Model artifacts and files
    model_file_path: Optional[str] = None
    artifacts_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Status and validation
    training_status: str = "completed"
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    is_best_model: bool = False
    overfitting_score: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Define relationships after both classes are defined to avoid forward reference issues
MLPipelineRun.models = Relationship(back_populates="pipeline_run")
MLModel.pipeline_run = Relationship(back_populates="models")


class MLExperiment(SQLModel, table=True):
    """Group related ML pipeline runs for comparison"""
    __tablename__ = "ml_experiment"
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: str = Field(unique=True, index=True, default_factory=lambda: str(uuid.uuid4()))
    
    # Experiment metadata
    name: str
    description: Optional[str] = None
    
    # Dataset information
    dataset_file_id: int = Field(foreign_key="uploadedfilelog.id", index=True)
    problem_type: ProblemTypeEnum
    target_variable: str
    
    # Experiment status
    status: str = "active"  # active, completed, archived
    
    # Summary statistics (computed from runs)
    total_runs: int = 0
    best_score: Optional[float] = None
    best_algorithm: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Tags for organization
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))


# Response Models for API

class MLPipelineConfigResponse(SQLModel):
    """Response model for ML pipeline configuration"""
    success: bool
    message: str
    config: Optional[MLPipelineConfig] = None
    validation_errors: List[str] = []


class MLPipelineRunResponse(SQLModel):
    """Response model for ML pipeline run status"""
    success: bool
    message: str
    run_id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    estimated_completion_time: Optional[datetime] = None


class MLResultsResponse(SQLModel):
    """Response model for ML training results"""
    success: bool
    message: str
    
    # Summary information
    total_models: int = 0
    completed_models: int = 0
    failed_models: int = 0
    
    # Best model information
    best_model: Optional[MLResult] = None
    
    # All model results
    results: List[MLResult] = []
    
    # Experiment metadata
    experiment_summary: Optional[Dict[str, Any]] = None
    
    # Processing information
    total_training_time_seconds: float = 0.0
    data_quality_warnings: List[str] = []


class ModelComparisonResponse(SQLModel):
    """Response model for comparing multiple models"""
    success: bool
    message: str
    
    # Comparison data
    models: List[MLResult] = []
    comparison_metrics: List[str] = []
    ranking: List[Dict[str, Any]] = []  # Ranked model performance
    
    # Recommendations
    recommended_model_id: Optional[str] = None
    recommendation_reasoning: Optional[str] = None
    
    # Statistical analysis
    statistical_significance: Optional[Dict[str, Any]] = None


# Request Models for API

class MLPipelineCreateRequest(SQLModel):
    """Request model to create and trigger ML pipeline (simplified for frontend)"""
    uploaded_file_log_id: int
    target_variable: str
    problem_type: ProblemTypeEnum
    algorithms: List[Dict[str, Any]] = Field(default=[])  # List of algorithm configs
    preprocessing_config: Dict[str, Any] = Field(default={})
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None


class MLPipelineCreateResponse(SQLModel):
    """Response model for ML pipeline creation"""
    success: bool
    message: str
    run_uuid: str
    status: str = "PENDING"
    estimated_completion_time: Optional[datetime] = None


class MLPipelineStatusResponse(SQLModel):
    """Response model for ML pipeline status"""
    success: bool
    message: str
    run_uuid: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    current_stage: Optional[str] = None
    progress_percentage: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None


class MLPipelineResultResponse(SQLModel):
    """Response model for ML pipeline results"""
    success: bool
    message: str
    run_uuid: str
    status: str
    
    # Model results
    model_metrics: List[ModelMetrics] = Field(default=[])
    best_model_id: Optional[str] = None
    best_model_score: Optional[float] = None
    best_model_metric: Optional[str] = None
    
    # Training summary
    total_models_trained: int = 0
    total_training_time_seconds: Optional[float] = None
    dataset_info: Optional[Dict[str, Any]] = None
    
    # Comparison and analysis
    model_comparison_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    
    # Quality and warnings
    data_quality_warnings: List[str] = Field(default=[])
    preprocessing_warnings: List[str] = Field(default=[])


class AlgorithmSuggestion(SQLModel):
    """Algorithm suggestion for ML training"""
    name: str
    display_name: str
    description: str
    complexity: str  # "low", "medium", "high"
    problem_types: List[str]  # ["classification", "regression"]
    default_hyperparameters: Dict[str, Any] = Field(default={})
    hyperparameter_info: Optional[Dict[str, Any]] = None


class MLPipelineStartRequest(SQLModel):
    """Request model to start ML pipeline"""
    file_id: int
    config: MLPipelineConfig
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None


class ModelPredictionRequest(SQLModel):
    """Request model for making predictions with trained model"""
    model_id: str
    input_data: Dict[str, Any]  # Feature values
    return_probabilities: bool = False  # For classification
    
    
class HyperparameterTuningRequest(SQLModel):
    """Request model for hyperparameter tuning"""
    algorithm_name: AlgorithmNameEnum
    parameter_space: Dict[str, Any]  # Parameter ranges/options
    optimization_metric: MetricNameEnum
    max_iterations: int = Field(default=50, ge=10, le=200)
    optimization_method: str = Field(default="random_search")  # random_search, grid_search, bayesian
    
    class Config:
        json_schema_extra = {
            "example": {
                "algorithm_name": "random_forest_classifier",
                "parameter_space": {
                    "n_estimators": {"min": 50, "max": 200},
                    "max_depth": {"min": 5, "max": 20}
                },
                "optimization_metric": "accuracy",
                "max_iterations": 100,
                "optimization_method": "random_search"
            }
        } 