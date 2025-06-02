"""
Algorithm Registry for DS1.2.2
Centralizes algorithm definitions, default hyperparameters, and validation schemas
"""

from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
import sys
import os

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from models.ml_models import (
    AlgorithmNameEnum, ProblemTypeEnum, MetricNameEnum, 
    AlgorithmConfig, PreprocessingStepEnum
)


@dataclass
class HyperparameterSpec:
    """Specification for a single hyperparameter"""
    name: str
    type: Type  # int, float, str, bool, list
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    is_required: bool = True


@dataclass
class AlgorithmDefinition:
    """Complete definition of an ML algorithm"""
    name: AlgorithmNameEnum
    display_name: str
    description: str
    problem_types: List[ProblemTypeEnum]
    hyperparameters: List[HyperparameterSpec]
    default_metrics: List[MetricNameEnum]
    recommended_preprocessing: List[PreprocessingStepEnum]
    sklearn_class: str  # sklearn class path
    min_samples: int = 10  # Minimum samples required
    supports_feature_importance: bool = True
    supports_probabilities: bool = True  # For classification
    training_complexity: str = "medium"  # low, medium, high


class AlgorithmRegistry:
    """
    Central registry for all supported ML algorithms with their configurations
    """
    
    def __init__(self):
        self._algorithms: Dict[AlgorithmNameEnum, AlgorithmDefinition] = {}
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize all algorithm definitions"""
        # Classification Algorithms
        self._register_logistic_regression()
        self._register_decision_tree_classifier()
        self._register_random_forest_classifier()
        self._register_svm_classifier()
        self._register_knn_classifier()
        
        # Regression Algorithms
        self._register_linear_regression()
        self._register_decision_tree_regressor()
        self._register_random_forest_regressor()
        self._register_svm_regressor()
        self._register_knn_regressor()
    
    # Classification Algorithm Definitions
    
    def _register_logistic_regression(self):
        """Register Logistic Regression algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization strength (smaller values = stronger regularization)"
            ),
            HyperparameterSpec(
                name="max_iter", type=int, default=1000,
                min_value=100, max_value=10000,
                description="Maximum number of iterations"
            ),
            HyperparameterSpec(
                name="penalty", type=str, default="l2",
                allowed_values=["l1", "l2", "elasticnet", "none"],
                description="Regularization penalty type"
            ),
            HyperparameterSpec(
                name="solver", type=str, default="lbfgs",
                allowed_values=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                description="Optimization algorithm"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.LOGISTIC_REGRESSION] = AlgorithmDefinition(
            name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            display_name="Logistic Regression",
            description="Linear classifier using logistic function. Fast and interpretable.",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION, 
                MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.linear_model.LogisticRegression",
            min_samples=50,
            supports_feature_importance=True,
            supports_probabilities=True,
            training_complexity="low"
        )
    
    def _register_decision_tree_classifier(self):
        """Register Decision Tree Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="max_depth", type=int, default=10,
                min_value=1, max_value=50,
                description="Maximum depth of the tree"
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=100,
                description="Minimum samples required to split an internal node"
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=50,
                description="Minimum samples required to be at a leaf node"
            ),
            HyperparameterSpec(
                name="criterion", type=str, default="gini",
                allowed_values=["gini", "entropy", "log_loss"],
                description="Function to measure quality of a split"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.DECISION_TREE_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.DECISION_TREE_CLASSIFIER,
            display_name="Decision Tree",
            description="Tree-based classifier. Highly interpretable with feature importance.",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION, 
                MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL
            ],
            sklearn_class="sklearn.tree.DecisionTreeClassifier",
            min_samples=20,
            supports_feature_importance=True,
            supports_probabilities=True,
            training_complexity="low"
        )
    
    def _register_random_forest_classifier(self):
        """Register Random Forest Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of trees in the forest"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=10,
                min_value=1, max_value=50,
                description="Maximum depth of the trees"
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=20,
                description="Minimum samples required to split an internal node"
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=20,
                description="Minimum samples required to be at a leaf node"
            ),
            HyperparameterSpec(
                name="max_features", type=str, default="sqrt",
                allowed_values=["sqrt", "log2", "none"],
                description="Number of features to consider when looking for the best split"
            ),
            HyperparameterSpec(
                name="bootstrap", type=bool, default=True,
                description="Whether bootstrap samples are used when building trees"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            display_name="Random Forest",
            description="Ensemble of decision trees. Robust and handles overfitting well.",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION, 
                MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL
            ],
            sklearn_class="sklearn.ensemble.RandomForestClassifier",
            min_samples=50,
            supports_feature_importance=True,
            supports_probabilities=True,
            training_complexity="medium"
        )
    
    def _register_svm_classifier(self):
        """Register SVM Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization parameter"
            ),
            HyperparameterSpec(
                name="kernel", type=str, default="rbf",
                allowed_values=["linear", "poly", "rbf", "sigmoid"],
                description="Kernel type for the algorithm"
            ),
            HyperparameterSpec(
                name="gamma", type=str, default="scale",
                allowed_values=["scale", "auto"],
                description="Kernel coefficient for rbf, poly and sigmoid"
            ),
            HyperparameterSpec(
                name="degree", type=int, default=3,
                min_value=1, max_value=10,
                description="Degree of the polynomial kernel function"
            ),
            HyperparameterSpec(
                name="probability", type=bool, default=True,
                description="Whether to enable probability estimates"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.SVM_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.SVM_CLASSIFIER,
            display_name="Support Vector Machine",
            description="Powerful classifier using support vectors. Good for high-dimensional data.",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION, 
                MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.svm.SVC",
            min_samples=100,
            supports_feature_importance=False,
            supports_probabilities=True,
            training_complexity="high"
        )
    
    def _register_knn_classifier(self):
        """Register KNN Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_neighbors", type=int, default=5,
                min_value=1, max_value=50,
                description="Number of neighbors to use"
            ),
            HyperparameterSpec(
                name="weights", type=str, default="uniform",
                allowed_values=["uniform", "distance"],
                description="Weight function used in prediction"
            ),
            HyperparameterSpec(
                name="algorithm", type=str, default="sqrt",
                allowed_values=["auto", "ball_tree", "kd_tree", "brute"],
                description="Algorithm used to compute the nearest neighbors"
            ),
            HyperparameterSpec(
                name="metric", type=str, default="minkowski",
                allowed_values=["euclidean", "manhattan", "minkowski", "chebyshev"],
                description="Distance metric to use"
            ),
            HyperparameterSpec(
                name="p", type=int, default=2,
                min_value=1, max_value=5,
                description="Power parameter for the Minkowski metric"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.KNN_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.KNN_CLASSIFIER,
            display_name="K-Nearest Neighbors",
            description="Instance-based classifier. Simple and effective for small datasets.",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.ACCURACY, MetricNameEnum.PRECISION, 
                MetricNameEnum.RECALL, MetricNameEnum.F1_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.neighbors.KNeighborsClassifier",
            min_samples=30,
            supports_feature_importance=False,
            supports_probabilities=True,
            training_complexity="low"
        )
    
    # Regression Algorithm Definitions
    
    def _register_linear_regression(self):
        """Register Linear Regression algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="fit_intercept", type=bool, default=True,
                description="Whether to calculate the intercept"
            ),
            
            HyperparameterSpec(
                name="positive", type=bool, default=False,
                description="Whether to force coefficients to be positive"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.LINEAR_REGRESSION] = AlgorithmDefinition(
            name=AlgorithmNameEnum.LINEAR_REGRESSION,
            display_name="Linear Regression",
            description="Simple linear regression. Fast and interpretable.",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.MAE, MetricNameEnum.MSE, 
                MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.linear_model.LinearRegression",
            min_samples=30,
            supports_feature_importance=True,
            supports_probabilities=False,
            training_complexity="low"
        )
    
    def _register_decision_tree_regressor(self):
        """Register Decision Tree Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="max_depth", type=int, default=10,
                min_value=1, max_value=50,
                description="Maximum depth of the tree"
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=100,
                description="Minimum samples required to split an internal node"
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=50,
                description="Minimum samples required to be at a leaf node"
            ),
            HyperparameterSpec(
                name="criterion", type=str, default="squared_error",
                allowed_values=["squared_error", "friedman_mse", "absolute_error", "poisson"],
                description="Function to measure quality of a split"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.DECISION_TREE_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.DECISION_TREE_REGRESSOR,
            display_name="Decision Tree Regressor",
            description="Tree-based regressor. Highly interpretable with feature importance.",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.MAE, MetricNameEnum.MSE, 
                MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL
            ],
            sklearn_class="sklearn.tree.DecisionTreeRegressor",
            min_samples=20,
            supports_feature_importance=True,
            supports_probabilities=False,
            training_complexity="low"
        )
    
    def _register_random_forest_regressor(self):
        """Register Random Forest Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of trees in the forest"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=10,
                min_value=1, max_value=50,
                description="Maximum depth of the trees"
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=20,
                description="Minimum samples required to split an internal node"
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=20,
                description="Minimum samples required to be at a leaf node"
            ),
            HyperparameterSpec(
                name="max_features", type=str, default="sqrt",
                allowed_values=["sqrt", "log2", "none"],
                description="Number of features to consider when looking for the best split"
            ),
            HyperparameterSpec(
                name="bootstrap", type=bool, default=True,
                description="Whether bootstrap samples are used when building trees"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            display_name="Random Forest Regressor",
            description="Ensemble of decision trees for regression. Robust and handles overfitting well.",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.MAE, MetricNameEnum.MSE, 
                MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL
            ],
            sklearn_class="sklearn.ensemble.RandomForestRegressor",
            min_samples=50,
            supports_feature_importance=True,
            supports_probabilities=False,
            training_complexity="medium"
        )
    
    def _register_svm_regressor(self):
        """Register SVM Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization parameter"
            ),
            HyperparameterSpec(
                name="kernel", type=str, default="rbf",
                allowed_values=["linear", "poly", "rbf", "sigmoid"],
                description="Kernel type for the algorithm"
            ),
            HyperparameterSpec(
                name="gamma", type=str, default="scale",
                allowed_values=["scale", "auto"],
                description="Kernel coefficient for rbf, poly and sigmoid"
            ),
            HyperparameterSpec(
                name="degree", type=int, default=3,
                min_value=1, max_value=10,
                description="Degree of the polynomial kernel function"
            ),
            HyperparameterSpec(
                name="epsilon", type=float, default=0.1,
                min_value=0.001, max_value=1.0,
                description="Epsilon in the epsilon-SVR model"
            ),
            HyperparameterSpec(
                name="random_state", type=int, default=42,
                description="Random state for reproducibility", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.SVM_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.SVM_REGRESSOR,
            display_name="Support Vector Regressor",
            description="Powerful regressor using support vectors. Good for high-dimensional data.",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.MAE, MetricNameEnum.MSE, 
                MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.svm.SVR",
            min_samples=100,
            supports_feature_importance=False,
            supports_probabilities=False,
            training_complexity="high"
        )
    
    def _register_knn_regressor(self):
        """Register KNN Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_neighbors", type=int, default=5,
                min_value=1, max_value=50,
                description="Number of neighbors to use"
            ),
            HyperparameterSpec(
                name="weights", type=str, default="uniform",
                allowed_values=["uniform", "distance"],
                description="Weight function used in prediction"
            ),
            HyperparameterSpec(
                name="algorithm", type=str, default="sqrt",
                allowed_values=["auto", "ball_tree", "kd_tree", "brute"],
                description="Algorithm used to compute the nearest neighbors"
            ),
            HyperparameterSpec(
                name="metric", type=str, default="minkowski",
                allowed_values=["euclidean", "manhattan", "minkowski", "chebyshev"],
                description="Distance metric to use"
            ),
            HyperparameterSpec(
                name="p", type=int, default=2,
                min_value=1, max_value=5,
                description="Power parameter for the Minkowski metric"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.KNN_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.KNN_REGRESSOR,
            display_name="K-Nearest Neighbors Regressor",
            description="Instance-based regressor. Simple and effective for small datasets.",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[
                MetricNameEnum.MAE, MetricNameEnum.MSE, 
                MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE
            ],
            recommended_preprocessing=[
                PreprocessingStepEnum.HANDLE_MISSING,
                PreprocessingStepEnum.ENCODE_CATEGORICAL,
                PreprocessingStepEnum.SCALE_FEATURES
            ],
            sklearn_class="sklearn.neighbors.KNeighborsRegressor",
            min_samples=30,
            supports_feature_importance=False,
            supports_probabilities=False,
            training_complexity="low"
        )
    
    # Registry Access Methods
    
    def get_algorithm(self, name: AlgorithmNameEnum) -> Optional[AlgorithmDefinition]:
        """Get algorithm definition by name"""
        return self._algorithms.get(name)
    
    def get_all_algorithms(self) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get all algorithm definitions"""
        return self._algorithms.copy()
    
    def get_algorithms_by_problem_type(self, problem_type: ProblemTypeEnum) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get algorithms suitable for a specific problem type"""
        return {
            name: algo for name, algo in self._algorithms.items()
            if problem_type in algo.problem_types
        }
    
    def get_default_algorithms(self, problem_type: ProblemTypeEnum, max_count: int = 3) -> List[AlgorithmNameEnum]:
        """Get recommended default algorithms for a problem type"""
        algorithms = self.get_algorithms_by_problem_type(problem_type)
        
        # Sort by training complexity and pick most reliable ones
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            recommended = [
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                AlgorithmNameEnum.LOGISTIC_REGRESSION,
                AlgorithmNameEnum.DECISION_TREE_CLASSIFIER
            ]
        else:  # REGRESSION
            recommended = [
                AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
                AlgorithmNameEnum.LINEAR_REGRESSION,
                AlgorithmNameEnum.DECISION_TREE_REGRESSOR
            ]
        
        # Filter to only available algorithms and limit count
        available = [algo for algo in recommended if algo in algorithms]
        return available[:max_count]
    
    def create_algorithm_config(self, 
                              algorithm_name: AlgorithmNameEnum, 
                              hyperparameters: Optional[Dict[str, Any]] = None,
                              pipeline_run_id: Optional[str] = None) -> AlgorithmConfig:
        """Create an AlgorithmConfig with default or provided hyperparameters"""
        algo_def = self.get_algorithm(algorithm_name)
        if not algo_def:
            raise ValueError(f"Algorithm {algorithm_name} not found in registry")
        
        # Start with default hyperparameters
        default_hyperparams = {
            param.name: param.default 
            for param in algo_def.hyperparameters
        }
        
        # Generate unique random state based on pipeline run ID to ensure different results per run
        if pipeline_run_id and 'random_state' in default_hyperparams:
            import hashlib
            # Create deterministic but unique random seed from pipeline run ID
            hash_object = hashlib.md5(f"{pipeline_run_id}_{algorithm_name.value}".encode())
            unique_seed = int(hash_object.hexdigest()[:8], 16) % (2**31 - 1)  # Ensure it's a valid int32
            default_hyperparams['random_state'] = unique_seed
        
        # Override with provided hyperparameters
        if hyperparameters:
            # Validate provided hyperparameters
            validated_params = self.validate_hyperparameters(algorithm_name, hyperparameters)
            default_hyperparams.update(validated_params)
        
        return AlgorithmConfig(
            algorithm_name=algorithm_name,
            hyperparameters=default_hyperparams,
            is_enabled=True
        )
    
    def validate_hyperparameters(self, 
                                algorithm_name: AlgorithmNameEnum, 
                                hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters against algorithm specification"""
        algo_def = self.get_algorithm(algorithm_name)
        if not algo_def:
            raise ValueError(f"Algorithm {algorithm_name} not found in registry")
        
        # Create lookup for hyperparameter specs
        param_specs = {param.name: param for param in algo_def.hyperparameters}
        validated_params = {}
        errors = []
        
        for param_name, param_value in hyperparameters.items():
            if param_name not in param_specs:
                errors.append(f"Unknown hyperparameter: {param_name}")
                continue
            
            spec = param_specs[param_name]
            
            # Type validation
            if not isinstance(param_value, spec.type):
                try:
                    # Try to convert to correct type
                    param_value = spec.type(param_value)
                except (ValueError, TypeError):
                    errors.append(f"{param_name}: Expected {spec.type.__name__}, got {type(param_value).__name__}")
                    continue
            
            # Range validation
            if spec.min_value is not None and param_value < spec.min_value:
                errors.append(f"{param_name}: Value {param_value} below minimum {spec.min_value}")
                continue
            
            if spec.max_value is not None and param_value > spec.max_value:
                errors.append(f"{param_name}: Value {param_value} above maximum {spec.max_value}")
                continue
            
            # Allowed values validation
            if spec.allowed_values and param_value not in spec.allowed_values:
                errors.append(f"{param_name}: Value {param_value} not in allowed values {spec.allowed_values}")
                continue
            
            validated_params[param_name] = param_value
        
        if errors:
            raise ValueError(f"Hyperparameter validation errors: {'; '.join(errors)}")
        
        return validated_params
    
    def get_algorithm_info(self, algorithm_name: AlgorithmNameEnum) -> Dict[str, Any]:
        """Get comprehensive information about an algorithm"""
        algo_def = self.get_algorithm(algorithm_name)
        if not algo_def:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        return {
            "name": algo_def.name.value,
            "display_name": algo_def.display_name,
            "description": algo_def.description,
            "problem_types": [pt.value for pt in algo_def.problem_types],
            "hyperparameters": [
                {
                    "name": param.name,
                    "type": param.type.__name__,
                    "default": param.default,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "allowed_values": param.allowed_values,
                    "description": param.description,
                    "required": param.is_required
                }
                for param in algo_def.hyperparameters
            ],
            "default_metrics": [metric.value for metric in algo_def.default_metrics],
            "recommended_preprocessing": [step.value for step in algo_def.recommended_preprocessing],
            "min_samples": algo_def.min_samples,
            "supports_feature_importance": algo_def.supports_feature_importance,
            "supports_probabilities": algo_def.supports_probabilities,
            "training_complexity": algo_def.training_complexity
        }


# Global registry instance
_registry = None

def get_algorithm_registry() -> AlgorithmRegistry:
    """Get the global algorithm registry instance"""
    global _registry
    if _registry is None:
        _registry = AlgorithmRegistry()
    return _registry


# Convenience functions for external use

def get_supported_algorithms(problem_type: Optional[ProblemTypeEnum] = None) -> List[Dict[str, Any]]:
    """Get list of supported algorithms with their information"""
    registry = get_algorithm_registry()
    
    if problem_type:
        algorithms = registry.get_algorithms_by_problem_type(problem_type)
    else:
        algorithms = registry.get_all_algorithms()
    
    return [
        registry.get_algorithm_info(algo_name) 
        for algo_name in algorithms.keys()
    ]


def create_default_algorithm_configs(problem_type: ProblemTypeEnum, pipeline_run_id: Optional[str] = None) -> List[AlgorithmConfig]:
    """Create default algorithm configurations for a problem type"""
    registry = get_algorithm_registry()
    default_algos = registry.get_default_algorithms(problem_type)
    
    return [
        registry.create_algorithm_config(algo_name, pipeline_run_id=pipeline_run_id)
        for algo_name in default_algos
    ]


def validate_algorithm_config(algorithm_config: AlgorithmConfig) -> bool:
    """Validate an algorithm configuration"""
    registry = get_algorithm_registry()
    try:
        registry.validate_hyperparameters(
            algorithm_config.algorithm_name,
            algorithm_config.hyperparameters
        )
        return True
    except ValueError:
        return False 