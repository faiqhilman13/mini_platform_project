"""
Data Preprocessing Pipeline for DS1.2.3
Prefect tasks for comprehensive data preprocessing including missing values, encoding, scaling, and splitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import sys
import os
from pathlib import Path

# Scientific computing and ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, 
    mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression

# Prefect for workflow orchestration (optional)
try:
    from prefect import task, flow, get_run_logger
    from prefect.task_runners import SequentialTaskRunner
    PREFECT_AVAILABLE = True
except ImportError:
    # Fallback when Prefect is not available
    PREFECT_AVAILABLE = False
    
    # Create dummy decorators for testing
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
    
    # Mock SequentialTaskRunner
    class SequentialTaskRunner:
        def __init__(self):
            pass

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import models - use try/except for IDE compatibility
try:
    from models.ml_models import ProblemTypeEnum, PreprocessingStepEnum, AlgorithmNameEnum  # type: ignore
except ImportError:
    # Fallback for when running from different contexts
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
    from models.ml_models import ProblemTypeEnum, PreprocessingStepEnum, AlgorithmNameEnum  # type: ignore

try:
    from algorithm_registry import get_algorithm_registry
except ImportError:
    # Fallback for IDE/different execution contexts
    from .algorithm_registry import get_algorithm_registry


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations"""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessing_steps: List[str]
    transformations: Dict[str, Any]  # Store fitted transformers
    feature_names: List[str]
    preprocessing_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        try:
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types"""
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
                "train_shape": list(self.X_train.shape) if hasattr(self.X_train, 'shape') else [0, 0],
                "test_shape": list(self.X_test.shape) if hasattr(self.X_test, 'shape') else [0, 0],
                "train_samples": int(len(self.X_train)) if hasattr(self.X_train, '__len__') else 0,
                "test_samples": int(len(self.X_test)) if hasattr(self.X_test, '__len__') else 0,
                "target_train_samples": int(len(self.y_train)) if hasattr(self.y_train, '__len__') else 0,
                "target_test_samples": int(len(self.y_test)) if hasattr(self.y_test, '__len__') else 0,
                "preprocessing_steps": self.preprocessing_steps,
                "feature_names": self.feature_names,
                "preprocessing_summary": convert_numpy_types(self.preprocessing_summary),
                "transformation_keys": list(self.transformations.keys()) if self.transformations else []
            }
            
            return convert_numpy_types(result)
            
        except Exception as e:
            # Fallback to basic representation if conversion fails
            return {
                "preprocessing_steps": self.preprocessing_steps if hasattr(self, 'preprocessing_steps') else [],
                "feature_names": self.feature_names if hasattr(self, 'feature_names') else [],
                "conversion_error": str(e)
            }


@dataclass 
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    # Missing value handling
    missing_strategy: str = "mean"  # mean, median, mode, knn, drop
    missing_threshold: float = 0.8  # Drop columns with >80% missing values
    
    # Categorical encoding
    categorical_strategy: str = "onehot"  # onehot, label, ordinal, target
    max_categories: int = 20  # Max unique values for categorical encoding
    handle_unknown_categories: str = "ignore"  # ignore, error, infrequent
    
    # Feature scaling
    scaling_strategy: str = "standard"  # standard, minmax, robust, none
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True  # For classification
    
    # Feature selection
    feature_selection_method: Optional[str] = None  # selectkbest, rfe, lasso, mutual_info
    n_features_to_select: Optional[int] = None
    
    # Manual feature selection (NEW)
    selected_features: Optional[List[str]] = None  # Explicit list of features to use
    
    # Outlier handling
    outlier_method: str = "none"  # none, zscore, iqr, isolation_forest
    outlier_threshold: float = 3.0
    
    # Data validation
    min_samples_per_class: int = 5  # Minimum samples per class for classification
    max_memory_mb: int = 1000  # Maximum memory usage for processing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "missing_strategy": self.missing_strategy,
            "missing_threshold": self.missing_threshold,
            "categorical_strategy": self.categorical_strategy,
            "max_categories": self.max_categories,
            "handle_unknown_categories": self.handle_unknown_categories,
            "scaling_strategy": self.scaling_strategy,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "feature_selection_method": self.feature_selection_method,
            "n_features_to_select": self.n_features_to_select,
            "selected_features": self.selected_features,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
            "min_samples_per_class": self.min_samples_per_class,
            "max_memory_mb": self.max_memory_mb
        }


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for ML training
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.transformations = {}
        self.feature_names = []
        self.preprocessing_summary = {}
        self.logger = None
    
    def set_logger(self, logger):
        """Set Prefect logger"""
        self.logger = logger
    
    def log(self, message: str, level: str = "info"):
        """Log message with fallback to print"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def analyze_data_quality(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze data quality and generate preprocessing recommendations"""
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": {},
            "categorical_columns": [],
            "numerical_columns": [],
            "high_cardinality_columns": [],
            "constant_columns": [],
            "duplicate_rows": df.duplicated().sum(),
            "target_distribution": {}
        }
        
        # Analyze missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        analysis["missing_values"] = {
            col: {"count": int(count), "percentage": float(pct)}
            for col, count, pct in zip(df.columns, missing_counts, missing_percentages)
            if count > 0
        }
        
        # Classify column types
        for col in df.columns:
            if col == target_col:
                continue
                
            if df[col].dtype in ['object', 'category']:
                analysis["categorical_columns"].append(col)
                unique_count = df[col].nunique()
                if unique_count > self.config.max_categories:
                    analysis["high_cardinality_columns"].append(col)
            else:
                analysis["numerical_columns"].append(col)
            
            # Check for constant columns
            if df[col].nunique() <= 1:
                analysis["constant_columns"].append(col)
        
        # Analyze target variable
        if target_col in df.columns:
            target_series = df[target_col]
            if target_series.dtype in ['object', 'category']:
                analysis["target_distribution"] = target_series.value_counts().to_dict()
            else:
                analysis["target_distribution"] = {
                    "mean": float(target_series.mean()),
                    "std": float(target_series.std()),
                    "min": float(target_series.min()),
                    "max": float(target_series.max()),
                    "median": float(target_series.median())
                }
        
        return analysis
    
    def handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle missing values using specified strategy"""
        self.log(f"Handling missing values using strategy: {self.config.missing_strategy}")
        
        df = df.copy()
        
        # Drop columns with too many missing values
        missing_percentages = (df.isnull().sum() / len(df))
        columns_to_drop = missing_percentages[missing_percentages > self.config.missing_threshold].index
        columns_to_drop = [col for col in columns_to_drop if col != target_col]
        
        if len(columns_to_drop) > 0:
            self.log(f"Dropping columns with >{self.config.missing_threshold*100}% missing: {list(columns_to_drop)}")
            df = df.drop(columns=columns_to_drop)
        
        # Handle missing values in remaining columns
        if self.config.missing_strategy == "drop":
            initial_rows = len(df)
            df = df.dropna()
            self.log(f"Dropped rows with missing values: {initial_rows - len(df)} rows removed")
        
        elif self.config.missing_strategy in ["mean", "median", "most_frequent"]:
            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Exclude target column
            numerical_cols = [col for col in numerical_cols if col != target_col]
            categorical_cols = [col for col in categorical_cols if col != target_col]
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                strategy = self.config.missing_strategy if self.config.missing_strategy != "most_frequent" else "mean"
                imputer = SimpleImputer(strategy=strategy)
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                self.transformations["numerical_imputer"] = imputer
                self.log(f"Imputed {len(numerical_cols)} numerical columns with {strategy}")
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy="most_frequent")
                df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                self.transformations["categorical_imputer"] = imputer
                self.log(f"Imputed {len(categorical_cols)} categorical columns with most_frequent")
        
        elif self.config.missing_strategy == "knn":
            # KNN imputation
            feature_cols = [col for col in df.columns if col != target_col]
            imputer = KNNImputer(n_neighbors=5)
            df[feature_cols] = imputer.fit_transform(df[feature_cols])
            self.transformations["knn_imputer"] = imputer
            self.log(f"Applied KNN imputation to {len(feature_cols)} columns")
        
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str, 
                                   problem_type: ProblemTypeEnum) -> pd.DataFrame:
        """Encode categorical variables using specified strategy"""
        self.log(f"Encoding categorical variables using strategy: {self.config.categorical_strategy}")
        
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        if len(categorical_cols) == 0:
            self.log("No categorical columns to encode")
            return df
        
        encoded_dfs = []
        original_columns = df.columns.tolist()
        high_cardinality_cols = []  # Track columns to drop
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            # Skip high cardinality columns and mark them for removal
            if unique_count > self.config.max_categories:
                self.log(f"Skipping high cardinality column {col} (unique values: {unique_count})")
                high_cardinality_cols.append(col)
                continue
            
            if self.config.categorical_strategy == "onehot":
                # One-hot encoding
                encoder = OneHotEncoder(
                    drop='first', 
                    sparse_output=False,
                    handle_unknown=self.config.handle_unknown_categories
                )
                encoded_array = encoder.fit_transform(df[[col]])
                
                # Create column names
                if hasattr(encoder, 'get_feature_names_out'):
                    encoded_columns = encoder.get_feature_names_out([col])
                else:
                    encoded_columns = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]  # Skip first due to drop='first'
                
                encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
                encoded_dfs.append(encoded_df)
                
                self.transformations[f"onehot_{col}"] = encoder
                self.log(f"One-hot encoded {col} into {len(encoded_columns)} columns")
            
            elif self.config.categorical_strategy == "label":
                # Label encoding
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col])
                self.transformations[f"label_{col}"] = encoder
                self.log(f"Label encoded {col}")
            
            elif self.config.categorical_strategy == "ordinal":
                # Ordinal encoding
                encoder = OrdinalEncoder(handle_unknown=self.config.handle_unknown_categories)
                df[f"{col}_encoded"] = encoder.fit_transform(df[[col]])
                self.transformations[f"ordinal_{col}"] = encoder
                self.log(f"Ordinal encoded {col}")
        
        # Drop high cardinality columns
        if high_cardinality_cols:
            df = df.drop(columns=high_cardinality_cols)
            self.log(f"Dropped high cardinality columns: {high_cardinality_cols}")
        
        # Combine original dataframe with one-hot encoded columns
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)
            # Drop original categorical columns that were one-hot encoded
            onehot_encoded_cols = [col for col in categorical_cols 
                                 if f"onehot_{col}" in self.transformations]
            df = df.drop(columns=onehot_encoded_cols)
            
        return df
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features using specified strategy"""
        if self.config.scaling_strategy == "none":
            self.log("Skipping feature scaling")
            return X_train, X_test
        
        self.log(f"Scaling features using strategy: {self.config.scaling_strategy}")
        
        # Select numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.log("No numerical columns to scale")
            return X_train, X_test
        
        # Initialize scaler
        if self.config.scaling_strategy == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_strategy == "minmax":
            scaler = MinMaxScaler()
        elif self.config.scaling_strategy == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {self.config.scaling_strategy}")
        
        # Fit on training data and transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        self.transformations["scaler"] = scaler
        self.log(f"Scaled {len(numerical_cols)} numerical columns")
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, problem_type: ProblemTypeEnum) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform feature selection using specified method"""
        if not self.config.feature_selection_method:
            self.log("Skipping feature selection")
            return X_train, X_test
        
        self.log(f"Performing feature selection using: {self.config.feature_selection_method}")
        
        n_features = self.config.n_features_to_select or min(20, X_train.shape[1])
        
        if self.config.feature_selection_method == "selectkbest":
            # Select K best features
            if problem_type == ProblemTypeEnum.CLASSIFICATION:
                selector = SelectKBest(f_classif, k=n_features)
            else:
                selector = SelectKBest(f_regression, k=n_features)
        
        elif self.config.feature_selection_method == "mutual_info":
            # Mutual information
            if problem_type == ProblemTypeEnum.CLASSIFICATION:
                selector = SelectKBest(mutual_info_classif, k=n_features)
            else:
                selector = SelectKBest(mutual_info_regression, k=n_features)
        
        elif self.config.feature_selection_method == "rfe":
            # Recursive feature elimination
            if problem_type == ProblemTypeEnum.CLASSIFICATION:
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            selector = RFE(estimator, n_features_to_select=n_features)
        
        elif self.config.feature_selection_method == "lasso":
            # Lasso-based feature selection
            if problem_type == ProblemTypeEnum.CLASSIFICATION:
                estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=self.config.random_state)
            else:
                estimator = LassoCV(random_state=self.config.random_state)
            selector = SelectFromModel(estimator, max_features=n_features)
        
        else:
            raise ValueError(f"Unknown feature selection method: {self.config.feature_selection_method}")
        
        # Fit selector and transform data
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_features = X_train.columns[selector.get_support()].tolist()
        else:
            selected_features = X_train.columns[:n_features].tolist()  # Fallback
        
        # Convert back to DataFrame
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        self.transformations["feature_selector"] = selector
        self.log(f"Selected {len(selected_features)} features from {X_train.shape[1]} original features")
        
        return X_train_selected, X_test_selected
    
    def split_data(self, df: pd.DataFrame, target_col: str, 
                   problem_type: ProblemTypeEnum) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets"""
        self.log(f"Splitting data: test_size={self.config.test_size}, stratify={self.config.stratify}")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine stratification
        stratify = None
        if self.config.stratify and problem_type == ProblemTypeEnum.CLASSIFICATION:
            # Check if stratification is possible
            value_counts = y.value_counts()
            min_class_count = value_counts.min()
            
            if min_class_count >= self.config.min_samples_per_class:
                stratify = y
                self.log("Using stratified split for classification")
            else:
                self.log(f"Skipping stratification: minimum class has only {min_class_count} samples")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify
        )
        
        self.log(f"Split completed: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_manual_feature_selection(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply manual feature selection if specified in config"""
        if not self.config.selected_features:
            self.log("No manual feature selection specified, using all features")
            return df
        
        self.log(f"Applying manual feature selection: {len(self.config.selected_features)} features selected")
        
        # Ensure target column is included
        selected_features = list(self.config.selected_features)
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        # Check which features exist in the dataframe
        available_features = df.columns.tolist()
        valid_features = [col for col in selected_features if col in available_features]
        missing_features = [col for col in selected_features if col not in available_features]
        
        if missing_features:
            self.log(f"Warning: Selected features not found in dataset: {missing_features}", "warning")
        
        if not valid_features or (len(valid_features) == 1 and valid_features[0] == target_col):
            raise ValueError(f"No valid features selected. Available features: {available_features}")
        
        # Filter dataframe to only include selected features
        df_filtered = df[valid_features].copy()
        
        self.log(f"Manual feature selection applied: {len(valid_features)} features retained")
        self.log(f"Selected features: {[col for col in valid_features if col != target_col]}")
        
        return df_filtered
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str, 
                       problem_type: ProblemTypeEnum) -> PreprocessingResult:
        """Main preprocessing pipeline"""
        self.log("Starting data preprocessing pipeline")
        
        # Analyze original data quality
        self.preprocessing_summary = self.analyze_data_quality(df, target_col)
        self.log(f"Data analysis: {self.preprocessing_summary['total_rows']} rows, "
                f"{self.preprocessing_summary['total_columns']} columns")
        
        # Check memory usage
        memory_mb = self.preprocessing_summary["memory_usage_mb"]
        if memory_mb > self.config.max_memory_mb:
            self.log(f"Warning: Dataset size ({memory_mb:.1f}MB) exceeds limit ({self.config.max_memory_mb}MB)")
        
        # Step 0: Apply manual feature selection FIRST (if specified)
        df_selected = self.apply_manual_feature_selection(df, target_col)
        
        # Step 1: Handle missing values
        df_cleaned = self.handle_missing_values(df_selected, target_col)
        
        # Step 2: Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_cleaned, target_col, problem_type)
        
        # Step 3: Split the data
        X_train, X_test, y_train, y_test = self.split_data(df_encoded, target_col, problem_type)
        
        # Step 4: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 5: Automatic feature selection (optional, only if no manual selection was done)
        if not self.config.selected_features and self.config.feature_selection_method:
            X_train_final, X_test_final = self.select_features(X_train_scaled, X_test_scaled, y_train, problem_type)
        else:
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
            if self.config.selected_features:
                self.log("Skipping automatic feature selection because manual selection was applied")
        
        # Store final feature names
        self.feature_names = X_train_final.columns.tolist()
        
        # Create preprocessing summary
        preprocessing_steps = []
        if self.config.selected_features:
            preprocessing_steps.append(f"manual_feature_selection_{len(self.config.selected_features)}_features")
        if self.config.missing_strategy != "none":
            preprocessing_steps.append(f"missing_values_{self.config.missing_strategy}")
        if self.config.categorical_strategy != "none":
            preprocessing_steps.append(f"categorical_{self.config.categorical_strategy}")
        if self.config.scaling_strategy != "none":
            preprocessing_steps.append(f"scaling_{self.config.scaling_strategy}")
        if not self.config.selected_features and self.config.feature_selection_method:
            preprocessing_steps.append(f"feature_selection_{self.config.feature_selection_method}")
        
        self.log(f"Preprocessing completed: {len(preprocessing_steps)} steps applied")
        self.log(f"Final feature set: {len(self.feature_names)} features")
        
        return PreprocessingResult(
            X_train=X_train_final,
            X_test=X_test_final,
            y_train=y_train,
            y_test=y_test,
            preprocessing_steps=preprocessing_steps,
            transformations=self.transformations,
            feature_names=self.feature_names,
            preprocessing_summary=self.preprocessing_summary
        )


# Prefect Tasks for Data Preprocessing

@task(name="load_and_validate_data")
def load_and_validate_data(file_path: str, target_column: str) -> pd.DataFrame:
    """Load dataset and perform basic validation"""
    logger = get_run_logger()
    logger.info(f"Loading dataset from: {file_path}")
    
    try:
        # Support multiple file formats
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        if df[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' contains only null values")
        
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def generate_unique_random_state(pipeline_run_id: str, component: str = "preprocessing") -> int:
    """Generate a unique but deterministic random state based on pipeline run ID"""
    import hashlib
    
    # Create deterministic but unique random seed from pipeline run ID
    hash_input = f"{pipeline_run_id}_{component}"
    hash_object = hashlib.md5(hash_input.encode())
    unique_seed = int(hash_object.hexdigest()[:8], 16) % (2**31 - 1)  # Ensure it's a valid int32
    return unique_seed


@task(name="create_preprocessing_config")
def create_preprocessing_config(
    algorithm_names: List[str],
    problem_type: str,
    custom_config: Optional[Dict[str, Any]] = None,
    pipeline_run_id: Optional[str] = None
) -> PreprocessingConfig:
    """Create preprocessing configuration based on algorithm recommendations"""
    logger = get_run_logger()
    logger.info(f"Creating preprocessing config for algorithms: {algorithm_names}")
    
    # Get algorithm registry and recommendations
    registry = get_algorithm_registry()
    problem_type_enum = ProblemTypeEnum(problem_type)
    
    # Aggregate preprocessing recommendations from all algorithms
    recommended_steps = set()
    for algo_name in algorithm_names:
        try:
            algo_enum = AlgorithmNameEnum(algo_name)
            algo_def = registry.get_algorithm(algo_enum)
            if algo_def:
                recommended_steps.update(step.value for step in algo_def.recommended_preprocessing)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not get recommendations for algorithm {algo_name}: {e}")
    
    logger.info(f"Recommended preprocessing steps: {list(recommended_steps)}")
    
    # Create default config with unique random state
    unique_random_state = generate_unique_random_state(pipeline_run_id) if pipeline_run_id else 42
    config = PreprocessingConfig(random_state=unique_random_state)
    
    # Apply algorithm-specific recommendations
    if "scale_features" in recommended_steps:
        config.scaling_strategy = "standard"
    else:
        config.scaling_strategy = "none"
    
    if "encode_categorical" in recommended_steps:
        config.categorical_strategy = "onehot"
    
    if "handle_missing" in recommended_steps:
        config.missing_strategy = "mean"
    
    # Apply custom configuration overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Applied custom config: {key} = {value}")
            elif key in ["feature_columns", "features"]:
                # Handle user's explicit feature selection
                config.selected_features = value
                logger.info(f"Applied manual feature selection: {len(value)} features = {value}")
    
    # Log final configuration
    if config.selected_features:
        logger.info(f"Manual feature selection enabled: {config.selected_features}")
    else:
        logger.info("No manual feature selection, will use all available features")
    
    logger.info(f"Using random_state: {config.random_state} for pipeline {pipeline_run_id}")
    return config


@task(name="preprocess_dataset")
def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    config: PreprocessingConfig
) -> PreprocessingResult:
    """Execute complete data preprocessing pipeline"""
    logger = get_run_logger()
    logger.info("Starting data preprocessing pipeline")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    preprocessor.set_logger(logger)
    
    # Convert problem type
    problem_type_enum = ProblemTypeEnum(problem_type)
    
    # Execute preprocessing
    result = preprocessor.preprocess_data(df, target_column, problem_type_enum)
    
    logger.info(f"Preprocessing completed successfully:")
    logger.info(f"  - Training set: {result.X_train.shape}")
    logger.info(f"  - Test set: {result.X_test.shape}")
    logger.info(f"  - Features: {len(result.feature_names)}")
    logger.info(f"  - Steps applied: {result.preprocessing_steps}")
    
    return result


@task(name="validate_preprocessing_result")
def validate_preprocessing_result(result: PreprocessingResult) -> Dict[str, Any]:
    """Validate preprocessing results and return quality metrics"""
    logger = get_run_logger()
    logger.info("Validating preprocessing results")
    
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "metrics": {}
    }
    
    try:
        # Check for empty datasets
        if result.X_train.empty or result.X_test.empty:
            validation_results["errors"].append("Training or test set is empty after preprocessing")
            validation_results["is_valid"] = False
        
        # Check for missing values
        train_missing = result.X_train.isnull().sum().sum()
        test_missing = result.X_test.isnull().sum().sum()
        
        if train_missing > 0:
            validation_results["warnings"].append(f"Training set has {train_missing} missing values")
        
        if test_missing > 0:
            validation_results["warnings"].append(f"Test set has {test_missing} missing values")
        
        # Check feature consistency
        if list(result.X_train.columns) != list(result.X_test.columns):
            validation_results["errors"].append("Training and test sets have different features")
            validation_results["is_valid"] = False
        
        # Calculate metrics
        validation_results["metrics"] = {
            "train_samples": len(result.X_train),
            "test_samples": len(result.X_test),
            "n_features": len(result.feature_names),
            "train_missing_values": int(train_missing),
            "test_missing_values": int(test_missing),
            "preprocessing_steps_count": len(result.preprocessing_steps)
        }
        
        if validation_results["is_valid"]:
            logger.info("Preprocessing validation passed")
        else:
            logger.error(f"Preprocessing validation failed: {validation_results['errors']}")
        
        if validation_results["warnings"]:
            for warning in validation_results["warnings"]:
                logger.warning(warning)
        
    except Exception as e:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Validation error: {str(e)}")
        logger.error(f"Preprocessing validation failed: {str(e)}")
    
    return validation_results


# Main Preprocessing Flow

@flow(name="data_preprocessing_flow", task_runner=SequentialTaskRunner())
def data_preprocessing_flow(
    file_path: str,
    target_column: str,
    problem_type: str,
    algorithm_names: List[str],
    custom_config: Optional[Dict[str, Any]] = None,
    pipeline_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete data preprocessing flow for ML training
    
    Args:
        file_path: Path to the dataset file
        target_column: Name of the target variable column
        problem_type: 'classification' or 'regression'
        algorithm_names: List of algorithm names to optimize preprocessing for
        custom_config: Optional custom preprocessing configuration
        pipeline_run_id: Unique pipeline run ID for generating unique random seeds
    
    Returns:
        Dictionary containing preprocessing results and validation
    """
    logger = get_run_logger()
    logger.info(f"Starting preprocessing flow for {problem_type} problem (run: {pipeline_run_id})")
    
    try:
        # Step 1: Load and validate data
        df = load_and_validate_data(file_path, target_column)
        
        # Step 2: Create preprocessing configuration with unique random state
        config = create_preprocessing_config(algorithm_names, problem_type, custom_config, pipeline_run_id)
        
        # Step 3: Execute preprocessing
        result = preprocess_dataset(df, target_column, problem_type, config)
        
        # Step 4: Validate results
        validation = validate_preprocessing_result(result)
        
        # Prepare return data
        return_data = {
            "success": validation["is_valid"],
            "preprocessing_result": result,  # Return actual object for ML training
            "validation": validation,
            "config_used": config.to_dict(),  # Convert config to dict for JSON serialization
            "summary": {
                "original_shape": df.shape,
                "final_train_shape": result.X_train.shape,
                "final_test_shape": result.X_test.shape,
                "preprocessing_steps": result.preprocessing_steps,
                "feature_names": result.feature_names
            }
        }
        
        if validation["is_valid"]:
            logger.info("Data preprocessing flow completed successfully")
        else:
            logger.error(f"Data preprocessing flow failed validation: {validation['errors']}")
        
        return return_data
        
    except Exception as e:
        logger.error(f"Data preprocessing flow failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "preprocessing_result": None,
            "validation": {"is_valid": False, "errors": [str(e)]},
            "config_used": None,
            "summary": {}
        }


# Utility functions for external integration

def get_preprocessing_recommendations(algorithm_names: List[str]) -> Dict[str, Any]:
    """Get preprocessing recommendations for given algorithms"""
    registry = get_algorithm_registry()
    recommendations = {
        "scaling_needed": False,
        "categorical_encoding_needed": False,
        "missing_value_handling_needed": True,  # Always recommended
        "recommended_steps": set(),
        "algorithm_specific": {}
    }
    
    for algo_name in algorithm_names:
        try:
            algo_enum = AlgorithmNameEnum(algo_name)
            algo_def = registry.get_algorithm(algo_enum)
            
            if algo_def:
                algo_steps = [step.value for step in algo_def.recommended_preprocessing]
                recommendations["algorithm_specific"][algo_name] = algo_steps
                recommendations["recommended_steps"].update(algo_steps)
                
                if "scale_features" in algo_steps:
                    recommendations["scaling_needed"] = True
                if "encode_categorical" in algo_steps:
                    recommendations["categorical_encoding_needed"] = True
                    
        except (ValueError, AttributeError):
            continue
    
    recommendations["recommended_steps"] = list(recommendations["recommended_steps"])
    return recommendations


def validate_preprocessing_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate preprocessing configuration dictionary"""
    try:
        # Try to create PreprocessingConfig from dictionary
        config = PreprocessingConfig(**config_dict)
        return {"valid": True, "config": config, "errors": []}
    except Exception as e:
        return {"valid": False, "config": None, "errors": [str(e)]} 