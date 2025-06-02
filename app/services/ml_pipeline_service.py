"""
ML Pipeline Service for DS1.3.3
Service layer for ML pipeline orchestration, configuration validation, progress tracking, and result storage
"""

import logging
import uuid
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import sys
import os

from fastapi import HTTPException
from sqlmodel import Session, select

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'workflows', 'pipelines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'workflows', 'ml'))

from app.models.pipeline_models import PipelineRun, PipelineType, PipelineRunStatus
from app.models.file_models import UploadedFileLog
from app.models.ml_models import (
    MLPipelineRun, MLPipelineConfig, ProblemTypeEnum, AlgorithmNameEnum,
    MLResult, ModelMetrics, MLPipelineCreateRequest, MLPipelineCreateResponse,
    MLPipelineStatusResponse, MLPipelineResultResponse, MLModel
)

# Import ML workflow components
from workflows.pipelines.ml_training import (
    ml_training_flow, create_ml_training_config, 
    get_algorithm_suggestions, validate_algorithm_config
)
from workflows.ml.evaluation import (
    evaluate_model_comprehensive, compare_models, export_evaluation_report,
    ComprehensiveEvaluation
)

logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """
    Main orchestrator for ML pipeline operations
    Handles configuration validation, workflow execution, and result management
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.supported_file_types = ['.csv', '.xlsx', '.xls']
        self.max_file_size_mb = 100  # Maximum file size in MB
        self.max_rows = 50000  # Maximum number of rows to process
    
    def validate_ml_config(self, config: MLPipelineCreateRequest) -> Dict[str, Any]:
        """
        Validate ML pipeline configuration
        
        Args:
            config: ML pipeline configuration request
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Validate file exists and is accessible
            uploaded_file = self.db.get(UploadedFileLog, config.uploaded_file_log_id)
            if not uploaded_file:
                validation_result["errors"].append(f"File with ID {config.uploaded_file_log_id} not found")
                validation_result["is_valid"] = False
                return validation_result
            
            file_path = Path(uploaded_file.storage_location)
            if not file_path.exists():
                validation_result["errors"].append(f"File not found at {file_path}")
                validation_result["is_valid"] = False
                return validation_result
            
            # Validate file type
            if file_path.suffix.lower() not in self.supported_file_types:
                validation_result["errors"].append(
                    f"Unsupported file type: {file_path.suffix}. Supported types: {self.supported_file_types}"
                )
                validation_result["is_valid"] = False
            
            # Validate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                validation_result["errors"].append(
                    f"File size ({file_size_mb:.1f}MB) exceeds maximum ({self.max_file_size_mb}MB)"
                )
                validation_result["is_valid"] = False
            
            # Validate problem type
            try:
                ProblemTypeEnum(config.problem_type.upper())
            except ValueError:
                validation_result["errors"].append(f"Invalid problem type: {config.problem_type}")
                validation_result["is_valid"] = False
            
            # Validate algorithms
            valid_algorithms = []
            for algo_config in config.algorithms:
                try:
                    algo_validation = validate_algorithm_config(algo_config.dict())
                    if algo_validation.get("is_valid", False):
                        valid_algorithms.append(algo_config)
                    else:
                        validation_result["warnings"].append(
                            f"Invalid algorithm config for {algo_config.name}: {algo_validation.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    validation_result["warnings"].append(f"Error validating algorithm {algo_config.name}: {str(e)}")
            
            if not valid_algorithms:
                validation_result["errors"].append("No valid algorithms provided")
                validation_result["is_valid"] = False
            elif len(valid_algorithms) < len(config.algorithms):
                validation_result["warnings"].append(
                    f"Only {len(valid_algorithms)} out of {len(config.algorithms)} algorithms are valid"
                )
            
            # Validate target column (basic check - will be validated further during data loading)
            if not config.target_column or not config.target_column.strip():
                validation_result["errors"].append("Target column name is required")
                validation_result["is_valid"] = False
            
            # Validate preprocessing config
            if config.preprocessing_config:
                preprocessing = config.preprocessing_config
                
                # Validate missing value strategy
                valid_missing_strategies = ['mean', 'median', 'mode', 'drop', 'knn']
                if preprocessing.missing_strategy and preprocessing.missing_strategy not in valid_missing_strategies:
                    validation_result["warnings"].append(
                        f"Unknown missing value strategy: {preprocessing.missing_strategy}. Using default."
                    )
                
                # Validate categorical strategy
                valid_categorical_strategies = ['onehot', 'label', 'ordinal']
                if preprocessing.categorical_strategy and preprocessing.categorical_strategy not in valid_categorical_strategies:
                    validation_result["warnings"].append(
                        f"Unknown categorical strategy: {preprocessing.categorical_strategy}. Using default."
                    )
                
                # Validate scaling strategy
                valid_scaling_strategies = ['standard', 'minmax', 'robust', 'none']
                if preprocessing.scaling_strategy and preprocessing.scaling_strategy not in valid_scaling_strategies:
                    validation_result["warnings"].append(
                        f"Unknown scaling strategy: {preprocessing.scaling_strategy}. Using default."
                    )
                
                # Validate test size
                if preprocessing.test_size:
                    if not (0.1 <= preprocessing.test_size <= 0.5):
                        validation_result["warnings"].append(
                            f"Test size {preprocessing.test_size} is outside recommended range [0.1, 0.5]"
                        )
            
            # Generate recommendations
            if len(config.algorithms) > 5:
                validation_result["recommendations"].append(
                    "Consider reducing the number of algorithms to speed up training"
                )
            
            if file_size_mb > 50:
                validation_result["recommendations"].append(
                    "Large file detected. Consider using a subset for faster experimentation"
                )
            
        except Exception as e:
            logger.exception("Error during ML config validation")
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def create_ml_pipeline_run(self, config: MLPipelineCreateRequest) -> MLPipelineCreateResponse:
        """
        Create and execute ML pipeline run
        
        Args:
            config: ML pipeline configuration
            
        Returns:
            Response with run details
        """
        logger.info(f"Creating ML pipeline run for file ID: {config.uploaded_file_log_id}")
        
        # Validate configuration
        validation_result = self.validate_ml_config(config)
        if not validation_result["is_valid"]:
            logger.error(f"ML config validation failed: {validation_result['errors']}")
            raise HTTPException(
                status_code=400, 
                detail=f"Configuration validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Get uploaded file info
        uploaded_file = self.db.get(UploadedFileLog, config.uploaded_file_log_id)
        if not uploaded_file:
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        
        # Create base pipeline run
        pipeline_run = PipelineRun(
            uploaded_file_log_id=config.uploaded_file_log_id,
            pipeline_type=PipelineType.ML_TRAINING,
            status=PipelineRunStatus.PENDING
        )
        
        try:
            self.db.add(pipeline_run)
            self.db.commit()
            self.db.refresh(pipeline_run)
            
            # Create ML-specific pipeline run
            ml_pipeline_run = MLPipelineRun(
                run_uuid=pipeline_run.run_uuid,
                problem_type=ProblemTypeEnum(config.problem_type.upper()),
                target_variable=config.target_column,
                algorithms_config={"algorithms": [algo.dict() for algo in config.algorithms]},
                preprocessing_config=config.preprocessing_config.dict() if config.preprocessing_config else {},
                validation_results=validation_result
            )
            
            self.db.add(ml_pipeline_run)
            self.db.commit()
            self.db.refresh(ml_pipeline_run)
            
            logger.info(f"Created ML pipeline run with UUID: {pipeline_run.run_uuid}")
            
            # Execute the pipeline asynchronously (in production, this would be submitted to a task queue)
            try:
                result = self._execute_ml_pipeline(pipeline_run.run_uuid, config, uploaded_file.storage_location)
                
                return MLPipelineCreateResponse(
                    run_uuid=pipeline_run.run_uuid,
                    status=PipelineRunStatus.COMPLETED if result.get("success") else PipelineRunStatus.FAILED,
                    uploaded_file_log_id=config.uploaded_file_log_id,
                    message="ML pipeline executed successfully" if result.get("success") else result.get("error", "Pipeline failed"),
                    validation_warnings=validation_result.get("warnings", []),
                    validation_recommendations=validation_result.get("recommendations", [])
                )
                
            except Exception as e:
                logger.exception(f"Error executing ML pipeline for run UUID: {pipeline_run.run_uuid}")
                
                # Update status to failed
                pipeline_run.status = PipelineRunStatus.FAILED
                pipeline_run.error_message = str(e)
                pipeline_run.updated_at = datetime.now(timezone.utc)
                self.db.add(pipeline_run)
                self.db.commit()
                
                raise HTTPException(status_code=500, detail=f"ML pipeline execution failed: {str(e)}")
        
        except Exception as e:
            self.db.rollback()
            logger.exception("Error creating ML pipeline run")
            raise HTTPException(status_code=500, detail=f"Failed to create ML pipeline run: {str(e)}")
    
    def _execute_ml_pipeline(self, run_uuid: uuid.UUID, config: MLPipelineCreateRequest, file_path: str) -> Dict[str, Any]:
        """
        Execute the ML training pipeline
        
        Args:
            run_uuid: Pipeline run UUID
            config: ML pipeline configuration
            file_path: Path to the data file
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing ML pipeline for run UUID: {run_uuid}")
        
        try:
            # Update status to running
            pipeline_run = self.db.get(PipelineRun, run_uuid)
            pipeline_run.status = PipelineRunStatus.RUNNING
            pipeline_run.updated_at = datetime.now(timezone.utc)
            self.db.add(pipeline_run)
            self.db.commit()
            
            # Create ML training configuration
            algorithms_list = []
            for algo in config.algorithms:
                algo_dict = {"name": algo.name}
                if algo.hyperparameters:
                    algo_dict["hyperparameters"] = algo.hyperparameters
                algorithms_list.append(algo_dict)
            
            # Build preprocessing config
            preprocessing_config = {}
            if config.preprocessing_config:
                if config.preprocessing_config.missing_strategy:
                    preprocessing_config["missing_strategy"] = config.preprocessing_config.missing_strategy
                if config.preprocessing_config.categorical_strategy:
                    preprocessing_config["categorical_strategy"] = config.preprocessing_config.categorical_strategy
                if config.preprocessing_config.scaling_strategy:
                    preprocessing_config["scaling_strategy"] = config.preprocessing_config.scaling_strategy
                if config.preprocessing_config.test_size:
                    preprocessing_config["test_size"] = config.preprocessing_config.test_size
            
            # Create training config
            training_config = create_ml_training_config(
                file_path=file_path,
                target_column=config.target_column,
                problem_type=config.problem_type.lower(),
                algorithms=algorithms_list,
                preprocessing_config=preprocessing_config,
                pipeline_run_id=str(run_uuid)
            )
            
            # Execute the ML training flow
            flow_result = ml_training_flow(training_config)
            
            if flow_result.get("success"):
                # Process and store results
                ml_result = flow_result["result"]
                self._store_ml_results(run_uuid, ml_result, flow_result)
                
                # Update pipeline status
                pipeline_run.status = PipelineRunStatus.COMPLETED
                pipeline_run.result = {
                    "summary": ml_result.summary,
                    "best_model": ml_result.best_model,
                    "total_time": ml_result.total_training_time
                }
                pipeline_run.updated_at = datetime.now(timezone.utc)
                self.db.add(pipeline_run)
                self.db.commit()
                
                logger.info(f"ML pipeline completed successfully for run UUID: {run_uuid}")
                return flow_result
                
            else:
                # Handle failure
                error_message = flow_result.get("error", "Unknown error during ML training")
                
                pipeline_run.status = PipelineRunStatus.FAILED
                pipeline_run.error_message = error_message
                pipeline_run.updated_at = datetime.now(timezone.utc)
                self.db.add(pipeline_run)
                self.db.commit()
                
                logger.error(f"ML pipeline failed for run UUID: {run_uuid}: {error_message}")
                return flow_result
        
        except Exception as e:
            logger.exception(f"Error executing ML pipeline for run UUID: {run_uuid}")
            
            # Update status to failed
            pipeline_run = self.db.get(PipelineRun, run_uuid)
            if pipeline_run:
                pipeline_run.status = PipelineRunStatus.FAILED
                pipeline_run.error_message = str(e)
                pipeline_run.updated_at = datetime.now(timezone.utc)
                self.db.add(pipeline_run)
                self.db.commit()
            
            return {"success": False, "error": str(e)}
    
    def _store_ml_results(self, run_uuid: uuid.UUID, ml_result, flow_result: Dict[str, Any]):
        """
        Store detailed ML results in the database
        
        Args:
            run_uuid: Pipeline run UUID
            ml_result: ML training result object
            flow_result: Flow execution result
        """
        try:
            # Get ML pipeline run
            ml_pipeline_run = self.db.exec(
                select(MLPipelineRun).where(MLPipelineRun.run_uuid == run_uuid)
            ).one_or_none()
            
            if not ml_pipeline_run:
                logger.warning(f"ML pipeline run not found for UUID: {run_uuid}")
                return
            
            # Create ML result record
            result_data = MLResult(
                pipeline_run_id=run_uuid,
                summary=ml_result.summary,
                best_model_algorithm=ml_result.best_model.get("algorithm_name"),
                best_model_metrics=ml_result.best_model.get("metrics", {}),
                model_comparison=ml_result.aggregated_metrics,
                training_metadata={
                    "total_training_time": ml_result.total_training_time,
                    "algorithms_attempted": ml_result.summary.get("algorithms_attempted", 0),
                    "algorithms_successful": ml_result.summary.get("algorithms_successful", 0),
                    "preprocessing_steps": ml_result.summary.get("dataset_info", {}).get("preprocessing_steps", [])
                }
            )
            
            # Store individual model metrics
            model_metrics_list = []
            ml_models_list = []
            for idx, eval_result in enumerate(ml_result.evaluation_results):
                if not eval_result.error:
                    # Get corresponding training result
                    training_result = None
                    for tr in ml_result.training_results:
                        if tr.algorithm_name == eval_result.algorithm_name:
                            training_result = tr
                            break
                    
                    # Create ModelMetrics record
                    metrics_data = ModelMetrics(
                        pipeline_run_id=run_uuid,
                        algorithm_name=eval_result.algorithm_name,
                        metrics=eval_result.metrics,
                        feature_importance=eval_result.feature_importance,
                        model_artifacts={
                            "confusion_matrix": eval_result.confusion_matrix.tolist() if eval_result.confusion_matrix is not None else None,
                            "classification_report": eval_result.classification_report
                        }
                    )
                    model_metrics_list.append(metrics_data)
                    
                    # Create individual MLModel record
                    import time
                    timestamp = int(time.time() * 1000)
                    unique_model_id = f"{run_uuid}_{eval_result.algorithm_name}_{timestamp}_{idx}"
                    
                    ml_model = MLModel(
                        model_id=unique_model_id,
                        pipeline_run_id=ml_pipeline_run.id,
                        algorithm_name=eval_result.algorithm_name,
                        hyperparameters=training_result.hyperparameters if training_result else {},
                        performance_metrics=eval_result.metrics,
                        primary_metric_value=eval_result.metrics.get('accuracy') or eval_result.metrics.get('r2'),
                        primary_metric_name='accuracy' if 'accuracy' in eval_result.metrics else 'r2',
                        training_time_seconds=training_result.training_time if training_result else 0,
                        feature_importance=eval_result.feature_importance,
                        model_file_path=training_result.model_path if training_result else None,
                        training_status="completed"
                    )
                    ml_models_list.append(ml_model)
            
            # Update ML pipeline run with results
            ml_pipeline_run.metrics = result_data.best_model_metrics
            ml_pipeline_run.best_model_id = result_data.best_model_algorithm
            ml_pipeline_run.completed_at = datetime.now(timezone.utc)
            
            # Save to database
            self.db.add(result_data)
            for metrics in model_metrics_list:
                self.db.add(metrics)
            for ml_model in ml_models_list:
                self.db.add(ml_model)
            self.db.add(ml_pipeline_run)
            self.db.commit()
            
            logger.info(f"Stored ML results for run UUID: {run_uuid}")
            
        except Exception as e:
            logger.exception(f"Error storing ML results for run UUID: {run_uuid}")
            # Don't fail the pipeline if result storage fails
    
    def get_ml_pipeline_status(self, run_uuid: uuid.UUID) -> Optional[MLPipelineStatusResponse]:
        """
        Get ML pipeline run status
        
        Args:
            run_uuid: Pipeline run UUID
            
        Returns:
            ML pipeline status response or None if not found
        """
        try:
            # Get base pipeline run
            pipeline_run = self.db.get(PipelineRun, run_uuid)
            if not pipeline_run:
                return None
            
            # Get ML-specific run data
            ml_pipeline_run = self.db.exec(
                select(MLPipelineRun).where(MLPipelineRun.run_uuid == run_uuid)
            ).one_or_none()
            
            return MLPipelineStatusResponse(
                run_uuid=pipeline_run.run_uuid,
                status=pipeline_run.status,
                pipeline_type=pipeline_run.pipeline_type,
                uploaded_file_log_id=pipeline_run.uploaded_file_log_id,
                problem_type=ml_pipeline_run.problem_type if ml_pipeline_run else None,
                target_variable=ml_pipeline_run.target_variable if ml_pipeline_run else None,
                algorithms_config=ml_pipeline_run.algorithms_config if ml_pipeline_run else {},
                preprocessing_config=ml_pipeline_run.preprocessing_config if ml_pipeline_run else {},
                validation_results=ml_pipeline_run.validation_results if ml_pipeline_run else {},
                progress_info=self._get_progress_info(pipeline_run.status),
                error_message=pipeline_run.error_message,
                created_at=pipeline_run.created_at,
                updated_at=pipeline_run.updated_at,
                completed_at=ml_pipeline_run.completed_at if ml_pipeline_run else None
            )
            
        except Exception as e:
            logger.exception(f"Error getting ML pipeline status for run UUID: {run_uuid}")
            return None
    
    def get_ml_pipeline_results(self, run_uuid: uuid.UUID) -> Optional[MLPipelineResultResponse]:
        """
        Get detailed ML pipeline results
        
        Args:
            run_uuid: Pipeline run UUID
            
        Returns:
            ML pipeline results response or None if not found
        """
        try:
            # Get status first
            status_response = self.get_ml_pipeline_status(run_uuid)
            if not status_response:
                return None
            
            # If not completed, return status only
            if status_response.status != PipelineRunStatus.COMPLETED:
                return MLPipelineResultResponse(
                    **status_response.dict(),
                    results=None,
                    model_metrics=[],
                    comparison_report=None
                )
            
            # Get detailed results
            ml_result = self.db.exec(
                select(MLResult).where(MLResult.pipeline_run_id == run_uuid)
            ).one_or_none()
            
            model_metrics = self.db.exec(
                select(ModelMetrics).where(ModelMetrics.pipeline_run_id == run_uuid)
            ).all()
            
            # Generate comparison report
            comparison_report = None
            if ml_result and model_metrics:
                comparison_report = self._generate_comparison_report(ml_result, model_metrics)
            
            return MLPipelineResultResponse(
                **status_response.dict(),
                results=ml_result,
                model_metrics=list(model_metrics),
                comparison_report=comparison_report
            )
            
        except Exception as e:
            logger.exception(f"Error getting ML pipeline results for run UUID: {run_uuid}")
            return None
    
    def _get_progress_info(self, status: PipelineRunStatus) -> Dict[str, Any]:
        """Get progress information based on pipeline status"""
        progress_mapping = {
            PipelineRunStatus.PENDING: {"stage": "Initializing", "percentage": 0},
            PipelineRunStatus.RUNNING: {"stage": "Training Models", "percentage": 50},
            PipelineRunStatus.COMPLETED: {"stage": "Completed", "percentage": 100},
            PipelineRunStatus.FAILED: {"stage": "Failed", "percentage": 0}
        }
        return progress_mapping.get(status, {"stage": "Unknown", "percentage": 0})
    
    def _generate_comparison_report(self, ml_result: MLResult, model_metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Generate a comparison report for all trained models"""
        report = {
            "summary": {
                "total_models": len(model_metrics),
                "best_model": ml_result.best_model_algorithm,
                "training_time": ml_result.training_metadata.get("total_training_time", 0)
            },
            "model_rankings": {},
            "performance_comparison": []
        }
        
        # Determine primary metric based on problem type
        # This would need to be extracted from the ML pipeline run
        # For now, use a simple heuristic
        has_accuracy = any("accuracy" in metrics.metrics for metrics in model_metrics)
        primary_metric = "accuracy" if has_accuracy else "r2"
        
        # Create performance comparison
        for metrics in model_metrics:
            model_perf = {
                "algorithm": metrics.algorithm_name,
                "primary_metric": metrics.metrics.get(primary_metric, 0),
                "all_metrics": metrics.metrics
            }
            report["performance_comparison"].append(model_perf)
        
        # Sort by primary metric (higher is better for most metrics)
        report["performance_comparison"].sort(
            key=lambda x: x["primary_metric"], 
            reverse=True
        )
        
        # Create rankings
        for i, model in enumerate(report["performance_comparison"]):
            report["model_rankings"][model["algorithm"]] = i + 1
        
        return report


# Service functions for external use

def create_ml_pipeline(db: Session, config: MLPipelineCreateRequest) -> MLPipelineCreateResponse:
    """
    Create and execute ML pipeline
    
    Args:
        db: Database session
        config: ML pipeline configuration
        
    Returns:
        ML pipeline creation response
    """
    orchestrator = MLPipelineOrchestrator(db)
    return orchestrator.create_ml_pipeline_run(config)


def get_ml_pipeline_status(db: Session, run_uuid: uuid.UUID) -> Optional[MLPipelineStatusResponse]:
    """
    Get ML pipeline status
    
    Args:
        db: Database session
        run_uuid: Pipeline run UUID
        
    Returns:
        ML pipeline status or None if not found
    """
    orchestrator = MLPipelineOrchestrator(db)
    return orchestrator.get_ml_pipeline_status(run_uuid)


def get_ml_pipeline_results(db: Session, run_uuid: uuid.UUID) -> Optional[MLPipelineResultResponse]:
    """
    Get ML pipeline results
    
    Args:
        db: Database session
        run_uuid: Pipeline run UUID
        
    Returns:
        ML pipeline results or None if not found
    """
    orchestrator = MLPipelineOrchestrator(db)
    return orchestrator.get_ml_pipeline_results(run_uuid)


def get_algorithm_suggestions_service(problem_type: str) -> List[Dict[str, Any]]:
    """
    Get algorithm suggestions for a problem type
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        List of algorithm suggestions
    """
    try:
        return get_algorithm_suggestions(problem_type)
    except Exception as e:
        logger.exception(f"Error getting algorithm suggestions for {problem_type}")
        return []


def validate_ml_configuration(db: Session, config: MLPipelineCreateRequest) -> Dict[str, Any]:
    """
    Validate ML pipeline configuration
    
    Args:
        db: Database session
        config: ML pipeline configuration
        
    Returns:
        Validation results
    """
    orchestrator = MLPipelineOrchestrator(db)
    return orchestrator.validate_ml_config(config) 