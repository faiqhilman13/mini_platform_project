# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session, select
import sqlalchemy

from app.core.config import settings
from app.db.session import get_session
from app.models.pipeline_models import PipelineType, PipelineRunStatusResponse, PipelineRun
from app.models.ml_models import MLPipelineRun, MLModel
from app.services.pipeline_service import trigger_pipeline_flow, get_pipeline_run_status

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/pipelines/ml",
    tags=["ML Pipelines"],
)

algorithms_router = APIRouter(
    prefix=f"{settings.API_V1_STR}/algorithms",
    tags=["ML Algorithms"],
)


@router.post("/trigger")
def trigger_ml_pipeline(
    request: dict,
    db: Session = Depends(get_session)
):
    """
    Create and execute ML training pipeline.
    
    This endpoint creates a new ML pipeline run with the specified configuration
    and executes the training workflow.
    """
    logger.info(f"Received ML pipeline trigger request for file ID: {request.get('uploaded_file_log_id')}")
    
    try:
        # Convert request to pipeline format
        config = {
            "target_variable": request.get("target_column") or request.get("target_variable"),
            "problem_type": request.get("problem_type", "classification"),
            "algorithms": request.get("algorithms", []),
            "preprocessing_config": request.get("preprocessing_config", {})
        }
        
        response = trigger_pipeline_flow(
            db=db,
            uploaded_file_log_id=request["uploaded_file_log_id"],
            pipeline_type=PipelineType.ML_TRAINING,
            config=config
        )
        
        logger.info(f"ML pipeline triggered successfully: {response.run_uuid}")
        return {"run_uuid": str(response.run_uuid), "status": response.status.value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error triggering ML pipeline")
        raise HTTPException(status_code=500, detail=f"Failed to trigger ML pipeline: {str(e)}")


@router.get("/status/{run_uuid}")
def get_ml_pipeline_status_endpoint(
    run_uuid: uuid.UUID,
    db: Session = Depends(get_session)
):
    """
    Get ML pipeline run status and progress.
    
    Returns detailed status information including results and error messages.
    """
    logger.info(f"Getting ML pipeline status for run UUID: {run_uuid}")
    
    try:
        status = get_pipeline_run_status(db, run_uuid)
        if not status:
            raise HTTPException(status_code=404, detail="ML pipeline run not found")
        
        # Extract ML-specific data from the result
        ml_result = None
        problem_type = "unknown"
        target_variable = "unknown"
        best_model_id = None
        
        if status.result and status.result.get("success"):
            ml_result = status.result.get("result", {})
            problem_type = ml_result.get("problem_type", "unknown")
            target_variable = ml_result.get("target_variable", "unknown")
            
            # Find best model
            if "best_model" in ml_result and ml_result["best_model"]:
                best_model_info = ml_result["best_model"]
                best_model_id = best_model_info.get("algorithm_name")
        
        # Format response to match what MLResultsPage expects
        response = {
            "run_uuid": str(status.run_uuid),
            "status": status.status.value,
            "problem_type": problem_type,
            "target_variable": target_variable,
            "best_model_id": best_model_id,
            "created_at": status.created_at.isoformat() if status.created_at else None,
            "updated_at": status.updated_at.isoformat() if status.updated_at else None,
            "success": status.status.value == "COMPLETED",
            "message": f"Pipeline {status.status.value.lower()}",
            "ml_result": status.result,
            "error_message": status.error_message
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting ML pipeline status for {run_uuid}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML pipeline status: {str(e)}")


@router.get("/models/{run_uuid}")
def get_ml_models(
    run_uuid: str,  # Change to string to avoid UUID conversion issues
    db: Session = Depends(get_session)
):
    """
    Get all trained models for a specific ML pipeline run.
    
    Returns metrics and details for all models trained during the pipeline extensions.
    """
    logger.info(f"Getting models for ML pipeline run UUID: {run_uuid}")
    
    try:
        # Normalize UUID formats to handle both with and without hyphens
        run_uuid_str = str(run_uuid)
        run_uuid_no_hyphens = run_uuid_str.replace('-', '')
        
        # Use SQLModel select with text comparison to find matching pipeline
        pipeline_runs = db.exec(select(PipelineRun)).all()
        
        # Find the matching pipeline run by comparing string representations
        pipeline_run = None
        for run in pipeline_runs:
            run_uuid_db = str(run.run_uuid).replace('-', '')
            if run_uuid_db == run_uuid_no_hyphens or str(run.run_uuid) == run_uuid_str:
                pipeline_run = run
                break
        
        if not pipeline_run:
            logger.warning(f"Pipeline run {run_uuid} not found in pipeline_run table")
            return {"error": f"Pipeline run {run_uuid} not found", "models": []}
        
        # Check if it's an ML training pipeline
        if pipeline_run.pipeline_type != PipelineType.ML_TRAINING:
            logger.warning(f"Pipeline run {run_uuid} is not an ML training pipeline")
            return {"error": f"Pipeline run {run_uuid} is not an ML training pipeline", "models": []}
        
        # Extract models from the result
        models = []
        if pipeline_run.result:
            import json
            try:
                result_data = pipeline_run.result if isinstance(pipeline_run.result, dict) else json.loads(str(pipeline_run.result))
                
                if result_data and result_data.get("success"):
                    ml_result = result_data.get("result", {})
                    evaluation_results = ml_result.get("evaluation_results", [])
                    training_results = ml_result.get("training_results", [])
                    
                    # Create lookups for training data
                    training_times = {}
                    hyperparameters_lookup = {}
                    for training_result in training_results:
                        algorithm_name = training_result.get("algorithm_name")
                        training_times[algorithm_name] = training_result.get("training_time", 0)
                        hyperparameters_lookup[algorithm_name] = training_result.get("hyperparameters", {})
                    
                    for idx, eval_result in enumerate(evaluation_results):
                        if not eval_result.get("error"):
                            algorithm_name = eval_result.get("algorithm_name", "unknown")
                            # Create a stable model ID based on pipeline run UUID and algorithm
                            # Use the actual pipeline run UUID from database to ensure uniqueness per training run
                            stable_model_id = f"{pipeline_run.run_uuid}_{algorithm_name}_{idx}"
                            model_data = {
                                "model_id": stable_model_id,
                                "pipeline_run_id": run_uuid_str,
                                "algorithm_name": algorithm_name,
                                "hyperparameters": hyperparameters_lookup.get(algorithm_name, {}),
                                "performance_metrics": eval_result.get("metrics", {}),
                                "model_path": "",
                                "feature_importance": eval_result.get("feature_importance"),
                                "training_time": training_times.get(algorithm_name, 0)
                            }
                            models.append(model_data)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Error parsing result data: {e}")
                return {"error": f"Error parsing pipeline results", "models": []}
        
        if not models:
            return {"error": f"No models found for pipeline run {run_uuid}", "models": []}
        
        return models
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting ML models for {run_uuid}")
        return {"error": f"Failed to get ML models: {str(e)}", "models": []}


@algorithms_router.get("/suggestions")
def get_algorithm_suggestions(
    problem_type: Optional[str] = None
):
    """
    Get algorithm suggestions for ML training.
    
    Returns a list of recommended algorithms with their descriptions.
    """
    logger.info(f"Getting algorithm suggestions for problem type: {problem_type}")
    
    try:
        # Basic algorithm suggestions
        classification_algorithms = [
            {
                "name": "logistic_regression",
                "display_name": "Logistic Regression",
                "description": "Linear model for binary and multiclass classification",
                "complexity": "low",
                "problem_types": ["classification"]
            },
            {
                "name": "random_forest_classifier",
                "display_name": "Random Forest",
                "description": "Ensemble method using multiple decision trees",
                "complexity": "medium",
                "problem_types": ["classification"]
            },
            {
                "name": "decision_tree_classifier", 
                "display_name": "Decision Tree",
                "description": "Tree-based model with interpretable rules",
                "complexity": "medium",
                "problem_types": ["classification"]
            }
        ]
        
        regression_algorithms = [
            {
                "name": "linear_regression",
                "display_name": "Linear Regression",
                "description": "Linear model for continuous target prediction",
                "complexity": "low",
                "problem_types": ["regression"]
            },
            {
                "name": "random_forest_regressor",
                "display_name": "Random Forest",
                "description": "Ensemble method using multiple decision trees",
                "complexity": "medium", 
                "problem_types": ["regression"]
            },
            {
                "name": "decision_tree_regressor",
                "display_name": "Decision Tree",
                "description": "Tree-based model with interpretable rules",
                "complexity": "medium",
                "problem_types": ["regression"]
            }
        ]
        
        if problem_type == "classification":
            return classification_algorithms
        elif problem_type == "regression":
            return regression_algorithms
        else:
            return classification_algorithms + regression_algorithms
        
    except Exception as e:
        logger.exception(f"Error getting algorithm suggestions for {problem_type}")
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm suggestions: {str(e)}")


# Create a combined router to include both ML and algorithms endpoints
ml_router = APIRouter()
ml_router.include_router(router)
ml_router.include_router(algorithms_router) 