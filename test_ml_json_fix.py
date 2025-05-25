#!/usr/bin/env python3
"""
Test script to verify ML training JSON serialization is fixed
"""

import sys
import os
sys.path.append('app')
sys.path.append('workflows/ml')
sys.path.append('workflows/pipelines')

from workflows.pipelines.ml_training import MLTrainingResult, TrainingResult, EvaluationResult
from workflows.ml.preprocessing import PreprocessingResult
import pandas as pd
import numpy as np
import json

def test_ml_json_serialization():
    print("=== Testing ML Training JSON Serialization Fix ===")
    
    # Create dummy data structures similar to real ML results
    preprocessing_result = PreprocessingResult(
        X_train=pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
        X_test=pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10]}),
        y_train=pd.Series([0, 1, 0]),
        y_test=pd.Series([1, 0]),
        preprocessing_steps=['scaling_standard', 'categorical_onehot'],
        transformations={},
        feature_names=['feature1', 'feature2'],
        preprocessing_summary={'total_rows': 5}
    )

    training_result = TrainingResult(
        algorithm_name='linear_regression',
        model=None,  # Don't include actual model for JSON test
        training_time=np.float64(2.5),
        hyperparameters={'fit_intercept': np.bool_(True)},
        feature_importance={'feature1': np.float64(0.6), 'feature2': np.float64(0.4)},
        model_path='/path/to/model.joblib'
    )

    evaluation_result = EvaluationResult(
        algorithm_name='linear_regression',
        metrics={'r2': np.float64(0.8965), 'mse': np.float64(0.1234)},
        predictions=np.array([0.1, 0.9, 0.2, 0.8]),
        prediction_probabilities=None,
        confusion_matrix=None,
        classification_report=None,
        feature_importance={'feature1': np.float64(0.6), 'feature2': np.float64(0.4)}
    )

    ml_result = MLTrainingResult(
        pipeline_run_id='test_run_123',
        problem_type='regression',
        target_variable='exam_score',
        preprocessing_result=preprocessing_result,
        training_results=[training_result],
        evaluation_results=[evaluation_result],
        best_model={'algorithm_name': 'linear_regression', 'r2': np.float64(0.8965)},
        aggregated_metrics={'models_trained': np.int64(1)},
        total_training_time=np.float64(2.5),
        summary={'best_score': np.float64(0.8965)}
    )

    print('Testing MLTrainingResult.to_dict()...')
    try:
        result_dict = ml_result.to_dict()
        print('✓ to_dict() conversion successful')
        
        json_str = json.dumps(result_dict)
        print('✓ JSON serialization successful')
        print(f'JSON length: {len(json_str)} characters')
        
        # Verify some key data is preserved
        parsed = json.loads(json_str)
        assert parsed['problem_type'] == 'regression'
        assert parsed['target_variable'] == 'exam_score'
        assert parsed['pipeline_run_id'] == 'test_run_123'
        assert len(parsed['training_results']) == 1
        assert len(parsed['evaluation_results']) == 1
        
        print('✓ All JSON serialization tests passed!')
        print('✓ ML training results can now be stored in database!')
        
    except Exception as e:
        print(f'✗ JSON serialization failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_ml_json_serialization()
    if success:
        print("\n🎉 JSON serialization issue is FIXED!")
        print("ML training results should now display properly in the frontend.")
    else:
        print("\n❌ JSON serialization issue still exists.") 