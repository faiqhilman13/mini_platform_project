#!/usr/bin/env python3
"""
Test script to debug JSON serialization issues with numpy types
"""

import json
import numpy as np
from typing import Any, Dict

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

def test_numpy_serialization():
    """Test different numpy serialization scenarios"""
    print("=== Testing NumPy JSON Serialization ===")
    
    # Test basic numpy types
    test_cases = [
        ("numpy int64", np.int64(42)),
        ("numpy float64", np.float64(3.14159)),
        ("numpy array", np.array([1, 2, 3, 4, 5])),
        ("numpy bool", np.bool_(True)),
        ("numpy complex", np.complex128(1+2j)),
    ]
    
    for name, value in test_cases:
        print(f"\n--- Testing {name} ---")
        print(f"Original: {value} (type: {type(value)})")
        
        # Test direct JSON serialization (should fail)
        try:
            json_str = json.dumps(value)
            print(f"Direct JSON: SUCCESS - {json_str}")
        except Exception as e:
            print(f"Direct JSON: FAILED - {e}")
        
        # Test with convert_numpy_types
        try:
            converted = convert_numpy_types(value)
            json_str = json.dumps(converted)
            print(f"Converted JSON: SUCCESS - {json_str}")
        except Exception as e:
            print(f"Converted JSON: FAILED - {e}")
    
    # Test complex nested structure (like ML results)
    print(f"\n--- Testing Complex ML Result Structure ---")
    ml_result = {
        "algorithm_name": "linear_regression",
        "metrics": {
            "r2": np.float64(0.8965),
            "mse": np.float64(0.1234),
            "mae": np.float64(0.0987)
        },
        "feature_importance": {
            "feature1": np.float64(0.25),
            "feature2": np.float64(0.30),
            "feature3": np.float64(0.45)
        },
        "predictions": np.array([1.2, 2.3, 3.4, 4.5]),
        "training_time": np.float64(2.5),
        "hyperparameters": {
            "fit_intercept": np.bool_(True),
            "normalize": np.bool_(False)
        }
    }
    
    print(f"Original structure keys: {list(ml_result.keys())}")
    
    # Test direct serialization
    try:
        json_str = json.dumps(ml_result)
        print("Direct ML result JSON: SUCCESS")
    except Exception as e:
        print(f"Direct ML result JSON: FAILED - {e}")
    
    # Test with conversion
    try:
        converted_result = convert_numpy_types(ml_result)
        json_str = json.dumps(converted_result)
        print("Converted ML result JSON: SUCCESS")
        print(f"JSON length: {len(json_str)} characters")
    except Exception as e:
        print(f"Converted ML result JSON: FAILED - {e}")

if __name__ == "__main__":
    test_numpy_serialization() 