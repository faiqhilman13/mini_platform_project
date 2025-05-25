#!/usr/bin/env python3
"""
Test script to verify ML training with JSON serialization fix
"""
import requests
import json
import time

def test_ml_training():
    """Test ML training pipeline with the JSON serialization fix"""
    
    # First, we need to get the uploaded file ID
    # Use file ID 4 (student_habits_performance.csv)
    file_id = 4
    
    # Test 1: Use the new ML-specific endpoint
    ml_request = {
        "uploaded_file_log_id": file_id,
        "target_column": "exam_score",
        "problem_type": "regression",
        "algorithms": [
            {
                "name": "linear_regression",
                "hyperparameters": {}
            },
            {
                "name": "decision_tree_regressor", 
                "hyperparameters": {
                    "max_depth": 10
                }
            }
        ],
        "preprocessing_config": {
            "test_size": 0.2,
            "scaling_strategy": "standard"
        }
    }
    
    print("🚀 Testing NEW ML-specific endpoint...")
    
    try:
        # Test the new ML endpoint
        response = requests.post(
            "http://localhost:8001/api/v1/pipelines/ml/trigger",
            json=ml_request
        )
        
        print(f"✅ ML endpoint request sent. Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ ML endpoint failed: {response.text}")
            return False
            
        result = response.json()
        print(f"✅ ML endpoint response: {result}")
        
        if "run_uuid" in result:
            run_uuid = result["run_uuid"]
            print(f"🔄 Training started with run_uuid: {run_uuid}")
            
            # Check status using the ML status endpoint
            print("⏳ Checking training status via ML endpoint...")
            time.sleep(5)  # Give it some time
            
            status_response = requests.get(f"http://localhost:8001/api/v1/pipelines/ml/status/{run_uuid}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 ML endpoint status: {status_data.get('status', 'unknown')}")
                
                if status_data.get("status") == "COMPLETED":
                    print("🎉 ML endpoint training completed successfully!")
                    
                    # Try to get models via ML endpoint
                    models_response = requests.get(f"http://localhost:8001/api/v1/pipelines/ml/models/{run_uuid}")
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        print(f"🤖 ML endpoint found {len(models_data)} trained models:")
                        
                        for model in models_data:
                            algorithm = model.get('algorithm_name', 'unknown')
                            metrics = model.get('performance_metrics', {})
                            if 'r2' in metrics:
                                r2_score = metrics['r2']
                                print(f"   📈 {algorithm}: R² Score = {r2_score:.4f}")
                            else:
                                print(f"   📈 {algorithm}: Metrics = {metrics}")
                        
                        return True
                    else:
                        print(f"❌ Failed to get models: {models_response.status_code}")
                        return False
                        
                elif status_data.get("status") == "FAILED":
                    error_msg = status_data.get("error_message", "Unknown error")
                    print(f"❌ ML endpoint training failed: {error_msg}")
                    
                    if "JSON serializable" in error_msg:
                        print("💥 JSON SERIALIZATION ISSUE STILL EXISTS!")
                        return False
                    else:
                        print("⚠️ Training failed for other reasons")
                        return False
                        
                else:
                    print(f"⏳ Training still in progress: {status_data.get('status')}")
                    return None  # Still running
            else:
                print(f"❌ Failed to get ML endpoint status: {status_response.status_code} - {status_response.text}")
                return False
                    
        else:
            print(f"❌ Failed to start ML endpoint training: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ML endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ML Training with JSON Serialization Fix")
    print("🔗 Testing NEW ML endpoints using existing pipeline infrastructure")
    print("=" * 70)
    
    result = test_ml_training()
    
    if result is True:
        print("\n🎉 COMPLETE SUCCESS!")
        print("✅ JSON serialization is FIXED!")
        print("✅ ML training completed and results stored properly")
        print("✅ NEW ML endpoints working perfectly!")
        print("✅ Frontend will now be able to display ML results!")
    elif result is False:
        print("\n❌ FAILED: Issues detected") 
        print("🔧 Need to investigate further")
    else:
        print("\n⏳ Training is still running, check again later")
        
    print("=" * 70) 