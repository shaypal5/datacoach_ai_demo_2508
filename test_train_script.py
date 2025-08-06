#!/usr/bin/env python3
"""
Simple test script for train_heart_models.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_help():
    """Test help command."""
    result = run_command(["python", "train_heart_models.py", "--help"])
    assert result.returncode == 0, "Help command failed"
    assert "Train heart disease prediction models" in result.stdout
    print("✓ Help command works")

def test_single_model():
    """Test training a single model."""
    # Clean up previous files
    for f in Path(".").glob("logreg_*"):
        f.unlink()
    
    result = run_command(["python", "train_heart_models.py", "-m", "logreg"])
    assert result.returncode == 0, f"Single model training failed: {result.stderr}"
    
    # Check files were created
    assert Path("logreg_model.pkl").exists(), "Model file not created"
    assert Path("logreg_metrics.txt").exists(), "Metrics file not created"
    assert Path("logreg_predictions.txt").exists(), "Predictions file not created"
    print("✓ Single model training works")

def test_multiple_models():
    """Test training multiple models."""
    # Clean up previous files
    for pattern in ["xgb_*", "svm_*"]:
        for f in Path(".").glob(pattern):
            f.unlink()
    
    result = run_command(["python", "train_heart_models.py", "-m", "xgb", "-m", "svm"])
    assert result.returncode == 0, f"Multiple model training failed: {result.stderr}"
    
    # Check files were created for both models
    for model in ["xgb", "svm"]:
        assert Path(f"{model}_model.pkl").exists(), f"{model} model file not created"
        assert Path(f"{model}_metrics.txt").exists(), f"{model} metrics file not created"
        assert Path(f"{model}_predictions.txt").exists(), f"{model} predictions file not created"
    
    print("✓ Multiple model training works")

def test_invalid_model():
    """Test error handling for invalid model."""
    result = run_command(["python", "train_heart_models.py", "-m", "invalid"])
    assert result.returncode != 0, "Should fail with invalid model"
    assert "invalid choice" in result.stderr
    print("✓ Invalid model handling works")

def test_missing_data():
    """Test error handling for missing data file."""
    result = run_command(["python", "train_heart_models.py", "-m", "logreg", "--data", "missing.csv"])
    assert result.returncode != 0, "Should fail with missing data file"
    print("✓ Missing data file handling works")

def main():
    """Run all tests."""
    print("Testing train_heart_models.py...")
    
    # Check if heart.csv exists
    if not Path("heart.csv").exists():
        print("Error: heart.csv not found in current directory")
        sys.exit(1)
    
    try:
        test_help()
        test_single_model()
        test_multiple_models()
        test_invalid_model()
        test_missing_data()
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()