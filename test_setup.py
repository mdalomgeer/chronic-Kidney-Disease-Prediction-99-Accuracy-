#!/usr/bin/env python3
"""
Test Setup Script for Chronic Kidney Disease Prediction Project

This script tests the basic setup and functionality of the project.
Run this to verify that everything is working correctly.

Author: Md Alomgeer Hussein
Institution: University of Maryland Baltimore County
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import scikit-learn: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import seaborn: {e}")
        return False
    
    return True


def test_project_structure():
    """Test if the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'results',
        'results/models',
        'results/plots',
        'results/reports'
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'LICENSE',
        'CONTRIBUTING.md',
        'PROJECT_DOCUMENTATION.md',
        'train_model.py',
        'src/__init__.py',
        'src/preprocessing.py',
        'src/feature_engineering.py',
        'src/models.py',
        'src/evaluation.py'
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            return False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False
    
    return True


def test_data_files():
    """Test if data files are accessible."""
    print("\nTesting data files...")
    
    data_files = [
        'data/kidney_disease.csv',
        'data/kidney_disease_new.csv',
        'data/kidney_disease_PCA.csv'
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / 1024  # KB
            print(f"✓ Data file exists: {file_path} ({file_size:.1f} KB)")
        else:
            print(f"✗ Missing data file: {file_path}")
            return False
    
    return True


def test_source_modules():
    """Test if source modules can be imported."""
    print("\nTesting source modules...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / 'src'))
    
    try:
        from preprocessing import preprocess_data
        print("✓ Preprocessing module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import preprocessing module: {e}")
        return False
    
    try:
        from feature_engineering import create_features
        print("✓ Feature engineering module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import feature engineering module: {e}")
        return False
    
    try:
        from models import CKDEnsembleModel
        print("✓ Models module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models module: {e}")
        return False
    
    try:
        from evaluation import plot_results
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation module: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of the modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Test data loading
        import pandas as pd
        df = pd.read_csv('data/kidney_disease.csv')
        print(f"✓ Data loaded successfully: {df.shape}")
        
        # Test basic preprocessing
        from preprocessing import load_data, clean_data
        df_loaded = load_data('data/kidney_disease.csv')
        df_cleaned = clean_data(df_loaded)
        print(f"✓ Basic preprocessing completed: {df_cleaned.shape}")
        
        # Test feature engineering
        from feature_engineering import create_clinical_features
        df_features = create_clinical_features(df_cleaned)
        print(f"✓ Feature engineering completed: {df_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Main test function."""
    print("="*60)
    print("CHRONIC KIDNEY DISEASE PREDICTION - SETUP TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Files", test_data_files),
        ("Source Modules", test_source_modules),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your project is ready to use.")
        print("\nNext steps:")
        print("1. Activate your virtual environment")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Run the training script: python train_model.py")
        print("4. Explore the Jupyter notebooks in the notebooks/ directory")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure all required packages are installed")
        print("2. Check that all files are in the correct locations")
        print("3. Verify your Python environment is set up correctly")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
