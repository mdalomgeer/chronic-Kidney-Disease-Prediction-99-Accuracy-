"""
Chronic Kidney Disease Prediction Package

A comprehensive machine learning solution for predicting chronic kidney disease
with 99% accuracy using advanced data preprocessing and ensemble learning techniques.

Author: [Your Name]
Institution: [Your University]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@university.edu]"

from .preprocessing import preprocess_data, handle_missing_values, encode_categorical
from .feature_engineering import create_features, select_features
from .models import train_ensemble_model, predict_ckd
from .evaluation import evaluate_model, plot_results

__all__ = [
    "preprocess_data",
    "handle_missing_values", 
    "encode_categorical",
    "create_features",
    "select_features",
    "train_ensemble_model",
    "predict_ckd",
    "evaluate_model",
    "plot_results",
]
