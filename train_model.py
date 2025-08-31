#!/usr/bin/env python3
"""
Main Training Script for Chronic Kidney Disease Prediction

This script demonstrates the complete machine learning pipeline including:
- Data preprocessing
- Feature engineering
- Model training
- Evaluation and visualization

Author: [Your Name]
Institution: [Your University]
License: MIT
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import preprocess_data
from feature_engineering import create_features, select_features
from models import train_ensemble_model
from evaluation import plot_results, generate_evaluation_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def main():
    """Main training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CKD prediction model')
    parser.add_argument('--data_path', type=str, default='data/kidney_disease.csv',
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--ensemble_method', type=str, default='voting',
                       choices=['voting', 'stacking'],
                       help='Ensemble method to use')
    parser.add_argument('--feature_selection', type=str, default='importance',
                       choices=['importance', 'rfe', 'univariate', 'pca'],
                       help='Feature selection method')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create and save visualization plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHRONIC KIDNEY DISEASE PREDICTION - TRAINING PIPELINE")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Ensemble method: {args.ensemble_method}")
    print(f"Feature selection: {args.feature_selection}")
    print("="*60)
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'reports').mkdir(exist_ok=True)
    
    try:
        # Step 1: Data Preprocessing
        print("\n1. DATA PREPROCESSING")
        print("-" * 30)
        df_processed, preprocess_objects = preprocess_data(
            args.data_path,
            missing_strategy='iterative',
            normalize=True,
            handle_outliers=True
        )
        
        print(f"Preprocessing completed. Final shape: {df_processed.shape}")
        
        # Save preprocessed data
        preprocessed_path = output_path / 'data' / 'kidney_disease_preprocessed.csv'
        preprocessed_path.parent.mkdir(exist_ok=True)
        df_processed.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to: {preprocessed_path}")
        
        # Step 2: Feature Engineering
        print("\n2. FEATURE ENGINEERING")
        print("-" * 30)
        df_features = create_features(df_processed)
        print(f"Feature engineering completed. Final shape: {df_features.shape}")
        
        # Save engineered features
        features_path = output_path / 'data' / 'kidney_disease_features.csv'
        df_features.to_csv(features_path, index=False)
        print(f"Engineered features saved to: {features_path}")
        
        # Step 3: Feature Selection
        print("\n3. FEATURE SELECTION")
        print("-" * 30)
        
        if args.feature_selection == 'pca':
            df_selected, pca_object = select_features(
                df_features, method='pca', n_components=0.95
            )
            # Save PCA data
            pca_path = output_path / 'data' / 'kidney_disease_pca.csv'
            df_selected.to_csv(pca_path, index=False)
            print(f"PCA data saved to: {pca_path}")
        else:
            df_selected, selected_features, feature_importance = select_features(
                df_features, method=args.feature_selection
            )
            print(f"Selected {len(selected_features)} features")
            
            # Save feature importance
            if feature_importance is not None:
                importance_path = output_path / 'reports' / 'feature_importance.csv'
                feature_importance.to_csv(importance_path, index=False)
                print(f"Feature importance saved to: {importance_path}")
        
        # Step 4: Model Training
        print("\n4. MODEL TRAINING")
        print("-" * 30)
        
        # Save model path
        model_path = output_path / 'models' / 'ckd_ensemble_model.pkl'
        
        # Train ensemble model
        model = train_ensemble_model(
            df_selected,
            target_col='classification',
            method=args.ensemble_method,
            save_path=str(model_path) if args.save_model else None
        )
        
        print(f"Model training completed successfully!")
        
        # Step 5: Model Evaluation
        print("\n5. MODEL EVALUATION")
        print("-" * 30)
        
        # Get test data from model
        X_test = model.scaler.transform(
            df_selected.drop(['classification', 'id'] if 'id' in df_selected.columns else ['classification'], axis=1)
        )
        y_test = df_selected['classification']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Generate evaluation report
        report_path = output_path / 'reports' / 'model_evaluation.txt'
        metrics = generate_evaluation_report(
            y_test, y_pred, y_prob, 
            model_name="CKD Ensemble Model",
            save_path=str(report_path)
        )
        
        print(f"Evaluation report saved to: {report_path}")
        
        # Step 6: Visualization
        if args.create_plots:
            print("\n6. CREATING VISUALIZATIONS")
            print("-" * 30)
            
            # Create comprehensive results plot
            results_plot_path = output_path / 'plots' / 'model_results.png'
            plot_results(
                y_test, y_pred, y_prob,
                model_name="CKD Ensemble Model",
                save_path=str(results_plot_path)
            )
            
            # Create feature importance plot
            if hasattr(model, 'get_feature_importance'):
                feature_importance_dict = model.get_feature_importance()
                if feature_importance_dict:
                    importance_plot_path = output_path / 'plots' / 'feature_importance.png'
                    from evaluation import plot_feature_importance
                    plot_feature_importance(
                        feature_importance_dict,
                        save_path=str(importance_plot_path)
                    )
        
        # Step 7: Summary
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final model accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {output_path}")
        print("="*60)
        
        # Save final metrics
        metrics_path = output_path / 'reports' / 'final_metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Final metrics saved to: {metrics_path}")
        
    except Exception as e:
        print(f"\nERROR: Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo_prediction():
    """Demonstrate model prediction on sample data."""
    
    print("\n" + "="*60)
    print("DEMO PREDICTION")
    print("="*60)
    
    # Sample patient data (normalized values)
    sample_patient = {
        'age': 55.0,
        'bp': 140.0,
        'sg': 1.020,
        'al': 2.0,
        'su': 0.0,
        'rbc': 1,  # encoded
        'pc': 1,   # encoded
        'pcc': 0,  # encoded
        'ba': 0,   # encoded
        'bgr': 120.0,
        'bu': 45.0,
        'sc': 1.8,
        'sod': 140.0,
        'pot': 4.5,
        'hemo': 12.5,
        'pcv': 38.0,
        'wc': 7500.0,
        'rc': 4.8,
        'htn': 1,  # encoded
        'dm': 1,   # encoded
        'cad': 0,  # encoded
        'appet': 1, # encoded
        'pe': 0,   # encoded
        'ane': 1   # encoded
    }
    
    print("Sample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    print("\nNote: This is a demonstration. In practice, you would:")
    print("1. Load a trained model")
    print("2. Preprocess new patient data")
    print("3. Make predictions using the model")
    print("4. Interpret results for clinical decision-making")


if __name__ == "__main__":
    # Run main training pipeline
    main()
    
    # Show demo prediction
    demo_prediction()
