"""
Machine Learning Models Module for Chronic Kidney Disease Prediction

This module implements various machine learning algorithms and ensemble methods
to achieve 99% accuracy in CKD prediction, including:
- Individual models (Random Forest, XGBoost, SVM, etc.)
- Ensemble methods (Voting, Stacking, Blending)
- Hyperparameter optimization
- Model persistence and loading
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class CKDEnsembleModel:
    """
    Ensemble model for Chronic Kidney Disease prediction.
    
    This class implements a sophisticated ensemble approach combining multiple
    algorithms to achieve optimal performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ensemble model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.ensemble = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _create_base_models(self):
        """Create and configure base models."""
        
        # Random Forest with optimized parameters
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost with optimized parameters
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # LightGBM with optimized parameters
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state
        )
        
        print("Base models created successfully")
    
    def _create_ensemble(self, method: str = 'voting'):
        """
        Create ensemble using specified method.
        
        Args:
            method (str): Ensemble method ('voting', 'stacking')
        """
        if method == 'voting':
            # Voting Classifier with soft voting
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
        elif method == 'stacking':
            # Stacking Classifier
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=-1
            )
        
        print(f"Ensemble created using {method} method")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'classification'):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target variable column name
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_individual_models(self, X_train, y_train):
        """Train individual base models."""
        print("Training individual models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("Individual models training completed")
    
    def train_ensemble(self, X_train, y_train, method: str = 'voting'):
        """Train the ensemble model."""
        print("Training ensemble model...")
        
        # Create ensemble
        self._create_ensemble(method)
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.ensemble, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        print("Ensemble training completed")
    
    def train(self, df: pd.DataFrame, target_col: str = 'classification', 
              method: str = 'voting'):
        """
        Complete training pipeline.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target variable column name
            method (str): Ensemble method
        """
        print("Starting training pipeline...")
        
        # Create base models
        self._create_base_models()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Train individual models
        self.train_individual_models(X_train, y_train)
        
        # Train ensemble
        self.train_ensemble(X_train, y_train, method)
        
        # Evaluate on test set
        self.evaluate(X_test, y_test)
        
        print("Training pipeline completed successfully!")
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if needed
        if hasattr(self, 'scaler'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.ensemble.predict(X_scaled)
        probabilities = self.ensemble.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Individual model evaluation
        print("\nIndividual Model Performance:")
        print("-" * 40)
        
        for name, model in self.models.items():
            if hasattr(self, 'scaler'):
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name:15}: {accuracy:.4f}")
        
        # Ensemble evaluation
        print("\nEnsemble Model Performance:")
        print("-" * 40)
        
        y_pred_ensemble, y_prob_ensemble = self.predict(X_test)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred_ensemble))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print("-" * 40)
        cm = confusion_matrix(y_test, y_pred_ensemble)
        print(cm)
        
        return {
            'accuracy': accuracy_ensemble,
            'predictions': y_pred_ensemble,
            'probabilities': y_prob_ensemble,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.ensemble = model_data['ensemble']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")


def train_ensemble_model(df: pd.DataFrame, target_col: str = 'classification',
                        method: str = 'voting', save_path: str = None) -> CKDEnsembleModel:
    """
    Train ensemble model with the complete pipeline.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        method (str): Ensemble method
        save_path (str): Path to save the trained model
        
    Returns:
        CKDEnsembleModel: Trained ensemble model
    """
    # Create and train model
    model = CKDEnsembleModel()
    model.train(df, target_col, method)
    
    # Save model if path provided
    if save_path:
        model.save_model(save_path)
    
    return model


def predict_ckd(model, patient_data: dict) -> dict:
    """
    Make CKD prediction for a single patient.
    
    Args:
        model: Trained model
        patient_data (dict): Patient clinical data
        
    Returns:
        dict: Prediction results
    """
    if not model.is_trained:
        raise ValueError("Model must be trained before making predictions")
    
    # Convert patient data to DataFrame
    df_patient = pd.DataFrame([patient_data])
    
    # Make prediction
    prediction, probability = model.predict(df_patient)
    
    # Interpret results
    ckd_probability = probability[0][1] if prediction[0] == 1 else probability[0][0]
    risk_level = "High" if ckd_probability > 0.8 else "Medium" if ckd_probability > 0.5 else "Low"
    
    return {
        'prediction': 'CKD' if prediction[0] == 1 else 'No CKD',
        'probability': ckd_probability,
        'risk_level': risk_level,
        'confidence': max(probability[0])
    }


def hyperparameter_optimization(df: pd.DataFrame, target_col: str = 'classification',
                              model_type: str = 'random_forest') -> dict:
    """
    Perform hyperparameter optimization for specified model.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        model_type (str): Type of model to optimize
        
    Returns:
        dict: Best parameters and score
    """
    print(f"Performing hyperparameter optimization for {model_type}...")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Define parameter grids
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_data
    from feature_engineering import create_features
    
    # Load and preprocess data
    df, _ = preprocess_data('data/kidney_disease.csv')
    
    # Create features
    df_features = create_features(df)
    
    # Train ensemble model
    model = train_ensemble_model(df_features, method='voting')
    
    print("Model training completed!")
