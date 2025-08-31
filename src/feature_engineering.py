"""
Feature Engineering Module for Chronic Kidney Disease Prediction

This module handles feature creation, selection, and engineering tasks including:
- Clinical feature combinations
- Statistical feature extraction
- Feature selection using various methods
- Dimensionality reduction
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinically relevant derived features based on medical knowledge.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with additional clinical features
    """
    df_features = df.copy()
    
    print("Creating clinical features...")
    
    # Kidney function indicators
    if 'bu' in df_features.columns and 'sc' in df_features.columns:
        # Blood Urea Nitrogen to Creatinine Ratio (BUN/Cr)
        df_features['bun_cr_ratio'] = df_features['bu'] / (df_features['sc'] + 1e-8)
        
        # Estimated Glomerular Filtration Rate (eGFR) - simplified version
        # Using MDRD formula components
        if 'age' in df_features.columns:
            df_features['egfr_estimate'] = 175 * (df_features['sc'] ** -1.154) * (df_features['age'] ** -0.203)
    
    # Anemia indicators
    if 'hemo' in df_features.columns and 'pcv' in df_features.columns:
        # Hemoglobin to Packed Cell Volume ratio
        df_features['hemo_pcv_ratio'] = df_features['hemo'] / (df_features['pcv'] + 1e-8)
        
        # Anemia severity (normal Hgb: 12-16 g/dL for females, 14-18 g/dL for males)
        df_features['anemia_severity'] = np.where(df_features['hemo'] < 12, 1, 0)
    
    # Blood pressure categories
    if 'bp' in df_features.columns:
        # Blood pressure categories
        df_features['bp_category'] = pd.cut(df_features['bp'], 
                                          bins=[0, 120, 140, 160, 300], 
                                          labels=[0, 1, 2, 3], 
                                          include_lowest=True).astype(int)
    
    # Age groups
    if 'age' in df_features.columns:
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 18, 45, 65, 100], 
                                        labels=[0, 1, 2, 3], 
                                        include_lowest=True).astype(int)
    
    # Specific gravity categories
    if 'sg' in df_features.columns:
        df_features['sg_category'] = pd.cut(df_features['sg'], 
                                          bins=[1.000, 1.010, 1.020, 1.030, 1.050], 
                                          labels=[0, 1, 2, 3], 
                                          include_lowest=True).astype(int)
    
    # Electrolyte balance
    if 'sod' in df_features.columns and 'pot' in df_features.columns:
        # Sodium to Potassium ratio
        df_features['na_k_ratio'] = df_features['sod'] / (df_features['pot'] + 1e-8)
        
        # Electrolyte imbalance indicator
        df_features['electrolyte_imbalance'] = np.where(
            (df_features['sod'] < 135) | (df_features['sod'] > 145) |
            (df_features['pot'] < 3.5) | (df_features['pot'] > 5.0), 1, 0
        )
    
    # Blood cell ratios
    if 'wc' in df_features.columns and 'rc' in df_features.columns:
        # White to Red blood cell ratio
        df_features['wc_rc_ratio'] = df_features['wc'] / (df_features['rc'] + 1e-8)
    
    # Risk factor combinations
    if all(col in df_features.columns for col in ['htn', 'dm', 'cad']):
        # Cardiovascular risk score
        df_features['cv_risk_score'] = df_features['htn'] + df_features['dm'] + df_features['cad']
        
        # Metabolic syndrome indicator
        df_features['metabolic_syndrome'] = np.where(
            (df_features['htn'] == 1) & (df_features['dm'] == 1), 1, 0
        )
    
    print(f"Created {len(df_features.columns) - len(df.columns)} new clinical features")
    return df_features


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical features using rolling windows and aggregations.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with statistical features
    """
    df_stats = df.copy()
    
    print("Creating statistical features...")
    
    # Identify numerical columns (excluding target and id)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'classification']]
    
    # Create polynomial features for important numerical variables
    if len(numerical_cols) > 0:
        # Select top 5 most important features for polynomial creation
        top_features = numerical_cols[:5]
        
        for i, col1 in enumerate(top_features):
            for j, col2 in enumerate(top_features[i+1:], i+1):
                # Interaction terms
                df_stats[f'{col1}_{col2}_interaction'] = df_stats[col1] * df_stats[col2]
                
                # Ratio terms (avoid division by zero)
                df_stats[f'{col1}_{col2}_ratio'] = df_stats[col1] / (df_stats[col2] + 1e-8)
    
    # Create rolling statistics for time-series like features
    for col in numerical_cols[:3]:  # Limit to first 3 columns to avoid explosion
        # Rolling mean and std (simulated)
        df_stats[f'{col}_rolling_mean'] = df_stats[col].rolling(window=3, min_periods=1).mean()
        df_stats[f'{col}_rolling_std'] = df_stats[col].rolling(window=3, min_periods=1).std()
    
    # Create percentile-based features
    for col in numerical_cols[:3]:
        df_stats[f'{col}_percentile'] = df_stats[col].rank(pct=True)
    
    print(f"Created {len(df_stats.columns) - len(df.columns)} new statistical features")
    return df_stats


def select_features_univariate(df: pd.DataFrame, target_col: str = 'classification', 
                             k: int = 20) -> pd.DataFrame:
    """
    Select features using univariate statistical tests.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        k (int): Number of top features to select
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    print(f"Performing univariate feature selection (top {k} features)...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Apply univariate feature selection
    selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create new dataframe with selected features
    df_selected = df[selected_features + [target_col]].copy()
    
    print(f"Selected {len(selected_features)} features using univariate selection")
    return df_selected, selected_features


def select_features_recursive(df: pd.DataFrame, target_col: str = 'classification',
                            estimator=None, n_features: int = 15) -> pd.DataFrame:
    """
    Select features using Recursive Feature Elimination (RFE).
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        estimator: Base estimator for RFE
        n_features (int): Number of features to select
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    print(f"Performing recursive feature elimination (top {n_features} features)...")
    
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Apply RFE
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, len(feature_cols)))
    X_selected = rfe.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[rfe.support_].tolist()
    
    # Create new dataframe with selected features
    df_selected = df[selected_features + [target_col]].copy()
    
    print(f"Selected {len(selected_features)} features using RFE")
    return df_selected, selected_features


def select_features_importance(df: pd.DataFrame, target_col: str = 'classification',
                             threshold: str = 'median') -> pd.DataFrame:
    """
    Select features based on importance scores from tree-based models.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        threshold (str): Threshold for feature selection
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    print("Performing feature selection based on importance scores...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select features based on threshold
    if threshold == 'median':
        threshold_value = feature_importance['importance'].median()
    elif threshold == 'mean':
        threshold_value = feature_importance['importance'].mean()
    else:
        threshold_value = float(threshold)
    
    selected_features = feature_importance[feature_importance['importance'] >= threshold_value]['feature'].tolist()
    
    # Create new dataframe with selected features
    df_selected = df[selected_features + [target_col]].copy()
    
    print(f"Selected {len(selected_features)} features using importance-based selection")
    return df_selected, selected_features, feature_importance


def apply_pca(df: pd.DataFrame, target_col: str = 'classification', 
              n_components: float = 0.95) -> pd.DataFrame:
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        n_components (float): Number of components or explained variance ratio
        
    Returns:
        pd.DataFrame: Dataset with PCA components
    """
    print("Applying Principal Component Analysis...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'id']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create new dataframe with PCA components
    pca_cols = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca[target_col] = y.values
    
    # Print explained variance information
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"PCA completed: {len(pca_cols)} components explain {cumulative_variance[-1]:.3f} of variance")
    print(f"Top 5 components explain: {sum(explained_variance[:5]):.3f} of variance")
    
    return df_pca, pca


def create_features(df: pd.DataFrame, target_col: str = 'classification') -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    print("Starting feature engineering pipeline...")
    
    # Create clinical features
    df_features = create_clinical_features(df)
    
    # Create statistical features
    df_features = create_statistical_features(df_features)
    
    # Remove any infinite or NaN values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())
    
    print(f"Feature engineering completed. Final shape: {df_features.shape}")
    return df_features


def select_features(df: pd.DataFrame, target_col: str = 'classification',
                   method: str = 'importance', **kwargs) -> tuple:
    """
    Feature selection using specified method.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target variable column name
        method (str): Feature selection method
        **kwargs: Additional arguments for specific methods
        
    Returns:
        tuple: (selected_dataset, selected_features, additional_info)
    """
    print(f"Performing feature selection using {method} method...")
    
    if method == 'univariate':
        k = kwargs.get('k', 20)
        return select_features_univariate(df, target_col, k)
    
    elif method == 'rfe':
        n_features = kwargs.get('n_features', 15)
        estimator = kwargs.get('estimator', None)
        return select_features_recursive(df, target_col, estimator, n_features)
    
    elif method == 'importance':
        threshold = kwargs.get('threshold', 'median')
        return select_features_importance(df, target_col, threshold)
    
    elif method == 'pca':
        n_components = kwargs.get('n_components', 0.95)
        return apply_pca(df, target_col, n_components)
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_data
    
    # Load and preprocess data
    df, _ = preprocess_data('data/kidney_disease.csv')
    
    # Create features
    df_features = create_features(df)
    
    # Select features
    df_selected, selected_features = select_features(df_features, method='importance')
    
    print("Feature engineering completed!")
