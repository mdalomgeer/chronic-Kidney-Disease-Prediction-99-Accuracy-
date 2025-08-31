"""
Data Preprocessing Module for Chronic Kidney Disease Prediction

This module handles all data preprocessing tasks including:
- Missing value imputation
- Categorical encoding
- Data normalization
- Outlier detection and treatment
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the kidney disease dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates and handling data quality issues.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Clean target variable
    df['classification'] = df['classification'].str.strip()
    df = df[df['classification'].isin(['ckd', 'notckd'])]
    
    # Remove rows with missing target
    df = df.dropna(subset=['classification'])
    
    print(f"Data cleaning completed. Final shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'iterative') -> pd.DataFrame:
    """
    Handle missing values using advanced imputation strategies.
    
    Args:
        df (pd.DataFrame): Input dataset
        strategy (str): Imputation strategy ('simple', 'knn', 'iterative')
        
    Returns:
        pd.DataFrame: Dataset with imputed values
    """
    df_imputed = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'id' and 'classification' from numerical columns
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if 'classification' in categorical_cols:
        categorical_cols.remove('classification')
    
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Handle numerical missing values
    if strategy == 'simple':
        # Simple imputation with median for numerical columns
        imputer = SimpleImputer(strategy='median')
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
        
    elif strategy == 'knn':
        # KNN imputation for numerical columns
        imputer = KNNImputer(n_neighbors=5)
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
        
    elif strategy == 'iterative':
        # Iterative imputation for numerical columns
        imputer = IterativeImputer(random_state=42, max_iter=10)
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    
    # Handle categorical missing values
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            # Fill with mode (most frequent value)
            mode_value = df_imputed[col].mode()[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
    
    print(f"Missing value imputation completed using {strategy} strategy")
    return df_imputed


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using label encoding.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with encoded categorical variables
    """
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding target)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'classification' in categorical_cols:
        categorical_cols.remove('classification')
    
    # Apply label encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df_encoded['classification'] = target_encoder.fit_transform(df_encoded['classification'])
    
    print(f"Categorical encoding completed for {len(categorical_cols)} columns")
    return df_encoded, label_encoders, target_encoder


def normalize_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Normalize numerical features using standardization.
    
    Args:
        df (pd.DataFrame): Input dataset
        exclude_cols (list): Columns to exclude from normalization
        
    Returns:
        pd.DataFrame: Dataset with normalized features
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'classification']
    
    df_normalized = df.copy()
    
    # Identify numerical columns for normalization
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Apply standardization
    scaler = StandardScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])
    
    print(f"Feature normalization completed for {len(numerical_cols)} columns")
    return df_normalized, scaler


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers in numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        method (str): Outlier detection method ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    df_clean = df.copy()
    
    # Identify numerical columns (excluding target and id)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'classification']]
    
    outliers_removed = 0
    
    for col in numerical_cols:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = z_scores > threshold
        
        # Cap outliers instead of removing them
        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        outliers_removed += outliers.sum()
    
    print(f"Outlier handling completed. {outliers_removed} outliers capped")
    return df_clean


def preprocess_data(file_path: str, 
                   missing_strategy: str = 'iterative',
                   normalize: bool = True,
                   handle_outliers: bool = True) -> tuple:
    """
    Complete data preprocessing pipeline.
    
    Args:
        file_path (str): Path to the dataset
        missing_strategy (str): Strategy for handling missing values
        normalize (bool): Whether to normalize features
        handle_outliers (bool): Whether to handle outliers
        
    Returns:
        tuple: (processed_dataframe, preprocessing_objects)
    """
    print("Starting data preprocessing pipeline...")
    
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Encode categorical variables
    df, label_encoders, target_encoder = encode_categorical(df)
    
    # Handle outliers
    if handle_outliers:
        df = detect_outliers(df)
    
    # Normalize features
    scaler = None
    if normalize:
        df, scaler = normalize_features(df)
    
    # Create preprocessing objects dictionary
    preprocessing_objects = {
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'scaler': scaler
    }
    
    print("Data preprocessing pipeline completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    
    return df, preprocessing_objects


def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save preprocessed dataset to CSV file.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        output_path (str): Output file path
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving preprocessed data: {str(e)}")


if __name__ == "__main__":
    # Example usage
    df, preprocess_objects = preprocess_data('data/kidney_disease.csv')
    print("Preprocessing completed!")
