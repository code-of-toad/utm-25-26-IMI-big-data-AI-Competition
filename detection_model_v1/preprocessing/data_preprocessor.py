"""
Data Preprocessing for Anomaly Detection

Prepares features for Isolation Forest by:
1. Selecting relevant features
2. Handling missing values
3. Scaling/normalizing features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


def select_features_for_anomaly_detection(features_df):
    """
    Select features suitable for anomaly detection.
    Excludes customer_id and rule-based features that are already used in rule-based scoring.
    
    Returns DataFrame with selected features.
    """
    # Start with all numeric features
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove customer_id if it's numeric (shouldn't be, but just in case)
    if 'customer_id' in numeric_cols:
        numeric_cols.remove('customer_id')
    
    # Exclude rule-based derived features to avoid double-counting
    # Keep raw/aggregated features that Isolation Forest can learn patterns from
    exclude_patterns = [
        'structuring_risk', 'channel_risk', 'geographic_risk', 
        'behavioral_risk', 'profile_risk', 'rule_based_score'
    ]
    
    selected_features = [col for col in numeric_cols 
                       if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    return features_df[selected_features]


def preprocess_features(features_df, scaler=None, fit_scaler=True):
    """
    Preprocess features for anomaly detection.
    
    Args:
        features_df: DataFrame with features
        scaler: Pre-fitted scaler (if None, will create new one)
        fit_scaler: Whether to fit the scaler (True for training, False for prediction)
    
    Returns:
        preprocessed_df: Preprocessed features
        scaler: Fitted scaler
    """
    # Select features
    feature_data = select_features_for_anomaly_detection(features_df)
    
    # Handle missing values - fill with median for numeric features
    feature_data = feature_data.fillna(feature_data.median())
    
    # Replace inf/-inf with NaN, then fill
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.fillna(feature_data.median())
    
    # Initialize scaler if needed
    if scaler is None:
        scaler = StandardScaler()
    
    # Fit and transform
    if fit_scaler:
        scaled_data = scaler.fit_transform(feature_data)
    else:
        scaled_data = scaler.transform(feature_data)
    
    # Convert back to DataFrame
    preprocessed_df = pd.DataFrame(
        scaled_data,
        columns=feature_data.columns,
        index=feature_data.index
    )
    
    return preprocessed_df, scaler


def save_scaler(scaler, path):
    """Save scaler to disk for reuse"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    """Load scaler from disk"""
    with open(path, 'rb') as f:
        return pickle.load(f)
