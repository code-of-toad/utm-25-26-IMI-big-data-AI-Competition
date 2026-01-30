"""
Isolation Forest Anomaly Detection

Trains Isolation Forest model and generates anomaly scores.
Outputs scores in 0-1 range where higher = more anomalous.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pickle
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_preprocessor import preprocess_features, save_scaler, load_scaler


def train_isolation_forest(features_df, contamination=0.05, random_state=42, n_estimators=100):
    """
    Train Isolation Forest on preprocessed features.
    
    Args:
        features_df: DataFrame with customer features
        contamination: Expected proportion of anomalies (default 0.05 = 5%)
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the forest
    
    Returns:
        detector: Trained Isolation Forest model
        scaler: Fitted scaler
        preprocessed_df: Preprocessed features used for training
    """
    # Preprocess features
    preprocessed_df, scaler = preprocess_features(features_df, fit_scaler=True)
    
    # Train Isolation Forest
    detector = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1  # Use all available cores
    )
    
    detector.fit(preprocessed_df)
    
    return detector, scaler, preprocessed_df


def predict_anomaly_scores(detector, scaler, features_df):
    """
    Generate anomaly scores for customers.
    
    Args:
        detector: Trained Isolation Forest model
        scaler: Fitted scaler
        features_df: DataFrame with customer features
    
    Returns:
        scores_df: DataFrame with customer_id and normalized score (0-1)
    """
    # Preprocess features using existing scaler
    preprocessed_df, _ = preprocess_features(features_df, scaler=scaler, fit_scaler=False)
    
    # Get anomaly scores (negative scores = more anomalous)
    anomaly_scores = detector.score_samples(preprocessed_df)
    
    # Isolation Forest returns negative scores for anomalies
    # Normalize to 0-1 range where 1 = most anomalous
    # Method: shift to positive, then normalize
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    
    if max_score == min_score:
        # All scores are the same
        normalized_scores = np.ones(len(anomaly_scores)) * 0.5
    else:
        # Normalize: (score - min) / (max - min)
        # But we want higher = more anomalous, so invert
        normalized_scores = 1.0 - ((anomaly_scores - min_score) / (max_score - min_score))
    
    # Create output DataFrame
    scores_df = pd.DataFrame({
        'customer_id': features_df['customer_id'].values,
        'score': normalized_scores
    })
    
    # Round to 4 decimal places
    scores_df['score'] = scores_df['score'].round(4)
    
    return scores_df


def save_model(detector, scaler, model_dir):
    """Save trained model and scaler"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detector
    with open(model_dir / 'isolation_forest.pkl', 'wb') as f:
        pickle.dump(detector, f)
    
    # Save scaler
    save_scaler(scaler, model_dir / 'scaler.pkl')


def load_model(model_dir):
    """Load trained model and scaler"""
    model_dir = Path(model_dir)
    
    # Load detector
    with open(model_dir / 'isolation_forest.pkl', 'rb') as f:
        detector = pickle.load(f)
    
    # Load scaler
    scaler = load_scaler(model_dir / 'scaler.pkl')
    
    return detector, scaler


def main(input_path=None, output_path=None, model_dir=None):
    """
    Main function to train Isolation Forest and generate scores.
    
    Args:
        input_path: Path to master_features.csv
        output_path: Path to save scores_isolation_forest.csv
        model_dir: Directory to save trained model
    """
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROJECT_ROOT = BASE_DIR.parent
    
    input_path = Path(input_path) if input_path else (
        PROJECT_ROOT / 'clean_data' / 'features' / 'final' / 'master_features.csv'
    )
    output_path = Path(output_path) if output_path else (
        BASE_DIR / 'data' / 'intermediate' / 'scores_isolation_forest.csv'
    )
    model_dir = Path(model_dir) if model_dir else (BASE_DIR / 'models' / 'saved')
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print("Loading features...")
    features_df = pd.read_csv(input_path)
    print(f"  Loaded {len(features_df):,} customers with {len(features_df.columns)} features")
    
    print("Training Isolation Forest...")
    detector, scaler, preprocessed_df = train_isolation_forest(features_df)
    print(f"  Trained on {len(preprocessed_df.columns)} features")
    
    print("Generating anomaly scores...")
    scores_df = predict_anomaly_scores(detector, scaler, features_df)
    print(f"  Score range: {scores_df['score'].min():.4f} .. {scores_df['score'].max():.4f}")
    print(f"  Mean score: {scores_df['score'].mean():.4f}")
    
    # Save scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    print(f"Saved scores to: {output_path}")
    
    # Save model
    save_model(detector, scaler, model_dir)
    print(f"Saved model to: {model_dir}")
    
    return scores_df


if __name__ == '__main__':
    main()
