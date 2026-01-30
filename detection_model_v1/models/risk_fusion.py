"""
Risk Score Fusion

Combines rule-based scores with unsupervised anomaly scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def fuse_risk_scores(rule_scores_df, anomaly_scores_df, rule_weight=0.7, anomaly_weight=0.3):
    """
    Fuse rule-based and anomaly scores into final risk score.
    
    Args:
        rule_scores_df: DataFrame with customer_id and rule_based_score
        anomaly_scores_df: DataFrame with customer_id and score
        rule_weight: Weight for rule-based score (default 0.7)
        anomaly_weight: Weight for anomaly score (default 0.3)
    
    Returns:
        fused_df: DataFrame with customer_id, risk_score, rule_based_score, anomaly_score
    """
    # Validate weights sum to 1.0
    if abs(rule_weight + anomaly_weight - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {rule_weight} + {anomaly_weight} = {rule_weight + anomaly_weight}")
    
    # Merge on customer_id
    merged = pd.merge(
        rule_scores_df[['customer_id', 'rule_based_score']],
        anomaly_scores_df[['customer_id', 'score']],
        on='customer_id',
        how='inner'
    )
    
    # Rename anomaly score column for clarity
    merged = merged.rename(columns={'score': 'anomaly_score'})
    
    # Calculate fused risk score
    merged['risk_score'] = (
        rule_weight * merged['rule_based_score'] + 
        anomaly_weight * merged['anomaly_score']
    )
    
    # Clip to [0, 1] range
    merged['risk_score'] = merged['risk_score'].clip(0.0, 1.0)
    
    # Round to 4 decimal places
    merged['risk_score'] = merged['risk_score'].round(4)
    merged['rule_based_score'] = merged['rule_based_score'].round(4)
    merged['anomaly_score'] = merged['anomaly_score'].round(4)
    
    return merged


def generate_predictions(fused_df, threshold=None, top_percentile=5):
    """
    Generate binary predictions from risk scores.
    
    Args:
        fused_df: DataFrame with risk_score
        threshold: Fixed threshold (if None, uses percentile)
        top_percentile: Top N% to flag (default 5%)
    
    Returns:
        predictions_df: DataFrame with customer_id, predicted_label, risk_score
    """
    predictions_df = fused_df[['customer_id', 'risk_score']].copy()
    
    # Determine threshold
    if threshold is None:
        # Use percentile-based threshold
        threshold = np.percentile(fused_df['risk_score'], 100 - top_percentile)
    
    # Generate predictions
    predictions_df['predicted_label'] = (fused_df['risk_score'] >= threshold).astype(int)
    
    # Reorder columns
    predictions_df = predictions_df[['customer_id', 'predicted_label', 'risk_score']]
    
    return predictions_df, threshold


def main(rule_scores_path=None, anomaly_scores_path=None, output_path=None, 
         rule_weight=0.7, anomaly_weight=0.3, threshold=None, top_percentile=5):
    """
    Main function to fuse scores and generate predictions.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    rule_scores_path = Path(rule_scores_path) if rule_scores_path else (
        BASE_DIR / 'data' / 'intermediate' / 'rule_based_scores.csv'
    )
    anomaly_scores_path = Path(anomaly_scores_path) if anomaly_scores_path else (
        BASE_DIR / 'data' / 'intermediate' / 'scores_isolation_forest.csv'
    )
    output_path = Path(output_path) if output_path else (
        BASE_DIR / 'data' / 'output' / 'model_output.csv'
    )
    
    if not rule_scores_path.exists():
        raise FileNotFoundError(f"Rule scores not found: {rule_scores_path}")
    if not anomaly_scores_path.exists():
        raise FileNotFoundError(f"Anomaly scores not found: {anomaly_scores_path}")
    
    print("Loading scores...")
    rule_scores_df = pd.read_csv(rule_scores_path)
    anomaly_scores_df = pd.read_csv(anomaly_scores_path)
    print(f"  Rule-based: {len(rule_scores_df):,} customers")
    print(f"  Anomaly: {len(anomaly_scores_df):,} customers")
    
    print("Fusing risk scores...")
    fused_df = fuse_risk_scores(rule_scores_df, anomaly_scores_df, rule_weight, anomaly_weight)
    print(f"  Fused: {len(fused_df):,} customers")
    print(f"  Risk score range: {fused_df['risk_score'].min():.4f} .. {fused_df['risk_score'].max():.4f}")
    print(f"  Mean risk score: {fused_df['risk_score'].mean():.4f}")
    
    print("Generating predictions...")
    predictions_df, used_threshold = generate_predictions(fused_df, threshold, top_percentile)
    flagged_count = predictions_df['predicted_label'].sum()
    print(f"  Threshold: {used_threshold:.4f}")
    print(f"  Flagged: {flagged_count:,} customers ({100*flagged_count/len(predictions_df):.2f}%)")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    
    return predictions_df, fused_df


if __name__ == '__main__':
    main()
