"""
Test Fusion with Different Unsupervised Models

This script allows you to easily test your rule-based algorithm with different
unsupervised models (Isolation Forest, LOF, CBLOF, ABOD) and compare results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.risk_fusion import fuse_risk_scores, generate_predictions

BASE_DIR = Path(__file__).parent.parent
INTERMEDIATE_DIR = BASE_DIR / 'data' / 'intermediate'
OUTPUT_DIR = BASE_DIR / 'data' / 'output'

# Available unsupervised models
AVAILABLE_MODELS = {
    'isolation_forest': 'scores_isolation_forest.csv',
    'lof': 'scores_lof.csv',
    'cblof': 'scores_cblof.csv',
    'abod': 'scores_abod.csv'
}


def test_fusion(model_name, rule_weight=0.7, anomaly_weight=0.3, top_percentile=5, output_suffix=None):
    """
    Test fusion with a specific unsupervised model.
    
    Args:
        model_name: Name of the model ('isolation_forest', 'lof', 'cblof', 'abod')
        rule_weight: Weight for rule-based score (default 0.7)
        anomaly_weight: Weight for anomaly score (default 0.3)
        top_percentile: Top N% to flag (default 5)
        output_suffix: Suffix for output filename (if None, uses model_name)
    
    Returns:
        predictions_df: DataFrame with predictions
        stats: Dictionary with statistics
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    score_file = AVAILABLE_MODELS[model_name]
    score_path = INTERMEDIATE_DIR / score_file
    
    if not score_path.exists():
        print(f"⚠️  Warning: {score_file} not found. Skipping {model_name}.")
        return None, None
    
    print(f"\n{'='*70}")
    print(f"Testing Fusion: Rule-Based + {model_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    
    # Load scores
    print("Loading scores...")
    rule_scores_df = pd.read_csv(INTERMEDIATE_DIR / 'rule_based_scores.csv')
    anomaly_scores_df = pd.read_csv(score_path)
    
    print(f"  Rule-based: {len(rule_scores_df):,} customers")
    print(f"  {model_name}: {len(anomaly_scores_df):,} customers")
    
    # Fuse scores
    print(f"\nFusing scores (Rule: {rule_weight:.1%}, {model_name}: {anomaly_weight:.1%})...")
    fused_df = fuse_risk_scores(rule_scores_df, anomaly_scores_df, rule_weight, anomaly_weight)
    
    print(f"  Fused: {len(fused_df):,} customers")
    print(f"  Risk score range: {fused_df['risk_score'].min():.4f} .. {fused_df['risk_score'].max():.4f}")
    print(f"  Mean risk score: {fused_df['risk_score'].mean():.4f}")
    
    # Generate predictions
    print(f"\nGenerating predictions (top {top_percentile}%)...")
    predictions_df, threshold = generate_predictions(fused_df, threshold=None, top_percentile=top_percentile)
    flagged_count = predictions_df['predicted_label'].sum()
    
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Flagged: {flagged_count:,} customers ({100*flagged_count/len(predictions_df):.2f}%)")
    
    # Calculate statistics
    stats = {
        'model': model_name,
        'rule_weight': rule_weight,
        'anomaly_weight': anomaly_weight,
        'threshold': threshold,
        'flagged_count': flagged_count,
        'flag_rate': 100 * flagged_count / len(predictions_df),
        'mean_risk_score': fused_df['risk_score'].mean(),
        'mean_rule_score': fused_df['rule_based_score'].mean(),
        'mean_anomaly_score': fused_df['anomaly_score'].mean(),
        'correlation': rule_scores_df['rule_based_score'].corr(anomaly_scores_df['score'])
    }
    
    # Save output
    if output_suffix is None:
        output_suffix = model_name
    
    output_path = OUTPUT_DIR / f'model_output_{output_suffix}.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    
    return predictions_df, stats


def compare_all_models(rule_weight=0.7, anomaly_weight=0.3, top_percentile=5):
    """
    Test fusion with all available unsupervised models and compare results.
    """
    print("="*70)
    print("COMPARING ALL MODEL COMBINATIONS")
    print("="*70)
    
    results = []
    
    for model_name in AVAILABLE_MODELS.keys():
        predictions, stats = test_fusion(model_name, rule_weight, anomaly_weight, top_percentile)
        if stats:
            results.append(stats)
    
    if not results:
        print("\n❌ No models found. Make sure your colleague's model outputs are in:")
        print(f"   {INTERMEDIATE_DIR}")
        return
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    comparison_df = pd.DataFrame(results)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = OUTPUT_DIR / 'fusion_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Comparison saved: {comparison_path}")
    
    return comparison_df


def main():
    """Main function - test all available models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fusion with different unsupervised models')
    parser.add_argument('--model', choices=list(AVAILABLE_MODELS.keys()), 
                       help='Test specific model (default: test all)')
    parser.add_argument('--rule-weight', type=float, default=0.7,
                       help='Weight for rule-based score (default: 0.7)')
    parser.add_argument('--anomaly-weight', type=float, default=0.3,
                       help='Weight for anomaly score (default: 0.3)')
    parser.add_argument('--top-percentile', type=int, default=5,
                       help='Top N%% to flag (default: 5)')
    
    args = parser.parse_args()
    
    if args.model:
        # Test single model
        test_fusion(args.model, args.rule_weight, args.anomaly_weight, args.top_percentile)
    else:
        # Test all models
        compare_all_models(args.rule_weight, args.anomaly_weight, args.top_percentile)


if __name__ == '__main__':
    main()
