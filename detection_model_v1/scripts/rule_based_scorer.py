"""
Rule-Based Risk Scorer for AML Detection

Reads master_features.csv, applies 5-category red-flag rules (DETECTION_MODEL_PLAN),
outputs rule_based_scores.csv with customer_id, rule_based_score (0-1), and category breakdown.

Input:  clean_data/features/final/master_features.csv (or path via env/arg)
Output: detection_model_v1/data/intermediate/rule_based_scores.csv

Contract (EXECUTION_PLAN): rule_based_scores.csv has at least customer_id, rule_based_score.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MASTER_FEATURES_PATH = PROJECT_ROOT / 'clean_data' / 'features' / 'final' / 'master_features.csv'
OUTPUT_PATH = BASE_DIR / 'data' / 'intermediate' / 'rule_based_scores.csv'

# ---------------------------------------------------------------------------
# Category 1: Structuring & Amount Patterns (Weight: 25%)
# ---------------------------------------------------------------------------
STRUCTURING_FEATURES = {
    'just_below_threshold_count': {'threshold': 3, 'weight': 0.15},
    'structuring_pattern_flag': {'threshold': 1, 'weight': 0.20},
    'round_amount_pct': {'threshold': 0.5, 'weight': 0.10},
    'round_amount_flag': {'threshold': 1, 'weight': 0.10},
    'large_txn_count_10k': {'threshold': 5, 'weight': 0.15},
    'large_txn_count_50k': {'threshold': 2, 'weight': 0.20},
    'amount_near_50k_increment_count': {'threshold': 3, 'weight': 0.10},
}

# ---------------------------------------------------------------------------
# Category 2: High-Risk Channels (Weight: 25%)
# ---------------------------------------------------------------------------
CHANNEL_FEATURES = {
    'has_wire_transfers': {'threshold': 1, 'weight': 0.15},
    'wire_large_count': {'threshold': 2, 'weight': 0.20},
    'wire_volume_total': {'threshold': 5_000_000, 'weight': 0.15},  # $50K cents
    'has_western_union': {'threshold': 1, 'weight': 0.25},
    'abm_cash_txn_count': {'threshold': 10, 'weight': 0.10},
    'abm_cash_large_count': {'threshold': 3, 'weight': 0.15},
    'structured_cash_deposits_same_day': {'threshold': 1, 'weight': 0.20},
    'multi_channel_flag': {'threshold': 1, 'weight': 0.10},
}

# ---------------------------------------------------------------------------
# Category 3: Geographic Risk (Weight: 20%)
# ---------------------------------------------------------------------------
GEOGRAPHIC_FEATURES = {
    'cross_border_flag': {'threshold': 1, 'weight': 0.15},
    'cross_border_txn_pct': {'threshold': 0.3, 'weight': 0.20},
    'is_country_offshore_structure_jurisdiction': {'threshold': 1, 'weight': 0.15},
    'is_country_tbml_high_risk': {'threshold': 1, 'weight': 0.20},
    'is_country_shell_company_jurisdiction': {'threshold': 1, 'weight': 0.15},
    'high_geographic_dispersion': {'threshold': 1, 'weight': 0.15},
    'customer_country_offshore_structure_jurisdiction': {'threshold': 1, 'weight': 0.10},
}

# ---------------------------------------------------------------------------
# Category 4: Behavioral Anomalies (Weight: 20%)
# ---------------------------------------------------------------------------
BEHAVIORAL_FEATURES = {
    'lifestyle_mismatch': {'threshold': 1, 'weight': 0.20},
    'severe_lifestyle_mismatch': {'threshold': 1, 'weight': 0.30},
    'flow_through_velocity_hours': {'threshold': 24, 'weight': 0.15, 'comparison': 'less_than'},
    'account_turnover_rate': {'threshold': 2.0, 'weight': 0.15},
    'sudden_inflow_outflow_pattern': {'threshold': 1, 'weight': 0.20},
    'volume_eft_sudden_increase': {'threshold': 1, 'weight': 0.15},
    'rapid_fire_flag': {'threshold': 1, 'weight': 0.10},
    'single_txn_exceeds_revenue': {'threshold': 1, 'weight': 0.25},
}

# ---------------------------------------------------------------------------
# Category 5: Profile Risk (Weight: 10%)
# ---------------------------------------------------------------------------
PROFILE_FEATURES = {
    'is_msb_business': {'threshold': 1, 'weight': 0.25},
    'is_shell_company': {'threshold': 1, 'weight': 0.25},
    'is_cash_intensive': {'threshold': 1, 'weight': 0.15},
    'new_account_flag': {'threshold': 1, 'weight': 0.10},
    'very_new_business_flag': {'threshold': 1, 'weight': 0.15},
    'high_profile_risk': {'threshold': 1, 'weight': 0.20},
    'has_missing_income': {'threshold': 1, 'weight': 0.05},
    'has_missing_sales': {'threshold': 1, 'weight': 0.05},
}


def calculate_category_score(features_df, feature_config, category_weight):
    """
    For each feature: if value meets threshold, add weight.
    Sum triggered weights, then multiply by category_weight.
    """
    category_score = pd.Series(0.0, index=features_df.index, dtype=float)

    for feature_name, config in feature_config.items():
        if feature_name not in features_df.columns:
            continue
        vals = features_df[feature_name].fillna(0)
        thresh = config['threshold']
        w = config['weight']
        cmp = config.get('comparison', 'greater_equal')
        if cmp == 'less_than':
            triggered = (vals < thresh).astype(float)
        else:
            triggered = (vals >= thresh).astype(float)
        category_score += triggered * w

    return category_score * category_weight


def calculate_rule_based_risk(features_df):
    """
    Returns (risk_scores, risk_details).
    risk_scores: Series 0-1.
    risk_details: DataFrame with structuring_risk, channel_risk, geographic_risk, behavioral_risk, profile_risk.
    """
    idx = features_df.index
    risk_scores = pd.Series(0.0, index=idx, dtype=float)
    risk_details = pd.DataFrame(index=idx)

    s = calculate_category_score(features_df, STRUCTURING_FEATURES, 0.25)
    risk_scores += s
    risk_details['structuring_risk'] = s.values

    s = calculate_category_score(features_df, CHANNEL_FEATURES, 0.25)
    risk_scores += s
    risk_details['channel_risk'] = s.values

    s = calculate_category_score(features_df, GEOGRAPHIC_FEATURES, 0.20)
    risk_scores += s
    risk_details['geographic_risk'] = s.values

    s = calculate_category_score(features_df, BEHAVIORAL_FEATURES, 0.20)
    risk_scores += s
    risk_details['behavioral_risk'] = s.values

    s = calculate_category_score(features_df, PROFILE_FEATURES, 0.10)
    risk_scores += s
    risk_details['profile_risk'] = s.values

    risk_scores = risk_scores.clip(0.0, 1.0)
    return risk_scores, risk_details


def main(input_path=None, output_path=None):
    input_path = Path(input_path) if input_path else MASTER_FEATURES_PATH
    output_path = Path(output_path) if output_path else OUTPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"master_features not found: {input_path}")

    print("Loading master_features...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    print("Computing rule-based risk...")
    risk_scores, risk_details = calculate_rule_based_risk(df)

    out = df[['customer_id']].copy()
    out['rule_based_score'] = risk_scores.values
    out[['structuring_risk', 'channel_risk', 'geographic_risk', 'behavioral_risk', 'profile_risk']] = risk_details

    # Round all float columns to 2 decimal places to avoid floating-point issues downstream
    float_cols = ['rule_based_score', 'structuring_risk', 'channel_risk', 'geographic_risk', 'behavioral_risk', 'profile_risk']
    out[float_cols] = out[float_cols].round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  rule_based_score range: {out['rule_based_score'].min():.3f} .. {out['rule_based_score'].max():.3f}")
    print(f"  Mean: {out['rule_based_score'].mean():.3f}")
    return out


if __name__ == '__main__':
    main()
