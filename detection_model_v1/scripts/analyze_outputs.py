"""Analyze pipeline outputs and provide insights"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'output'
INTERMEDIATE_DIR = BASE_DIR / 'data' / 'intermediate'

print("="*70)
print("PIPELINE OUTPUT ANALYSIS")
print("="*70)

# Load outputs
model_output = pd.read_csv(OUTPUT_DIR / 'model_output.csv')
risk_details = pd.read_csv(INTERMEDIATE_DIR / 'risk_details.csv')
rule_scores = pd.read_csv(INTERMEDIATE_DIR / 'rule_based_scores.csv')
anomaly_scores = pd.read_csv(INTERMEDIATE_DIR / 'scores_isolation_forest.csv')

print("\n1. FINAL PREDICTIONS (model_output.csv)")
print("-" * 70)
print(f"Total customers: {len(model_output):,}")
flagged = model_output['predicted_label'].sum()
not_flagged = (model_output['predicted_label'] == 0).sum()
print(f"Flagged for investigation: {flagged:,} ({100*flagged/len(model_output):.2f}%)")
print(f"Not flagged: {not_flagged:,} ({100*not_flagged/len(model_output):.2f}%)")

print(f"\nRisk Score Statistics (all customers):")
print(f"  Min: {model_output['risk_score'].min():.4f}")
print(f"  Max: {model_output['risk_score'].max():.4f}")
print(f"  Mean: {model_output['risk_score'].mean():.4f}")
print(f"  Median: {model_output['risk_score'].median():.4f}")
print(f"  Std Dev: {model_output['risk_score'].std():.4f}")

print(f"\nFlagged customers - Risk score:")
flagged_scores = model_output[model_output['predicted_label'] == 1]['risk_score']
print(f"  Min: {flagged_scores.min():.4f}")
print(f"  Max: {flagged_scores.max():.4f}")
print(f"  Mean: {flagged_scores.mean():.4f}")
print(f"  Median: {flagged_scores.median():.4f}")

print(f"\nNot flagged customers - Risk score:")
not_flagged_scores = model_output[model_output['predicted_label'] == 0]['risk_score']
print(f"  Min: {not_flagged_scores.min():.4f}")
print(f"  Max: {not_flagged_scores.max():.4f}")
print(f"  Mean: {not_flagged_scores.mean():.4f}")
print(f"  Median: {not_flagged_scores.median():.4f}")

# Threshold analysis
threshold = flagged_scores.min()
print(f"\nThreshold (top 5%): {threshold:.4f}")
print(f"Customers just below threshold (0.30 - {threshold:.4f}): {(model_output['risk_score'].between(0.30, threshold)).sum():,}")

print("\n\n2. RISK SCORE BREAKDOWN (risk_details.csv)")
print("-" * 70)
print(f"Rule-based score statistics:")
print(f"  Min: {risk_details['rule_based_score'].min():.4f}")
print(f"  Max: {risk_details['rule_based_score'].max():.4f}")
print(f"  Mean: {risk_details['rule_based_score'].mean():.4f}")
print(f"  Median: {risk_details['rule_based_score'].median():.4f}")

print(f"\nAnomaly score statistics:")
print(f"  Min: {risk_details['anomaly_score'].min():.4f}")
print(f"  Max: {risk_details['anomaly_score'].max():.4f}")
print(f"  Mean: {risk_details['anomaly_score'].mean():.4f}")
print(f"  Median: {risk_details['anomaly_score'].median():.4f}")

print(f"\nFused risk score statistics:")
print(f"  Min: {risk_details['risk_score'].min():.4f}")
print(f"  Max: {risk_details['risk_score'].max():.4f}")
print(f"  Mean: {risk_details['risk_score'].mean():.4f}")
print(f"  Median: {risk_details['risk_score'].median():.4f}")

# Correlation analysis
correlation = risk_details['rule_based_score'].corr(risk_details['anomaly_score'])
print(f"\nCorrelation between rule-based and anomaly scores: {correlation:.4f}")

# Flagged customers breakdown
flagged_details = risk_details.merge(
    model_output[['customer_id', 'predicted_label']], 
    on='customer_id'
)
flagged_details = flagged_details[flagged_details['predicted_label'] == 1]

print(f"\nFlagged customers - Component scores:")
print(f"  Rule-based (mean): {flagged_details['rule_based_score'].mean():.4f}")
print(f"  Anomaly (mean): {flagged_details['anomaly_score'].mean():.4f}")
print(f"  Fused (mean): {flagged_details['risk_score'].mean():.4f}")

print("\n\n3. RULE-BASED CATEGORY BREAKDOWN")
print("-" * 70)
flagged_rule = rule_scores.merge(
    model_output[['customer_id', 'predicted_label']],
    on='customer_id'
)
flagged_rule = flagged_rule[flagged_rule['predicted_label'] == 1]

print("Average category scores for FLAGGED customers:")
print(f"  Structuring Risk: {flagged_rule['structuring_risk'].mean():.4f}")
print(f"  Channel Risk: {flagged_rule['channel_risk'].mean():.4f}")
print(f"  Geographic Risk: {flagged_rule['geographic_risk'].mean():.4f}")
print(f"  Behavioral Risk: {flagged_rule['behavioral_risk'].mean():.4f}")
print(f"  Profile Risk: {flagged_rule['profile_risk'].mean():.4f}")

print("\nAverage category scores for ALL customers:")
print(f"  Structuring Risk: {rule_scores['structuring_risk'].mean():.4f}")
print(f"  Channel Risk: {rule_scores['channel_risk'].mean():.4f}")
print(f"  Geographic Risk: {rule_scores['geographic_risk'].mean():.4f}")
print(f"  Behavioral Risk: {rule_scores['behavioral_risk'].mean():.4f}")
print(f"  Profile Risk: {rule_scores['profile_risk'].mean():.4f}")

print("\n\n4. SAMPLE FLAGGED CUSTOMERS")
print("-" * 70)
sample_flagged = model_output[model_output['predicted_label'] == 1].head(10)
for idx, row in sample_flagged.iterrows():
    customer_id = row['customer_id']
    risk_score = row['risk_score']
    
    # Get details
    details = risk_details[risk_details['customer_id'] == customer_id].iloc[0]
    rule_breakdown = rule_scores[rule_scores['customer_id'] == customer_id].iloc[0]
    
    print(f"\n{customer_id}:")
    print(f"  Risk Score: {risk_score:.4f} (Rule: {details['rule_based_score']:.4f}, Anomaly: {details['anomaly_score']:.4f})")
    print(f"  Top Risk Categories:")
    categories = {
        'Structuring': rule_breakdown['structuring_risk'],
        'Channel': rule_breakdown['channel_risk'],
        'Geographic': rule_breakdown['geographic_risk'],
        'Behavioral': rule_breakdown['behavioral_risk'],
        'Profile': rule_breakdown['profile_risk']
    }
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for cat, score in sorted_cats[:3]:
        if score > 0:
            print(f"    - {cat}: {score:.4f}")

print("\n\n5. BORDERLINE CASES (near threshold)")
print("-" * 70)
threshold = flagged_scores.min()
borderline = model_output[
    (model_output['risk_score'] >= threshold - 0.05) & 
    (model_output['risk_score'] <= threshold + 0.05)
].copy()
borderline = borderline.sort_values('risk_score', ascending=False)

print(f"Customers near threshold ({threshold - 0.05:.4f} - {threshold + 0.05:.4f}): {len(borderline):,}")
print(f"  Flagged: {borderline['predicted_label'].sum():,}")
print(f"  Not flagged: {(borderline['predicted_label'] == 0).sum():,}")

print("\n\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
