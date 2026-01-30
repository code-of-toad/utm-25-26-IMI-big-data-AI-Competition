"""Quick verification script for pipeline outputs"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'output'
INTERMEDIATE_DIR = BASE_DIR / 'data' / 'intermediate'

print("="*70)
print("Pipeline Output Verification")
print("="*70)

# Check model_output.csv
print("\n1. model_output.csv (Task 2 Deliverable):")
print("-" * 70)
output_file = OUTPUT_DIR / 'model_output.csv'
if output_file.exists():
    df = pd.read_csv(output_file)
    print(f"   [OK] File exists")
    print(f"   [OK] Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"   [OK] Columns: {', '.join(df.columns.tolist())}")
    print(f"\n   Data validation:")
    print(f"   - All customer_ids present: {df['customer_id'].nunique() == len(df)}")
    print(f"   - predicted_label in [0,1]: {(df['predicted_label'].isin([0,1])).all()}")
    print(f"   - risk_score in [0,1]: {(df['risk_score'] >= 0).all() and (df['risk_score'] <= 1).all()}")
    print(f"\n   Statistics:")
    print(f"   - Risk score range: {df['risk_score'].min():.4f} .. {df['risk_score'].max():.4f}")
    print(f"   - Mean risk score: {df['risk_score'].mean():.4f}")
    flagged = df['predicted_label'].sum()
    print(f"   - Flagged customers: {flagged:,} ({100*flagged/len(df):.2f}%)")
    print(f"\n   Sample rows:")
    print(df.head(10).to_string(index=False))
else:
    print(f"   [ERROR] File not found: {output_file}")

# Check intermediate files
print("\n\n2. Intermediate Files:")
print("-" * 70)

files_to_check = [
    ('rule_based_scores.csv', ['customer_id', 'rule_based_score']),
    ('scores_isolation_forest.csv', ['customer_id', 'score']),
    ('risk_details.csv', ['customer_id', 'risk_score']),
]

for filename, expected_cols in files_to_check:
    filepath = INTERMEDIATE_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"   [OK] {filename}: {len(df):,} rows")
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"      [WARNING] Missing columns: {missing_cols}")
        else:
            print(f"      [OK] All expected columns present")
    else:
        print(f"   [ERROR] {filename}: Not found")

print("\n" + "="*70)
print("Verification Complete")
print("="*70)
