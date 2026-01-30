# Detection Model - Version 1

## Overview
This directory contains the complete implementation of the AML detection model (Version 1).

**Approach**: Hybrid Rule-Based (70%) + Isolation Forest (30%)

**Status**: [IN_PROGRESS | COMPLETE | ARCHIVED]

**Date Created**: 2025-01-24

---

## Directory Structure

```
detection_model_v1/
├── README.md                          # This file
├── config/                            # Configuration files
│   ├── red_flag_config.py            # Red flag thresholds and weights
│   ├── model_config.py                # Model hyperparameters
│   └── feature_selection.py           # Feature lists for anomaly detection
├── data/                              # Data files (inputs/outputs)
│   ├── input/                         # Input data
│   │   └── master_features.csv       # Link to or copy of input features
│   ├── intermediate/                  # Intermediate processing results
│   │   ├── preprocessed_features.csv  # Cleaned/normalized features
│   │   ├── rule_based_scores.csv     # Rule-based risk scores
│   │   ├── anomaly_scores.csv        # Anomaly detection scores
│   │   └── risk_details.csv           # Category breakdowns
│   └── output/                        # Final outputs
│       ├── model_output.csv           # Task 2 output
│       └── model_output_explanations.csv  # Task 3 output
├── models/                            # Model code
│   ├── rule_based_scorer.py          # Rule-based scoring system
│   ├── anomaly_detector.py            # Isolation Forest implementation
│   └── risk_fusion.py                 # Score combination logic
├── preprocessing/                     # Data preprocessing
│   ├── feature_selector.py           # Feature selection
│   └── data_preprocessor.py           # Data cleaning and scaling
├── explainability/                    # Explanation generation
│   ├── explanation_generator.py      # Generate explanations
│   └── red_flag_mapper.py            # Map features to red flags
├── utils/                             # Utility functions
│   ├── validation.py                 # Output validation
│   └── visualization.py               # Risk score visualization
├── docs/                              # Documentation
│   ├── design_decisions.md            # Key design choices and rationale
│   ├── input_output_spec.md          # Input/output specifications
│   ├── model_performance.md           # Performance metrics (if available)
│   └── troubleshooting.md             # Known issues and solutions
├── scripts/                           # Execution scripts
│   ├── main.py                        # Main execution script
│   └── run_pipeline.py                # Full pipeline runner
├── logs/                              # Execution logs
│   └── execution_log_YYYYMMDD.txt    # Timestamped execution logs
└── archive/                           # Archived files (if needed)
    └── README.md                      # Why this version was archived
```

---

## Input/Output Documentation

### Inputs

| File | Source | Description | Date Loaded |
|------|--------|-------------|-------------|
| `master_features.csv` | `clean_data/features/final/` | 177 engineered features for 61,411 customers | YYYY-MM-DD |

### Outputs

| File | Description | Format | Validation |
|------|-------------|--------|------------|
| `model_output.csv` | Task 2: customer_id, predicted_label, risk_score | CSV | ✅ Passed |
| `model_output_explanations.csv` | Task 3: customer_id, explanation | CSV | ✅ Passed |

### Intermediate Files

| File | Description | Used By |
|------|-------------|---------|
| `preprocessed_features.csv` | Cleaned and normalized features | Rule-based scorer, Anomaly detector |
| `rule_based_scores.csv` | Risk scores from rule-based system | Risk fusion |
| `anomaly_scores.csv` | Anomaly scores from Isolation Forest | Risk fusion |
| `risk_details.csv` | Category-level risk breakdowns | Explanation generator |

---

## Execution Log

### Run 1: YYYY-MM-DD HH:MM
- **Input**: `master_features.csv` (61,411 customers, 177 features)
- **Process**: Rule-based scoring + Isolation Forest
- **Output**: 
  - Flagged: X customers (Y%)
  - Average risk score: Z.ZZ
- **Status**: ✅ Success | ❌ Failed
- **Notes**: [Any issues or observations]

---

## Configuration

### Model Parameters
- Rule-based weight: 0.7
- Anomaly detection weight: 0.3
- Threshold: Top 5% (percentile-based)

### Red Flag Categories
- Structuring & Amount Patterns: 25%
- High-Risk Channels: 25%
- Geographic Risk: 20%
- Behavioral Anomalies: 20%
- Profile Risk: 10%

---

## Key Design Decisions

1. **Why 70/30 split?**
   - Prioritize interpretability (rule-based) while catching anomalies
   
2. **Why Isolation Forest?**
   - Fast, scalable, works without labels
   
3. **Why top 5% threshold?**
   - Typical AML flag rate, balances detection vs. operational burden

---

## Known Issues

- [ ] Issue 1: Description
- [ ] Issue 2: Description

---

## Next Steps

- [ ] Validate outputs against requirements
- [ ] Generate explanations for all customers
- [ ] Performance analysis
- [ ] Documentation review

---

## Archive Notes

If this version is archived, document why:

**Reason for Archive**: [Description]
**Date Archived**: YYYY-MM-DD
**Lessons Learned**: [What worked, what didn't]
