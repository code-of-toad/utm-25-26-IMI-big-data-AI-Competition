# Partner Models Directory

This directory contains unsupervised model implementations from team members.

## Purpose

Store model scripts (LOF, CBLOF, ABOD, etc.) that generate score files for fusion with the rule-based algorithm.

## Structure

```
partner_models/
├── README.md           # This file
├── lof_model.py       # Local Outlier Factor implementation
├── cblof_model.py     # Cluster-Based LOF implementation
└── abod_model.py      # Angle-Based Outlier Detection implementation
```

## Output Requirements

All models must output CSV files to:
```
../data/intermediate/scores_*.csv
```

With format:
- `customer_id` (string)
- `score` (float, 0.0 to 1.0)

See `../../docs/PARTNER_CSV_REQUIREMENTS_FOR_FUSION.md` for complete specifications.

## Usage

1. Place model scripts in this directory
2. Run scripts to generate score files
3. Score files go to `../data/intermediate/`
4. Test fusion using `../test_fusion_combinations.py`
