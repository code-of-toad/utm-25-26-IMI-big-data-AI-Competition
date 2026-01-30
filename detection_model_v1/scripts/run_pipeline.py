"""
Main Pipeline Execution Script

This script runs the complete detection model pipeline:
1. Load and preprocess features
2. Calculate rule-based scores
3. Train and run anomaly detection
4. Fuse risk scores
5. Generate predictions
6. Generate explanations
7. Validate outputs

All inputs and outputs are logged for documentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main pipeline execution"""
    logger.info("="*70)
    logger.info("AML Detection Model Pipeline - Version 1")
    logger.info("="*70)
    
    # Define paths
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / 'data' / 'input'
    INTERMEDIATE_DIR = BASE_DIR / 'data' / 'intermediate'
    OUTPUT_DIR = BASE_DIR / 'data' / 'output'
    
    # Create directories
    for dir_path in [INPUT_DIR, INTERMEDIATE_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load input data
        logger.info("\n[STEP 1] Loading input data...")
        input_file = BASE_DIR.parent / 'clean_data' / 'features' / 'final' / 'master_features.csv'
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        logger.info(f"Loading from: {input_file}")
        features_df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(features_df):,} customers with {len(features_df.columns)} features")
        
        # Log input summary
        logger.info(f"Input summary:")
        logger.info(f"  - Customer IDs: {features_df['customer_id'].nunique():,}")
        logger.info(f"  - Date loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save copy of input for documentation
        input_copy = INPUT_DIR / 'master_features.csv'
        features_df.to_csv(input_copy, index=False)
        logger.info(f"Input saved to: {input_copy}")
        
        # Step 2: Preprocess features (for anomaly detection)
        logger.info("\n[STEP 2] Preprocessing features for anomaly detection...")
        from preprocessing.data_preprocessor import preprocess_features
        preprocessed_df, scaler = preprocess_features(features_df, fit_scaler=True)
        preprocessed_df.to_csv(INTERMEDIATE_DIR / 'preprocessed_features.csv', index=False)
        logger.info(f"Preprocessed {len(preprocessed_df.columns)} features")
        logger.info("Preprocessing complete")
        
        # Step 3: Rule-based scoring
        logger.info("\n[STEP 3] Calculating rule-based risk scores...")
        from scripts.rule_based_scorer import main as run_rule_based_scorer
        rule_scores_path = INTERMEDIATE_DIR / 'rule_based_scores.csv'
        rule_scores_df = run_rule_based_scorer(
            input_path=input_file,
            output_path=rule_scores_path
        )
        logger.info(f"Rule-based scoring complete: {len(rule_scores_df):,} customers scored")
        logger.info(f"  Score range: {rule_scores_df['rule_based_score'].min():.3f} .. {rule_scores_df['rule_based_score'].max():.3f}")
        
        # Step 4: Anomaly detection
        logger.info("\n[STEP 4] Training Isolation Forest...")
        from models.anomaly_detector import train_isolation_forest, predict_anomaly_scores, save_model
        MODEL_DIR = BASE_DIR / 'models' / 'saved'
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        detector, scaler, _ = train_isolation_forest(features_df, contamination=0.05)
        logger.info("Isolation Forest trained")
        
        anomaly_scores_df = predict_anomaly_scores(detector, scaler, features_df)
        anomaly_scores_path = INTERMEDIATE_DIR / 'scores_isolation_forest.csv'
        anomaly_scores_df.to_csv(anomaly_scores_path, index=False)
        logger.info(f"Anomaly detection complete: {len(anomaly_scores_df):,} customers scored")
        logger.info(f"  Score range: {anomaly_scores_df['score'].min():.4f} .. {anomaly_scores_df['score'].max():.4f}")
        
        # Save model for future use
        save_model(detector, scaler, MODEL_DIR)
        logger.info(f"Model saved to: {MODEL_DIR}")
        
        # Step 5: Risk score fusion
        logger.info("\n[STEP 5] Fusing risk scores...")
        from models.risk_fusion import fuse_risk_scores, generate_predictions
        fused_df = fuse_risk_scores(rule_scores_df, anomaly_scores_df, rule_weight=0.7, anomaly_weight=0.3)
        fused_df.to_csv(INTERMEDIATE_DIR / 'risk_details.csv', index=False)
        logger.info(f"Risk score fusion complete: {len(fused_df):,} customers")
        logger.info(f"  Fused score range: {fused_df['risk_score'].min():.4f} .. {fused_df['risk_score'].max():.4f}")
        
        # Step 6: Generate predictions
        logger.info("\n[STEP 6] Generating predictions...")
        predictions_df, threshold = generate_predictions(fused_df, threshold=None, top_percentile=5)
        predictions_df.to_csv(OUTPUT_DIR / 'model_output.csv', index=False)
        flagged_count = predictions_df['predicted_label'].sum()
        logger.info(f"Prediction generation complete")
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Flagged: {flagged_count:,} customers ({100*flagged_count/len(predictions_df):.2f}%)")
        
        # Step 7: Generate explanations
        logger.info("\n[STEP 7] Generating explanations...")
        # TODO: Implement explanation generation
        # explanations = generate_explanations(features_df, final_scores, predictions)
        # explanations.to_csv(OUTPUT_DIR / 'model_output_explanations.csv', index=False)
        logger.info("Explanation generation complete (placeholder)")
        
        # Step 8: Validate outputs
        logger.info("\n[STEP 8] Validating outputs...")
        # TODO: Implement validation
        # validate_outputs(predictions, explanations, features_df)
        logger.info("Validation complete (placeholder)")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Execution log saved to: {log_file}")
        logger.info(f"Outputs saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
