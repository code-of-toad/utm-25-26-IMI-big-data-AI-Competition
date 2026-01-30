"""Models module for anomaly detection and risk fusion"""

from .anomaly_detector import (
    train_isolation_forest,
    predict_anomaly_scores,
    save_model,
    load_model
)

from .risk_fusion import (
    fuse_risk_scores,
    generate_predictions
)

__all__ = [
    'train_isolation_forest',
    'predict_anomaly_scores',
    'save_model',
    'load_model',
    'fuse_risk_scores',
    'generate_predictions'
]
