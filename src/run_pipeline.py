"""
run_pipeline.py - End-to-End execution of Heart Disease classification pipeline
===========================================================================
This script orchestrates the full workflow:
1. Data preprocessing (cleaning, encoding, scaling, train/test split)
2. Advanced feature engineering + optional Autoencoder latent features
3. Model training (RandomForest or SVM)
4. Evaluation with comprehensive metrics

The design respects CONFIG['feature_mode'] to control feature representations.
"""

import sys
import os

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.train_model import train_classifier, evaluate_classifier
# No separate evaluate_model module needed; functions are in train_model
from src.utils import setup_logger, get_timestamp, CONFIG

logger = setup_logger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("  END-TO-END PIPELINE STARTED")
    logger.info("=" * 70)

    # 1️⃣ Preprocessing
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing_pipeline()
    logger.info(f"Preprocessing complete – train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 2️⃣ Feature Engineering (including Autoencoder based on CONFIG)
    X_train_fe, X_test_fe = run_feature_engineering(
        X_train, X_test, use_autoencoder=True
    )
    logger.info(f"Feature engineering complete – final shapes: {X_train_fe.shape}, {X_test_fe.shape}")

    # 3️⃣ Model Training
    # Combine features and target into a single DataFrame for the training helper
    import pandas as pd
    train_df = pd.concat([X_train_fe, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_fe, y_test.reset_index(drop=True)], axis=1)

    # Use the training pipeline from train_model (it expects raw df and will split again; we bypass by calling its internal functions)
    # For simplicity, we'll directly train a classifier here.
    from src.train_model import train_classifier, evaluate_classifier

    clf = train_classifier(
        X_train_fe.values,
        y_train.values,
        model_type="random_forest",
    )
    metrics = evaluate_classifier(clf, X_test_fe.values, y_test.values)
    logger.info(f"Evaluation metrics: {metrics}")

    # Persist model and encoder if needed
    from src.utils import save_model, get_timestamp, MODELS_DIR
    timestamp = get_timestamp()
    model_path = os.path.join(MODELS_DIR, f"random_forest_{timestamp}.joblib")
    save_model(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info("=" * 70)
    logger.info("  END-TO-END PIPELINE COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
