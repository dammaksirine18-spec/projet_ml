"""
preprocessing.py - Data Preprocessing Pipeline
================================================
Handles data cleaning, missing value imputation, encoding, scaling,
and train/test splitting for the Heart Disease dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

import joblib

# Add project root to path for imports
# sys.path insertion removed; project root is in PYTHONPATH
from src.utils import (
    CONFIG, setup_logger, ensure_directories, load_raw_data,
    save_dataframe, save_model, dataset_summary,
    DATA_PROCESSED_DIR, DATA_TRAIN_TEST_DIR, MODELS_DIR,
)

logger = setup_logger(__name__)


# ─────────────────────── Data Cleaning ───────────────────────────────

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the patient_id column (non-predictive)."""
    id_col = CONFIG["id_column"]
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        logger.info(f"Dropped column: '{id_col}'")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info(f"Removed {n_removed} duplicate rows ({n_before} -> {len(df)})")
    return df


def handle_outliers(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """
    Cap extreme outliers using IQR method.

    Values beyond Q1 - factor*IQR or Q3 + factor*IQR are clipped.
    Uses a generous factor=3.0 to preserve clinical edge cases.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = CONFIG["target_column"]
    if target in numeric_cols:
        numeric_cols.remove(target)

    n_clipped_total = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        n_clipped_total += n_clipped
        df[col] = df[col].clip(lower=lower, upper=upper)

    logger.info(f"Outlier capping (IQR×{factor}): {n_clipped_total} values clipped")
    return df


# ─────────────────────── Missing Values ──────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with a multi-strategy approach:
    - Numeric columns: KNN imputation (k=5) for clinical accuracy.
    - Categorical columns: mode imputation.
    """
    n_missing_before = df.isnull().sum().sum()

    if n_missing_before == 0:
        logger.info("No missing values found — skipping imputation")
        return df

    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # KNN imputation for numeric features
    if numeric_cols:
        cols_with_missing = [c for c in numeric_cols if df[c].isnull().any()]
        if cols_with_missing:
            knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
            logger.info(f"KNN-imputed {len(cols_with_missing)} numeric columns")

    # Mode imputation for categorical features
    if categorical_cols:
        cols_with_missing = [c for c in categorical_cols if df[c].isnull().any()]
        if cols_with_missing:
            mode_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])
            logger.info(f"Mode-imputed {len(cols_with_missing)} categorical columns")

    n_missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {n_missing_before} -> {n_missing_after}")
    return df


# ─────────────────────── Encoding ────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical/ordinal features.

    For this dataset, most features are already numeric. This function
    handles any remaining object-type columns via LabelEncoding.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        logger.info("No categorical columns to encode")
        return df

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"Label-encoded '{col}' -> {len(le.classes_)} classes")

    # Save encoders for inference
    save_model(encoders, str(MODELS_DIR / "label_encoders.pkl"))
    return df


# ─────────────────────── Scaling ─────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "robust"
) -> tuple:
    """
    Scale features using RobustScaler (default) or StandardScaler.

    RobustScaler is preferred for clinical data as it handles
    outliers gracefully using median and IQR.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    method : str
        'robust' or 'standard'

    Returns
    -------
    tuple : (X_train_scaled, X_test_scaled, scaler)
    """
    if method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    feature_names = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    # Persist scaler for inference
    save_model(scaler, str(MODELS_DIR / "feature_scaler.pkl"))
    logger.info(f"Features scaled using {method.capitalize()}Scaler")

    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────── Train / Test Split ──────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = None,
    stratify: bool = True,
) -> tuple:
    """
    Split data into train and test sets with optional stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with target column.
    test_size : float
        Fraction of data for testing (default from CONFIG).
    stratify : bool
        Whether to stratify on the target column.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = CONFIG["test_size"]

    target = CONFIG["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=CONFIG["random_state"],
        stratify=stratify_col,
    )

    logger.info(f"Train/Test split: {X_train.shape[0]} / {X_test.shape[0]} samples")
    logger.info(f"Train target dist: {dict(y_train.value_counts())}")
    logger.info(f"Test  target dist: {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test


# ─────────────────────── Full Pipeline ───────────────────────────────

def run_preprocessing_pipeline(filepath: str = None) -> tuple:
    """
    Execute the complete preprocessing pipeline:

    1. Load raw data
    2. Drop ID column
    3. Remove duplicates
    4. Handle outliers (IQR capping)
    5. Handle missing values (KNN + mode)
    6. Encode categorical features
    7. Train/test split
    8. Scale features
    9. Save processed artifacts

    Returns
    -------
    tuple : (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    """
    ensure_directories()
    logger.info("=" * 70)
    logger.info("  PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 70)

    # Step 1: Load data
    df = load_raw_data(filepath)
    dataset_summary(df, "Raw Data")

    # Step 2: Drop ID
    df = drop_id_column(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Outlier handling
    df = handle_outliers(df)

    # Step 5: Missing values
    df = handle_missing_values(df)

    # Step 6: Encoding
    df = encode_features(df)

    # Save processed dataset
    processed_path = str(DATA_PROCESSED_DIR / "heart_disease_processed.csv")
    save_dataframe(df, processed_path)
    dataset_summary(df, "Processed Data")

    # Step 7: Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 8: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 9: Save train/test splits
    feature_names = X_train_scaled.columns.tolist()
    save_dataframe(X_train_scaled, str(DATA_TRAIN_TEST_DIR / "X_train.csv"))
    save_dataframe(X_test_scaled, str(DATA_TRAIN_TEST_DIR / "X_test.csv"))
    save_dataframe(y_train.to_frame(), str(DATA_TRAIN_TEST_DIR / "y_train.csv"))
    save_dataframe(y_test.to_frame(), str(DATA_TRAIN_TEST_DIR / "y_test.csv"))

    # Save feature names
    save_model(feature_names, str(MODELS_DIR / "feature_names.pkl"))

    logger.info("=" * 70)
    logger.info("  PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# ─────────────────────── CLI Entry Point ─────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline()
    print(f"\n[Success] Preprocessing complete!")
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Features: {len(features)}")
