"""
utils.py - Utility Functions for Heart Disease Risk Prediction
==============================================================
Provides logging, path management, configuration, and helper functions
used across all pipeline modules.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────── Project Paths ───────────────────────────

# Resolve project root dynamically (works in VS Code, Colab, CLI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"

# Dataset filename
RAW_DATASET = "heart_disease_synthetic_dataset.csv"
TARGET_COLUMN = "heart_disease"
ID_COLUMN = "patient_id"

# ─────────────────────────── Configuration ───────────────────────────

CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    "cv_folds": 5,
    "scoring_metric": "roc_auc",
    # Feature representation mode:
    #   'combined'        -> original/clinical features + AE latent space (default)
    #   'latent_only'     -> only the AE latent space (pure dimensionality reduction)
    #   'engineered_only' -> clinical features only, no autoencoder
    "feature_mode": "combined",
    "autoencoder": {
        "encoding_dim": 16,
        "epochs": 100,
        "batch_size": 64,
        "patience": 10,
        "learning_rate": 1e-3,
    },
    "target_column": TARGET_COLUMN,
    "id_column": ID_COLUMN,
}


# ─────────────────────────── Directory Setup ─────────────────────────

def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_TRAIN_TEST_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        REPORTS_DIR / "figures",
        NOTEBOOKS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── Logging Setup ───────────────────────────

def setup_logger(
    name: str = "heart_disease_ml",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a named logger with console and optional file handlers.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str, optional
        Path to log file. If None, logs go to reports/pipeline.log.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        ensure_directories()
        log_file = str(REPORTS_DIR / "pipeline.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ─────────────────────────── Data I/O ────────────────────────────────

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw Heart Disease dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV. Defaults to data/raw/heart_disease_synthetic_dataset.csv.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = str(DATA_RAW_DIR / RAW_DATASET)

    logger = setup_logger()
    logger.info(f"Loading raw data from: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded — shape: {df.shape}")
    return df


def save_dataframe(
    df: pd.DataFrame, filepath: str, index: bool = False
) -> None:
    """Save a DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger = setup_logger()
    logger.info(f"DataFrame saved to: {filepath}")


def save_model(model: Any, filepath: str) -> None:
    """Persist a model to disk using joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger = setup_logger()
    logger.info(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """Load a model from disk."""
    logger = setup_logger()
    logger.info(f"Loading model from: {filepath}")
    return joblib.load(filepath)


def save_json(data: dict, filepath: str) -> None:
    """Save a dictionary as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger = setup_logger()
    logger.info(f"JSON saved to: {filepath}")


def load_json(filepath: str) -> dict:
    """Load a JSON file and return as dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────── Helpers ─────────────────────────────────

def get_timestamp() -> str:
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def suppress_warnings() -> None:
    """Suppress common ML library warnings for cleaner output."""
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def print_separator(title: str = "", char: str = "═", width: int = 70) -> None:
    """Print a formatted section separator."""
    if title:
        padding = (width - len(title) - 2) // 2
        line = f"{char * padding} {title} {char * padding}"
    else:
        line = char * width
    print(f"\n{line}")


def dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a concise summary of a DataFrame."""
    logger = setup_logger()
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name} Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Shape          : {df.shape}")
    logger.info(f"  Memory Usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    logger.info(f"  Missing Values : {df.isnull().sum().sum()}")
    logger.info(f"  Duplicates     : {df.duplicated().sum()}")
    if CONFIG["target_column"] in df.columns:
        dist = df[CONFIG["target_column"]].value_counts()
        logger.info(f"  Target Dist    : {dict(dist)}")
    logger.info(f"{'='*60}\n")
