"""train_model.py - Model training utilities for Heart Disease pipeline"""


import logging
from pathlib import Path

# Removed erroneous duplicate tkinter imports


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
# TensorFlow/Keras will be imported lazily within functions that need it

from typing import Any

# sys.path insertion removed; project root is in PYTHONPATH
from src.utils import (CONFIG, ensure_directories, get_timestamp, load_raw_data,
                   load_model, save_model, setup_logger, dataset_summary, MODELS_DIR)

LOGGER = setup_logger(__name__)


def build_autoencoder(input_dim: int, encoding_dim: int = CONFIG["autoencoder"]["encoding_dim"]):
    """Build a feature extractor.

    Tries to create a TensorFlow/Keras autoencoder. If TensorFlow is not installed,
    falls back to a scikit-learn PCA wrapper that mimics the required methods.
    """
    try:
        # Lazy import to avoid import error when TensorFlow is missing
        from tensorflow import keras
        encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(encoding_dim, activation="relu", name="latent"),
        ], name="encoder")
        decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(encoding_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ], name="decoder")
        autoencoder = keras.Model(encoder.input, decoder(encoder.output), name="autoencoder")
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG["autoencoder"]["learning_rate"]),
            loss="mse",
        )
        return autoencoder
    except Exception as e:
        LOGGER.warning(f"TensorFlow not available ({e}), using PCA fallback for feature learning.")
        from sklearn.decomposition import PCA
        class PCAWrapper:
            def __init__(self, n_components: int):
                self.pca = PCA(n_components=n_components)
                self.n_components = n_components
            def fit(self, X, y=None, **kwargs):
                self.pca.fit(X)
                return self
            def predict(self, X):
                return self.pca.transform(X)
            def get_layer(self, name):
                # Mimic Keras encoder layer interface used elsewhere
                return self
            def __call__(self, X):
                return self.predict(X)
        # Use PCA to mimic encoder part; decoder is not needed for downstream tasks
        return PCAWrapper(encoding_dim)


def train_autoencoder(X: np.ndarray) -> Any:
    """Train the autoencoder on the provided feature matrix.

    Returns the fitted autoencoder. Handles both TensorFlow and PCA fallback.
    """
    # Build the autoencoder (may return a Keras model or a PCA wrapper)
    ae = build_autoencoder(input_dim=X.shape[1])
    # Try to import Keras for callbacks; if unavailable, skip them
    try:
        from tensorflow import keras
        early_stop = keras.callbacks.EarlyStopping(
            patience=CONFIG["autoencoder"]["patience"], restore_best_weights=True
        )
        callbacks = [early_stop]
    except Exception:
        callbacks = []
        LOGGER.warning("TensorFlow/Keras not available; training PCA fallback without callbacks.")
    # Fit the model (PCA wrapper's fit will ignore callbacks)
    ae.fit(
        X,
        X,
        epochs=CONFIG["autoencoder"]["epochs"] if hasattr(ae, "fit") else None,
        batch_size=CONFIG["autoencoder"]["batch_size"] if hasattr(ae, "fit") else None,
        validation_split=0.1 if hasattr(ae, "fit") else None,
        callbacks=callbacks,
        verbose=0,
    )
    LOGGER.info("Autoencoder training completed")
    return ae


def extract_latent_features(ae: Any, X: np.ndarray) -> np.ndarray:
    """Return the latent representation from the encoder part of the autoencoder."""
    encoder = ae.get_layer("encoder")
    return encoder.predict(X)


def prepare_features(df: pd.DataFrame, mode: str = CONFIG["feature_mode"]):
    """Prepare X, y according to the chosen feature mode.

    mode options:
        - "combined": original clinical features + latent vectors
        - "latent_only": only latent vectors
        - "engineered_only": only original clinical features
    """
    y = df[CONFIG["target_column"]].values
    X_original = df.drop(columns=[CONFIG["target_column"], CONFIG["id_column"]], errors="ignore").values
    if mode == "engineered_only":
        return X_original, y
    # Train autoencoder on original features
    ae = train_autoencoder(X_original)
    latent = extract_latent_features(ae, X_original)
    if mode == "latent_only":
        return latent, y
    # combined
    X_combined = np.hstack([X_original, latent])
    return X_combined, y


def train_classifier(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
    """Train a classifier and return the fitted model."""
    if model_type == "random_forest":
        clf = RandomForestClassifier(random_state=CONFIG["random_state"], n_estimators=200)
    elif model_type == "svm":
        clf = SVC(probability=True, random_state=CONFIG["random_state"])
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=CONFIG["random_state"], max_iter=1000)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=CONFIG["random_state"], eval_metric="logloss")
    elif model_type == "mlp_classifier":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(random_state=CONFIG["random_state"], max_iter=500)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    clf.fit(X, y)
    LOGGER.info(f"{model_type} training completed")
    return clf


def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute a suite of evaluation metrics and return them as a dictionary."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    LOGGER.info(f"Evaluation metrics: {metrics}")
    return metrics


import time
from typing import Tuple

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train all 5 classifiers and return dict with models and metadata."""
    models_to_train = {
        "Logistic_Regression": "logistic_regression",
        "Random_Forest": "random_forest",
        "XGBoost": "xgboost",
        "SVM": "svm",
        "MLP_Classifier": "mlp_classifier",
    }
    trained_models = {}
    for name, m_type in models_to_train.items():
        LOGGER.info(f"Training model: {name}...")
        try:
            start_time = time.time()
            model = train_classifier(X_train, y_train, model_type=m_type)
            train_time = time.time() - start_time
            trained_models[name] = {
                "model": model,
                "train_time": train_time,
            }
        except (ImportError, ModuleNotFoundError) as e:
            LOGGER.warning(f"Skipping model {name} because dependencies are not installed: {e}")
        except Exception as e:
            LOGGER.error(f"Failed to train model {name}: {e}")
    return trained_models


def select_best_model(trained_models: dict, X_train: np.ndarray = None, y_train: np.ndarray = None) -> Tuple[str, Any]:
    """Select the best model based on validation or CV score, and save it to models/best_model.pkl."""
    best_score = -1
    best_name = None
    best_model = None
    for name, info in trained_models.items():
        model = info["model"]
        if X_train is not None and y_train is not None:
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=CONFIG["cv_folds"], scoring=CONFIG["scoring_metric"])
            score = np.mean(scores)
            LOGGER.info(f"{name} CV {CONFIG['scoring_metric']}: {score:.4f}")
        else:
            # Fallback
            score = 0
            
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    ensure_directories()
    best_model_path = MODELS_DIR / "best_model.pkl"
    save_model(best_model, str(best_model_path))
    LOGGER.info(f"Best model selected: {best_name} with score {best_score:.4f}. Saved to {best_model_path}")
    return best_name, best_model


def full_training_pipeline(df: pd.DataFrame, model_type: str = "random_forest"):
    """End‑to‑end pipeline: split, feature engineering, train, evaluate and persist models.

    Returns a dict with metrics and paths to saved artifacts.
    """
    X, y = prepare_features(df, mode=CONFIG["feature_mode"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
    )
    clf = train_classifier(X_train, y_train, model_type=model_type)
    metrics = evaluate_classifier(clf, X_test, y_test)
    timestamp = get_timestamp()
    model_path = MODELS_DIR / f"{model_type}_{timestamp}.joblib"
    save_model(clf, str(model_path))
    return {"metrics": metrics, "model_path": str(model_path)}


if __name__ == "__main__":
    LOGGER.info("Starting training pipeline")
    df = load_raw_data()
    dataset_summary(df, name="Raw Dataset")
    results = full_training_pipeline(df, model_type="random_forest")
    LOGGER.info(f"Final results: {results}")

