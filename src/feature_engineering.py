"""
feature_engineering.py - Feature Engineering & Autoencoder Feature Learning
===========================================================================
Creates domain-specific clinical features and learns latent representations
using a TensorFlow/Keras Autoencoder for enhanced predictive power.

Feature Modes (controlled by CONFIG["feature_mode"])
-----------------------------------------------------
    'combined'        : Original/clinical features + AE latent space (default)
    'latent_only'     : Only the AE latent space (pure dimensionality reduction)
    'engineered_only' : Clinical features only, no Autoencoder
"""


import os
import sys
import numpy as np
import pandas as pd

from typing import Tuple

from src.utils import (CONFIG, setup_logger, ensure_directories, save_model, load_model, MODELS_DIR)

logger = setup_logger(__name__)
LOGGER = logger




# ─────────────────────── Domain Features ─────────────────────────────

def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features based on medical knowledge.

    Features Created
    ----------------
    Blood Pressure
      - pulse_pressure         : systolic_bp - diastolic_bp
      - mean_arterial_pressure : diastolic_bp + (pulse_pressure / 3)
      - bp_category            : risk bucket (0=normal … 4=hypertensive crisis)

    Lipid Profile
      - cholesterol_ratio      : total_cholesterol / hdl
      - ldl_hdl_ratio          : ldl / hdl
      - non_hdl_cholesterol    : total_cholesterol - hdl
      - triglyceride_hdl_ratio : triglycerides / hdl

    BMI / Anthropometry
      - bmi_category           : risk bucket (0=underweight … 3=obese)

    Age
      - age_decade             : age // 10
      - age_squared            : age²

    Metabolic
      - glucose_insulin_ratio  : glucose / (insulin + ε)
      - insulin_resistance_idx : glucose × insulin / 405  (HOMA-IR proxy)

    Composite Risk
      - cardiovascular_index   : (sbp/120) × (chol/200) × (age/50)
      - lifestyle_score        : mean of activity, sleep, diet

    Interactions & Comorbidities
      - smoking_age_interaction
      - comorbidity_count      : hypertension + diabetes + family_history

    Polynomial (clinical core features)
      - systolic_bp², bmi², age × bmi, cholesterol_ratio²
    """
    df = df.copy()

    # ── Blood Pressure Features ──
    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["mean_arterial_pressure"] = (
            df["diastolic_bp"] + (df["pulse_pressure"] / 3)
        )
        df["bp_category"] = pd.cut(
            df["systolic_bp"],
            bins=[0, 120, 130, 140, 180, 300],
            labels=[0, 1, 2, 3, 4],
        ).astype(float).fillna(0)
        # Polynomial
        df["systolic_bp_sq"] = df["systolic_bp"] ** 2
        logger.info("Created blood pressure features")

    # ── Lipid Profile Features ──
    if "total_cholesterol" in df.columns and "hdl" in df.columns:
        df["cholesterol_ratio"] = df["total_cholesterol"] / (df["hdl"] + 1e-6)
        df["cholesterol_ratio_sq"] = df["cholesterol_ratio"] ** 2
        if "ldl" in df.columns:
            df["ldl_hdl_ratio"] = df["ldl"] / (df["hdl"] + 1e-6)
        df["non_hdl_cholesterol"] = df["total_cholesterol"] - df["hdl"]
        if "triglycerides" in df.columns:
            df["triglyceride_hdl_ratio"] = (
                df["triglycerides"] / (df["hdl"] + 1e-6)
            )
        logger.info("Created lipid profile features")

    # ── BMI Features ──
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(1)
        df["bmi_sq"] = df["bmi"] ** 2
        logger.info("Created BMI category feature")

    # ── Age Features ──
    if "age" in df.columns:
        df["age_decade"] = (df["age"] // 10).astype(int)
        df["age_squared"] = df["age"] ** 2
        # Interaction: age × bmi
        if "bmi" in df.columns:
            df["age_bmi_interaction"] = df["age"] * df["bmi"]
        logger.info("Created age-based features")

    # ── Metabolic Features ──
    if "glucose" in df.columns and "insulin" in df.columns:
        df["glucose_insulin_ratio"] = df["glucose"] / (df["insulin"] + 1e-6)
        df["insulin_resistance_idx"] = (df["glucose"] * df["insulin"]) / 405
        logger.info("Created metabolic features")

    # ── Composite Cardiovascular Risk Index ──
    if all(c in df.columns for c in ["systolic_bp", "total_cholesterol", "age"]):
        df["cardiovascular_index"] = (
            (df["systolic_bp"] / 120)
            * (df["total_cholesterol"] / 200)
            * (df["age"] / 50)
        )
        logger.info("Created cardiovascular index feature")

    # ── Lifestyle Score ──
    lifestyle_cols = ["physical_activity_level", "sleep_hours", "diet_quality"]
    available = [c for c in lifestyle_cols if c in df.columns]
    if available:
        df["lifestyle_score"] = df[available].mean(axis=1)
        logger.info("Created lifestyle score feature")

    # ── Interaction & Comorbidity Features ──
    if "smoking_status" in df.columns and "age" in df.columns:
        df["smoking_age_interaction"] = df["smoking_status"] * df["age"]

    if "hypertension" in df.columns and "diabetes" in df.columns:
        fh = df.get("family_history_heart_disease", pd.Series(
            0, index=df.index
        ))
        df["comorbidity_count"] = (
            df["hypertension"].astype(int)
            + df["diabetes"].astype(int)
            + fh.astype(int)
        )
        logger.info("Created comorbidity count feature")

    # Replace any infinities or NaNs introduced
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    logger.info(f"Feature engineering complete — {df.shape[1]} total features")
    return df


# ─────────────────────── Autoencoder Feature Learning ────────────────

def build_autoencoder(input_dim: int, encoding_dim: int = None):
    """
    Build a symmetric Autoencoder for unsupervised feature learning.

    Architecture
    ------------
    Input → 128 → 64 → 32 → encoding_dim → 32 → 64 → 128 → Input

    Parameters
    ----------
    input_dim : int
        Number of input features.
    encoding_dim : int
        Dimension of the bottleneck layer (latent space).

    Returns
    -------
    tuple : (autoencoder, encoder)
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    if encoding_dim is None:
        encoding_dim = CONFIG["autoencoder"]["encoding_dim"]

    # ── Encoder ──
    input_layer = layers.Input(shape=(input_dim,), name="encoder_input")

    x = layers.Dense(128, name="enc_dense1")(input_layer)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2, name="enc_dropout1")(x)

    x = layers.Dense(64, name="enc_dense2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15, name="enc_dropout2")(x)

    x = layers.Dense(32, name="enc_dense3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.Activation("relu")(x)

    encoded = layers.Dense(
        encoding_dim, activation="relu", name="bottleneck"
    )(x)

    # ── Decoder ──
    x = layers.Dense(32, name="dec_dense1")(encoded)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15, name="dec_dropout1")(x)

    x = layers.Dense(64, name="dec_dense2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2, name="dec_dropout2")(x)

    x = layers.Dense(128, name="dec_dense3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.Activation("relu")(x)

    decoded = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)

    autoencoder = Model(input_layer, decoded, name="autoencoder")
    encoder = Model(input_layer, encoded, name="encoder")

    optimizer = keras.optimizers.Adam(
        learning_rate=CONFIG["autoencoder"]["learning_rate"]
    )
    autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    logger.info(
        f"Autoencoder built: {input_dim} -> 128 -> 64 -> 32 -> "
        f"{encoding_dim} (bottleneck) -> 32 -> 64 -> 128 -> {input_dim}"
    )
    return autoencoder, encoder


def train_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
) -> Tuple:
    """
    Train the Autoencoder on training data.

    Parameters
    ----------
    X_train : np.ndarray
    X_test  : np.ndarray, optional
        Validation data for early stopping.

    Returns
    -------
    tuple : (encoder, training_history)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except Exception as e:
        LOGGER.warning(f"TensorFlow not available ({e}), skipping autoencoder training.")
        # Fallback: use PCA as encoder
        from sklearn.decomposition import PCA
        encoder = PCA(n_components=CONFIG["autoencoder"]["encoding_dim"])
        encoder.fit(X_train)
        # Mimic Keras encoder interface
        class EncoderWrapper:
            def __init__(self, pca):
                self.pca = pca
            def predict(self, X, *args, **kwargs):
                return self.pca.transform(X)
            def get_layer(self, name):
                return self
        encoder = EncoderWrapper(encoder)
        history = None
        return encoder, history

    ae_config = CONFIG["autoencoder"]
    input_dim = X_train.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=ae_config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    validation_data = (X_test, X_test) if X_test is not None else None

    logger.info("Training Autoencoder for feature learning...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=ae_config["epochs"],
        batch_size=ae_config["batch_size"],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    ensure_directories()
    encoder_path = str(MODELS_DIR / "autoencoder_encoder.keras")
    encoder.save(encoder_path)
    logger.info(f"Encoder saved to: {encoder_path}")

    # Save encoding metadata
    meta = {
        "input_dim": input_dim,
        "encoding_dim": ae_config["encoding_dim"],
        "feature_mode": CONFIG["feature_mode"],
    }
    save_model(meta, str(MODELS_DIR / "ae_metadata.pkl"))

    return encoder, history


def extract_autoencoder_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_encoder: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract learned features from the Autoencoder.

    Behaviour depends on CONFIG["feature_mode"]:
      - 'combined'    : original features + latent AE features
      - 'latent_only' : only the latent AE features (dimensionality reduction)
      - 'engineered_only': passthrough (no autoencoder)

    Parameters
    ----------
    X_train, X_test  : pd.DataFrame
    train_encoder    : bool
        If True train a new encoder; if False load the saved one.

    Returns
    -------
    tuple : (X_train_enhanced, X_test_enhanced)
    """
    feature_mode = CONFIG.get("feature_mode", "combined")

    if feature_mode == "engineered_only":
        logger.info("Feature mode = 'engineered_only': skipping Autoencoder")
        # Since autoencoder is unavailable, ensure we return engineered features
        return X_train, X_test
    # If autoencoder functions failed earlier, they may have returned PCA wrapper.
    # The rest of the logic remains unchanged.


    if train_encoder:
        encoder, history = train_autoencoder(X_train.values, X_test.values)
    else:
        try:
            import tensorflow as tf
            encoder_path = str(MODELS_DIR / "autoencoder_encoder.keras")
            encoder = tf.keras.models.load_model(encoder_path)
        except Exception as e:
            LOGGER.warning(f"Failed to load TensorFlow encoder ({e}), using PCA fallback.")
            from sklearn.decomposition import PCA
            encoder = PCA(n_components=CONFIG["autoencoder"]["encoding_dim"])
            encoder.fit(X_train.values)
            class EncoderWrapper:
                def __init__(self, pca):
                    self.pca = pca
                def predict(self, X, **kwargs):
                    # Accept any kwargs like 'verbose' to match Keras API
                    return self.pca.transform(X)
                def get_layer(self, name):
                    return self
            encoder = EncoderWrapper(encoder)
            LOGGER.info("Loaded PCA fallback encoder for autoencoder features.")
        try:
            logger.info(f"Loaded pre-trained encoder from: {encoder_path}")
        except NameError:
            pass


    encoding_dim = CONFIG["autoencoder"]["encoding_dim"]
    encoded_cols = [f"ae_latent_{i}" for i in range(encoding_dim)]

    train_encoded = encoder.predict(X_train.values, verbose=0)
    test_encoded  = encoder.predict(X_test.values,  verbose=0)

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
    test_encoded_df  = pd.DataFrame(test_encoded,  columns=encoded_cols, index=X_test.index)

    if feature_mode == "latent_only":
        logger.info(
            f"Feature mode = 'latent_only': returning {encoding_dim} latent features only"
        )
        return train_encoded_df, test_encoded_df

    # Default: 'combined'
    X_train_enhanced = pd.concat([X_train, train_encoded_df], axis=1)
    X_test_enhanced  = pd.concat([X_test,  test_encoded_df],  axis=1)

    logger.info(
        f"Feature mode = 'combined': {X_train.shape[1]} -> "
        f"{X_train_enhanced.shape[1]} features (+{encoding_dim} latent)"
    )
    return X_train_enhanced, X_test_enhanced


# ─────────────────────── Full Feature Engineering Pipeline ───────────

def run_feature_engineering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_autoencoder: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full feature engineering pipeline.

    Steps
    -----
    1. Create domain-specific clinical features (ratios, BP, lipid, age,
       metabolic, composite risk, interaction, polynomial features).
    2. Re-scale the newly created features with RobustScaler.
    3. (Optional) Train Autoencoder and extract latent features according
       to CONFIG['feature_mode'].

    Parameters
    ----------
    X_train, X_test  : pd.DataFrame
        Scaled train/test feature DataFrames from preprocessing.
    use_autoencoder  : bool
        If False, forces 'engineered_only' mode.

    Returns
    -------
    tuple : (X_train_final, X_test_final)
    """
    logger.info("=" * 70)
    logger.info("  FEATURE ENGINEERING PIPELINE STARTED")
    logger.info(f"  Mode: {CONFIG.get('feature_mode', 'combined').upper()}")
    logger.info("=" * 70)

    # Step 1: Domain features
    X_train_eng = create_clinical_features(X_train)
    X_test_eng  = create_clinical_features(X_test)

    # Step 2: Scale newly created features
    from sklearn.preprocessing import RobustScaler
    new_cols = [c for c in X_train_eng.columns if c not in X_train.columns]
    if new_cols:
        scaler_new = RobustScaler()
        X_train_eng[new_cols] = scaler_new.fit_transform(X_train_eng[new_cols])
        X_test_eng[new_cols]  = scaler_new.transform(X_test_eng[new_cols])
        save_model(scaler_new, str(MODELS_DIR / "new_features_scaler.pkl"))
        logger.info(f"Scaled {len(new_cols)} new engineered features")

    # Step 3: Autoencoder features (or passthrough)
    if not use_autoencoder:
        # Override: ignore autoencoder even if mode is 'combined'
        X_train_final = X_train_eng
        X_test_final  = X_test_eng
    else:
        X_train_final, X_test_final = extract_autoencoder_features(
            X_train_eng, X_test_eng, train_encoder=True
        )

    # Save final feature names for inference alignment
    final_feature_names = X_train_final.columns.tolist()
    save_model(final_feature_names, str(MODELS_DIR / "final_feature_names.pkl"))

    logger.info("=" * 70)
    logger.info(
        f"  FEATURE ENGINEERING COMPLETE — {len(final_feature_names)} features"
    )
    logger.info("=" * 70)

    return X_train_final, X_test_final


# ─────────────────────── CLI Entry Point ─────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline

    X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline()
    X_train_final, X_test_final = run_feature_engineering(
        X_train, X_test, use_autoencoder=True
    )

    print(f"\n[Success] Feature engineering complete!")
    print(f"   Mode    : {CONFIG.get('feature_mode', 'combined').upper()}")
    print(f"   Train   : {X_train_final.shape}")
    print(f"   Test    : {X_test_final.shape}")
