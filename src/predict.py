"""
predict.py - Prediction Module
================================
Provides inference capabilities using the best trained model.
Supports single patient prediction, batch prediction from CSV,
and feature preprocessing for new data.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    CONFIG, setup_logger, load_model, MODELS_DIR,
)
from src.feature_engineering import create_clinical_features

logger = setup_logger(__name__)


# ─────────────────────── Load Pipeline Artifacts ─────────────────────

class HeartDiseasePrediction:
    """
    End-to-end prediction pipeline for Heart Disease Risk.

    Loads the saved model, scaler, and feature metadata to provide
    consistent inference for new patient data.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the prediction pipeline.

        Parameters
        ----------
        model_path : str, optional
            Path to the model file. Defaults to best_model.pkl.
        """
        self.logger = setup_logger("predictor")

        # Load model
        if model_path is None:
            model_path = str(MODELS_DIR / "best_model.pkl")
        self.model = load_model(model_path)
        self.logger.info(f"Model loaded: {type(self.model).__name__}")

        # Load scaler
        scaler_path = str(MODELS_DIR / "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = load_model(scaler_path)
            self.logger.info("Feature scaler loaded")
        else:
            self.scaler = None
            self.logger.warning("No scaler found — using raw features")

        # Load new features scaler (for engineered features)
        new_scaler_path = str(MODELS_DIR / "new_features_scaler.pkl")
        if os.path.exists(new_scaler_path):
            self.new_features_scaler = load_model(new_scaler_path)
        else:
            self.new_features_scaler = None

        # Load feature names
        feature_names_path = str(MODELS_DIR / "final_feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_names = load_model(feature_names_path)
            self.logger.info(f"Expected features: {len(self.feature_names)}")
        else:
            self.feature_names = None
            self.logger.warning("No feature names file found")

        # Load autoencoder encoder (optional)
        self.encoder = None
        encoder_path = str(MODELS_DIR / "autoencoder_encoder.keras")
        if os.path.exists(encoder_path):
            try:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf
                self.encoder = tf.keras.models.load_model(encoder_path)
                self.logger.info("Autoencoder encoder loaded")
            except ImportError:
                self.logger.warning("TensorFlow not available — skipping autoencoder features")

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new patient data through the same pipeline
        as training data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw patient data.

        Returns
        -------
        pd.DataFrame : Processed features ready for prediction.
        """
        df = df.copy()

        # Drop ID if present
        id_col = CONFIG["id_column"]
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

        # Drop target if present
        target = CONFIG["target_column"]
        if target in df.columns:
            df = df.drop(columns=[target])

        # Get the original feature columns (before engineering)
        original_features = [c for c in df.columns if c in df.select_dtypes(include=[np.number]).columns]

        # Scale original features
        if self.scaler is not None:
            scaler_features = list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else original_features
            # Fill missing columns with their training mean so they become 0 (neutral) after scaling
            # This avoids biasing the model with extreme values like HDL=0, LDL=0, etc.
            if hasattr(self.scaler, 'mean_'):
                for i, col in enumerate(scaler_features):
                    if col not in df.columns:
                        df[col] = self.scaler.mean_[i]  # training mean → scales to 0
            else:
                for col in scaler_features:
                    if col not in df.columns:
                        df[col] = 0
            df[scaler_features] = self.scaler.transform(df[scaler_features])

        # Create clinical features
        df = create_clinical_features(df)

        # Scale new engineered features
        if self.new_features_scaler is not None:
            scaler_cols = list(self.new_features_scaler.feature_names_in_) if hasattr(self.new_features_scaler, 'feature_names_in_') else []
            # Fill missing engineered features with their training mean
            if hasattr(self.new_features_scaler, 'mean_'):
                for i, col in enumerate(scaler_cols):
                    if col not in df.columns:
                        df[col] = self.new_features_scaler.mean_[i]
            else:
                for col in scaler_cols:
                    if col not in df.columns:
                        df[col] = 0
            if scaler_cols:
                df[scaler_cols] = self.new_features_scaler.transform(df[scaler_cols])

        # Add autoencoder features
        if self.encoder is not None:
            encoding_dim = CONFIG["autoencoder"]["encoding_dim"]
            encoded = self.encoder.predict(df.values, verbose=0)
            encoded_cols = [f"ae_latent_{i}" for i in range(encoding_dim)]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

        # Align with expected features
        if self.feature_names is not None:
            # Add missing columns with 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            # Select and order columns
            df = df[self.feature_names]

        return df

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Make predictions for patient data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw patient data (single or multiple rows).

        Returns
        -------
        dict : {
            "predictions": list of 0/1,
            "probabilities": list of float (probability of heart disease),
            "risk_levels": list of str ("Low" / "Moderate" / "High" / "Very High"),
        }
        """
        X = self.preprocess_input(df)

        predictions = self.model.predict(X.values)

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X.values)[:, 1]
        elif hasattr(self.model, "decision_function"):
            from scipy.special import expit
            probabilities = expit(self.model.decision_function(X.values))
        else:
            probabilities = predictions.astype(float)

        # Map to risk levels
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append("Low")
            elif prob < 0.5:
                risk_levels.append("Moderate")
            elif prob < 0.7:
                risk_levels.append("High")
            else:
                risk_levels.append("Very High")

        return {
            "predictions": predictions.tolist(),
            "probabilities": [round(p, 4) for p in probabilities.tolist()],
            "risk_levels": risk_levels,
        }

    def predict_single(self, patient_data: dict) -> dict:
        """
        Predict for a single patient from a dictionary of features.

        Parameters
        ----------
        patient_data : dict
            {feature_name: value} for one patient.

        Returns
        -------
        dict : Prediction result with risk level.
        """
        df = pd.DataFrame([patient_data])
        result = self.predict(df)

        return {
            "prediction": result["predictions"][0],
            "probability": result["probabilities"][0],
            "risk_level": result["risk_levels"][0],
            "diagnosis": "Heart Disease Detected" if result["predictions"][0] == 1 else "No Heart Disease",
        }

    def predict_from_csv(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """
        Batch prediction from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to input CSV.
        output_path : str, optional
            Path to save predictions. Auto-generated if None.

        Returns
        -------
        pd.DataFrame : Original data with prediction columns added.
        """
        df = pd.read_csv(filepath)
        result = self.predict(df)

        df["prediction"] = result["predictions"]
        df["probability"] = result["probabilities"]
        df["risk_level"] = result["risk_levels"]

        if output_path is None:
            output_path = filepath.replace(".csv", "_predictions.csv")
        df.to_csv(output_path, index=False)

        self.logger.info(f"Batch predictions saved to: {output_path}")
        return df


# ─────────────────────── CLI Entry Point ─────────────────────────────

if __name__ == "__main__":
    # Example: predict on a sample patient
    predictor = HeartDiseasePrediction()

    sample_patient = {
        "age": 55,
        "gender": 1,
        "height_cm": 170.0,
        "weight_kg": 85.0,
        "bmi": 29.4,
        "systolic_bp": 145,
        "diastolic_bp": 92,
        "resting_heart_rate": 78,
        "oxygen_saturation": 96.5,
        "glucose": 110.0,
        "insulin": 18.0,
        "total_cholesterol": 240,
        "hdl": 38,
        "ldl": 150,
        "triglycerides": 200,
        "hemoglobin": 14.5,
        "smoking_status": 1,
        "smoking_years": 20,
        "alcohol_consumption": 2.0,
        "physical_activity_level": 0,
        "sleep_hours": 5.5,
        "stress_level": 7.0,
        "diet_quality": 2.0,
        "diabetes": 1,
        "hypertension": 1,
        "family_history_heart_disease": 1,
        "cardiac_risk_score": 55.0,
        "metabolic_syndrome_score": 42.0,
    }

    result = predictor.predict_single(sample_patient)

    print("\n" + "=" * 50)
    print("  HEART DISEASE RISK PREDICTION")
    print("=" * 50)
    print(f"  Diagnosis   : {result['diagnosis']}")
    print(f"  Probability : {result['probability']:.1%}")
    print(f"  Risk Level  : {result['risk_level']}")
    print("=" * 50)
