# 🫀 Heart Disease Risk Prediction — ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready** machine learning pipeline for predicting heart disease risk from patient health data. Features an Autoencoder for feature learning, 5 ML classifiers with comprehensive evaluation, and a Flask web application for real-time predictions.

---

## 📁 Project Structure

```
project_ml_healthcare/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── heart_disease_synthetic_dataset.csv
│   ├── processed/                    # Cleaned & transformed data
│   └── train_test/                   # Train/test split files
│
├── notebooks/                        # Jupyter notebooks for EDA
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── utils.py                      # Utilities, config, logging, I/O
│   ├── preprocessing.py              # Data cleaning, imputation, scaling
│   ├── feature_engineering.py        # Domain features & Autoencoder
│   ├── train_model.py                # Model training & selection
│   ├── evaluate_model.py             # Metrics, plots, reports
│   └── predict.py                    # Inference pipeline
│
├── models/                           # Saved models & artifacts
│   ├── best_model.pkl
│   ├── feature_scaler.pkl
│   └── autoencoder_encoder.keras
│
├── app/
│   └── app.py                        # Flask web application
│
├── reports/
│   ├── figures/                      # Generated visualizations
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance_*.png
│   │   └── metrics_comparison.png
│   ├── model_comparison.csv
│   └── classification_reports.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Dataset

**Heart Disease Synthetic Dataset** — 10,000+ patient records with 29 features:

| Category | Features |
|----------|----------|
| **Demographics** | age, gender, height_cm, weight_kg, bmi |
| **Vitals** | systolic_bp, diastolic_bp, resting_heart_rate, oxygen_saturation |
| **Blood Work** | glucose, insulin, total_cholesterol, hdl, ldl, triglycerides, hemoglobin |
| **Lifestyle** | smoking_status, smoking_years, alcohol_consumption, physical_activity_level, sleep_hours, stress_level, diet_quality |
| **Medical History** | diabetes, hypertension, family_history_heart_disease |
| **Risk Scores** | cardiac_risk_score, metabolic_syndrome_score |
| **Target** | heart_disease (0 = No, 1 = Yes) |

---

## 🧠 Pipeline Architecture

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
                │                    │                   │              │
                ├── Missing Values   ├── Domain Features ├── LR, RF    ├── Confusion Matrix
                ├── Outlier Capping  ├── Autoencoder     ├── XGBoost   ├── ROC Curves
                ├── Encoding         │   (TensorFlow)    ├── SVM       ├── Feature Importance
                └── RobustScaler     └── Latent Repr.    └── MLP       └── Metrics Report
```

### Preprocessing
- **Missing values**: KNN imputation (k=5) for numerics, mode for categoricals
- **Outlier handling**: IQR-based capping (factor=3.0)
- **Encoding**: LabelEncoder for categorical features
- **Scaling**: RobustScaler (robust to outliers)
- **Split**: 80/20 stratified train/test

### Feature Engineering
- **Domain features**: pulse pressure, mean arterial pressure, cholesterol ratios, BMI categories, cardiovascular index, lifestyle score, comorbidity count
- **Autoencoder** (TensorFlow/Keras): Input→64→32→16→32→64→Input architecture for unsupervised feature learning

### Models
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear model with L2 regularization |
| **Random Forest** | 300 trees with balanced class weights |
| **XGBoost** | Gradient boosting with tuned hyperparameters |
| **SVM** | RBF kernel with probability calibration |
| **MLP Classifier** | 3-layer neural network (128→64→32) |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 5-Fold Stratified Cross-Validation
- Confusion matrices, ROC curves, feature importance charts

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd project_ml_healthcare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Option A: Run each step individually
python -m src.preprocessing       # Preprocess data
python -m src.train_model         # Train all models
python -m src.evaluate_model      # Evaluate & generate reports

# Option B: Run the notebook (recommended for exploration)
jupyter notebook notebooks/
```

### 3. Launch the Web App

```bash
python app/app.py
# Open: http://localhost:5000
```

---

## 🔧 Google Colab Setup

```python
# Mount Drive (if data is on Drive)
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -r requirements.txt

# Run pipeline
import sys
sys.path.insert(0, '/content/project_ml_healthcare')

from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.train_model import train_all_models, select_best_model
from src.evaluate_model import run_evaluation

# Execute
X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline()
X_train_final, X_test_final = run_feature_engineering(X_train, X_test)
trained_models = train_all_models(X_train_final, y_train)
best_name, best_model = select_best_model(trained_models)
results = run_evaluation(trained_models, X_test_final, y_test, X_train_final.columns.tolist())
```

---

## 📊 Sample Results

After running the pipeline, you'll find in `reports/`:

| File | Description |
|------|-------------|
| `model_comparison.csv` | Side-by-side metrics for all models |
| `cv_results.csv` | Cross-validation results |
| `classification_reports.json` | Detailed per-class metrics |
| `model_selection.json` | Best model metadata |
| `figures/confusion_matrices.png` | Confusion matrices |
| `figures/roc_curves.png` | ROC curves |
| `figures/feature_importance_*.png` | Feature importance charts |
| `figures/metrics_comparison.png` | Metrics comparison bar chart |

---

## 🌐 Web Application

The Flask app provides:
- **Interactive form** with all patient features
- **Real-time prediction** with probability and risk level
- **Auto BMI calculation** from height and weight
- **REST API** endpoint at `/api/predict` for programmatic access
- **Health check** at `/health`

### API Usage

```python
import requests

patient = {
    "age": 55, "gender": 1, "height_cm": 170, "weight_kg": 85,
    "bmi": 29.4, "systolic_bp": 145, "diastolic_bp": 92,
    "resting_heart_rate": 78, "oxygen_saturation": 96.5,
    "glucose": 110, "insulin": 18, "total_cholesterol": 240,
    "hdl": 38, "ldl": 150, "triglycerides": 200, "hemoglobin": 14.5,
    "smoking_status": 1, "smoking_years": 20, "alcohol_consumption": 2,
    "physical_activity_level": 0, "sleep_hours": 5.5, "stress_level": 7,
    "diet_quality": 2, "diabetes": 1, "hypertension": 1,
    "family_history_heart_disease": 1, "cardiac_risk_score": 55,
    "metabolic_syndrome_score": 42
}

response = requests.post("http://localhost:5000/api/predict", json=patient)
print(response.json())
# {"prediction": 1, "probability": 0.87, "risk_level": "Very High", "diagnosis": "Heart Disease Detected"}
```

---

## 🛠 Technical Details

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |

---

## 📜 License

This project is for **educational and research purposes only**. Not intended for clinical use.

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

<p align="center">
  Built with ❤️ for healthcare ML research
</p>
"# projet_ml" 
"# projet_ml" 
"# projet_ml" 
