"""
app.py - Flask Web Application for Heart Disease Risk Prediction
=================================================================
A production-ready web interface for real-time heart disease risk
prediction with a responsive, modern UI.
"""

import os
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS

from src.predict import HeartDiseasePrediction
from src.utils import setup_logger

# ─────────────────────── App Setup ───────────────────────────────────

app = Flask(__name__)
CORS(app)
logger = setup_logger("flask_app")

# Initialize predictor (lazy load)
predictor = None


def get_predictor():
    """Lazy-load the prediction model."""
    global predictor
    if predictor is None:
        predictor = HeartDiseasePrediction()
    return predictor


# ─────────────────────── HTML Template ───────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Risk Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #e74c3c;
            --primary-dark: #c0392b;
            --primary-light: #f5b7b1;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border: #475569;
            --shadow: rgba(0, 0, 0, 0.3);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            background-image:
                radial-gradient(at 20% 50%, rgba(231, 76, 60, 0.08) 0%, transparent 50%),
                radial-gradient(at 80% 20%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                radial-gradient(at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        /* Form Card */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            box-shadow: 0 4px 20px var(--shadow);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .form-group input, .form-group select {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.15);
        }

        .form-group input::placeholder {
            color: var(--text-secondary);
            opacity: 0.5;
        }

        /* Button */
        .btn-predict {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            margin-top: 1.5rem;
            transition: all 0.3s ease;
            letter-spacing: 0.5px;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(231, 76, 60, 0.35);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        .btn-predict:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Result */
        .result-card {
            display: none;
            animation: slideUp 0.5s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card.show { display: block; }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }

        .result-diagnosis {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .result-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .risk-low { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .risk-moderate { background: rgba(245, 158, 11, 0.15); color: var(--accent-yellow); border: 1px solid rgba(245, 158, 11, 0.3); }
        .risk-high { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }
        .risk-veryhigh { background: rgba(239, 68, 68, 0.25); color: #ff6b6b; border: 1px solid rgba(239, 68, 68, 0.5); }

        .probability-bar {
            width: 100%;
            height: 12px;
            background: var(--bg-input);
            border-radius: 6px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .probability-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 1s ease;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-yellow), var(--accent-red));
        }

        .probability-text {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }

        .probability-label {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        /* Disclaimer */
        .disclaimer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.75rem;
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid var(--border);
        }

        /* Error Message */
        .error-msg {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--accent-red);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            display: none;
        }

        .error-msg.show { display: block; }

        /* Responsive */
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr 1fr; }
            .header h1 { font-size: 1.5rem; }
            .result-header { flex-direction: column; gap: 0.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-icon">🫀</div>
            <h1>Heart Disease Risk Prediction</h1>
            <p>AI-powered cardiovascular risk assessment using machine learning</p>
        </div>

        <!-- Input Form -->
        <form id="predictionForm">
            <!-- Demographics -->
            <div class="card">
                <div class="card-title">👤 Informations du Patient</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Âge (ans)</label>
                        <input type="number" name="age" value="55" min="18" max="120" required>
                    </div>
                    <div class="form-group">
                        <label>Genre</label>
                        <select name="gender">
                            <option value="0">Femme</option>
                            <option value="1" selected>Homme</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Taille (cm)</label>
                        <input type="number" name="height_cm" id="heightInput" value="170" step="0.1" min="100" max="250">
                    </div>
                    <div class="form-group">
                        <label>Poids (kg)</label>
                        <input type="number" name="weight_kg" id="weightInput" value="85" step="0.1" min="30" max="300">
                    </div>
                    <div class="form-group">
                        <label>IMC <span style="color:var(--accent-blue);font-size:0.7rem">(auto)</span></label>
                        <input type="number" name="bmi" id="bmiInput" value="29.4" step="0.1" min="10" max="70" readonly style="opacity:0.75;cursor:not-allowed">
                    </div>
                </div>
            </div>

            <!-- Vitals -->
            <div class="card">
                <div class="card-title">💓 Signes Vitaux & Bilan Sanguin</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>TA Systolique (mmHg)</label>
                        <input type="number" name="systolic_bp" value="145" min="60" max="300">
                    </div>
                    <div class="form-group">
                        <label>TA Diastolique (mmHg)</label>
                        <input type="number" name="diastolic_bp" value="92" min="30" max="200">
                    </div>
                    <div class="form-group">
                        <label>Glycémie (mg/dL)</label>
                        <input type="number" name="glucose" value="110" step="0.1" min="20" max="500">
                    </div>
                    <div class="form-group">
                        <label>Cholestérol Total</label>
                        <input type="number" name="total_cholesterol" value="240" min="50" max="500">
                    </div>
                </div>
            </div>

            <!-- Lifestyle & History -->
            <div class="card">
                <div class="card-title">🏥 Mode de Vie & Antécédents</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Tabagisme</label>
                        <select name="smoking_status">
                            <option value="0">Jamais</option>
                            <option value="1" selected>Fumeur actuel</option>
                            <option value="2">Ex-fumeur</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Activité Physique</label>
                        <select name="physical_activity_level">
                            <option value="0" selected>Sédentaire</option>
                            <option value="1">Légère</option>
                            <option value="2">Modérée</option>
                            <option value="3">Active</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Diabète</label>
                        <select name="diabetes">
                            <option value="0">Non</option>
                            <option value="1" selected>Oui</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Hypertension</label>
                        <select name="hypertension">
                            <option value="0">Non</option>
                            <option value="1" selected>Oui</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Antécédents Familiaux</label>
                        <select name="family_history_heart_disease">
                            <option value="0">Non</option>
                            <option value="1" selected>Oui</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" class="btn-predict" id="predictBtn">
                🫀 Prédire le Risque Cardiaque
            </button>
        </form>

        <!-- Error Message -->
        <div class="error-msg" id="errorMsg"></div>

        <!-- Result Card -->
        <div class="card result-card" id="resultCard">
            <div class="result-header">
                <div class="result-diagnosis" id="resultDiagnosis"></div>
                <div class="result-badge" id="resultBadge"></div>
            </div>
            <div class="probability-label">Probability of Heart Disease</div>
            <div class="probability-text" id="probText"></div>
            <div class="probability-bar">
                <div class="probability-fill" id="probFill" style="width: 0%"></div>
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
            ⚠️ This tool is for educational and research purposes only. It is NOT a substitute for professional medical advice.
            Always consult a qualified healthcare provider for medical decisions.
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const btn = document.getElementById('predictBtn');
            const errorMsg = document.getElementById('errorMsg');
            const resultCard = document.getElementById('resultCard');

            // Show loading
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Analyzing...';
            errorMsg.classList.remove('show');
            resultCard.classList.remove('show');

            // Collect form data
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                });

                const result = await response.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                // Display result
                const diagnosis = document.getElementById('resultDiagnosis');
                const badge = document.getElementById('resultBadge');
                const probText = document.getElementById('probText');
                const probFill = document.getElementById('probFill');

                diagnosis.textContent = result.diagnosis;
                diagnosis.style.color = result.prediction === 1 ? 'var(--accent-red)' : 'var(--accent-green)';

                const risk = result.risk_level.toLowerCase().replace(' ', '');
                badge.textContent = result.risk_level + ' Risk';
                badge.className = 'result-badge risk-' + risk;

                const prob = (result.probability * 100).toFixed(1);
                probText.textContent = prob + '%';
                probFill.style.width = prob + '%';

                resultCard.classList.add('show');
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

            } catch (error) {
                errorMsg.textContent = '❌ Error: ' + error.message;
                errorMsg.classList.add('show');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🫀 Predict Heart Disease Risk';
            }
        });

        // Auto-calculate BMI
        const heightInput = document.getElementById('heightInput');
        const weightInput = document.getElementById('weightInput');
        const bmiInput = document.getElementById('bmiInput');

        function updateBMI() {
            const h = parseFloat(heightInput.value) / 100;
            const w = parseFloat(weightInput.value);
            if (h > 0 && w > 0) {
                bmiInput.value = (w / (h * h)).toFixed(1);
            }
        }

        heightInput.addEventListener('input', updateBMI);
        weightInput.addEventListener('input', updateBMI);
    </script>
</body>
</html>
"""


# ─────────────────────── Routes ──────────────────────────────────────

@app.route("/")
def home():
    """Render the prediction form."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        pred = get_predictor()
        result = pred.predict_single(data)

        logger.info(
            f"Prediction: {result['diagnosis']} "
            f"(prob={result['probability']:.4f}, risk={result['risk_level']})"
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic access."""
    return predict()


# ─────────────────────── Run Server ──────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 50)
    print("  Heart Disease Risk Prediction App")
    print(f"  Running on: http://localhost:{port}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=port, debug=debug)
