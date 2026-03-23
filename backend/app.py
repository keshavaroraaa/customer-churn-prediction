"""
app.py — Flask Backend for Customer Churn Prediction System
Course : VUIP111 Artificial Intelligence — Major Project
Endpoint: POST /predict
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s"
)
logger = logging.getLogger(__name__)

# ── App init ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Allow all origins (frontend on file:// or a different port).
# For production, replace "*" with your actual frontend origin.
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Model loading ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

def load_model(path: str):
    """Load the pickled ML model at startup and raise clearly if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Ensure 'churn_model.pkl' is placed inside the /model directory."
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully from: %s", path)
    return model

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as exc:
    logger.critical(exc)
    model = None          # server still starts; /predict returns a clear error

# ── Feature configuration ───────────────────────────────────────────────────
# These must exactly match the column names used during model training.
FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "TechSupport",
    "PaperlessBilling",
]

# ── Helper: parse & validate incoming JSON ───────────────────────────────────
def parse_input(data: dict) -> pd.DataFrame:
    """
    Accept both camelCase (from the frontend) and the canonical names.
    Returns a single-row DataFrame ready for model.predict().
    Raises ValueError with a human-readable message on bad input.
    """
    # Flexible key aliases so the frontend's snake_case still works
    alias = {
        "monthly_charges":  "MonthlyCharges",
        "total_charges":    "TotalCharges",
        "contract_type":    "Contract",
        "contract":         "Contract",
        "tech_support":     "TechSupport",
        "paperless_billing":"PaperlessBilling",
    }

    # Normalise keys
    normalised = {}
    for key, value in data.items():
        canonical = alias.get(key.lower(), key)
        normalised[canonical] = value

    # Check all required features are present
    missing = [col for col in FEATURE_COLUMNS if col not in normalised]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    # Type coercion for numeric fields
    try:
        normalised["tenure"]         = float(normalised["tenure"])
        normalised["MonthlyCharges"] = float(normalised["MonthlyCharges"])
        normalised["TotalCharges"]   = float(normalised["TotalCharges"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Numeric fields must be valid numbers. Detail: {exc}") from exc

    # Validate numeric ranges
    if normalised["tenure"] < 0:
        raise ValueError("'tenure' must be ≥ 0.")
    if normalised["MonthlyCharges"] < 0 or normalised["TotalCharges"] < 0:
        raise ValueError("Charge values must be ≥ 0.")

    # Validate categorical fields
    allowed = {
        "Contract":         {"Month-to-month", "One year", "Two year"},
        "TechSupport":      {"Yes", "No"},
        "PaperlessBilling": {"Yes", "No"},
    }
    for field, valid_values in allowed.items():
        if normalised[field] not in valid_values:
            raise ValueError(
                f"Invalid value '{normalised[field]}' for '{field}'. "
                f"Allowed: {sorted(valid_values)}"
            )

    row = {col: normalised[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    """Simple health-check so you can confirm the server is alive."""
    status = "ready" if model is not None else "degraded — model not loaded"
    return jsonify({"service": "ChurnGuard AI Backend", "status": status}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
        {
            "tenure":           24,
            "monthly_charges":  1499.0,
            "total_charges":    35976.0,
            "contract_type":    "Month-to-month",
            "tech_support":     "Yes",
            "paperless_billing":"No"
        }

    Response (JSON):
        {
            "prediction":  "High Risk" | "Low Risk",
            "probability": 0.85,
            "confidence":  "85.00%"
        }
    """
    # Guard: model unavailable
    if model is None:
        return jsonify({
            "error": "Model is not loaded. Check that 'model/churn_model.pkl' exists."
        }), 503

    # Guard: must be JSON
    if not request.is_json:
        return jsonify({
            "error": "Request Content-Type must be 'application/json'."
        }), 415

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body is empty or malformed JSON."}), 400

    # Validate & build DataFrame
    try:
        df = parse_input(data)
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    # Inference
    try:
        # Probability of the positive class (churn = 1)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df)[0][1])
        else:
            # Fallback for models without probability support
            proba = float(model.predict(df)[0])

        prediction_label = "High Risk" if proba >= 0.5 else "Low Risk"

        response = {
            "prediction":  prediction_label,
            "probability": round(proba, 4),
            "confidence":  f"{proba * 100:.2f}%",
        }

        logger.info(
            "Prediction: %s | Probability: %.4f | Input: %s",
            prediction_label, proba, data
        )
        return jsonify(response), 200

    except Exception as exc:                          # noqa: BLE001
        logger.error("Inference failed: %s", exc, exc_info=True)
        return jsonify({
            "error": "Prediction failed. Ensure the input features match model training."
        }), 500


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
