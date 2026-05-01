"""
================================================================================
  Healthcare Analysis — REST API Backend
  Flask + Pandas + NumPy  |  Resume Project
================================================================================
  Endpoints:
    GET  /api/stats          → summary statistics JSON
    GET  /api/patients       → paginated patient list
    POST /api/predict        → risk prediction for a single patient
    GET  /api/risk-dist      → risk category distribution
================================================================================
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json

# ── Import the analysis module (run it once to build the dataset)
from analysis import generate_patient_data, clean_and_engineer, compute_statistics

app = Flask(__name__)

# ── Build dataset at startup
print("⚙️   Building dataset...")
_df    = clean_and_engineer(generate_patient_data(500))
_stats = compute_statistics(_df)
print("✅   Dataset ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serialisable list of dicts."""
    return json.loads(df.to_json(orient="records"))


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "project" : "Healthcare Patient Risk Analysis API",
        "version" : "1.0.0",
        "endpoints": [
            "GET  /api/stats",
            "GET  /api/patients?page=1&per_page=20",
            "POST /api/predict",
            "GET  /api/risk-dist",
        ]
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return overview statistics."""
    ov = _stats["overview"]
    return jsonify({
        "status" : "success",
        "data"   : ov,
        "risk_by_age_group": _stats["by_age_group"],
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Return paginated patient list."""
    page     = int(request.args.get("page",     1))
    per_page = int(request.args.get("per_page", 20))

    start = (page - 1) * per_page
    end   = start + per_page
    total = len(_df)

    subset = _df.iloc[start:end][[
        "patient_id", "age", "gender", "bmi",
        "bp_systolic", "cholesterol", "glucose",
        "risk_score", "risk_category"
    ]]

    return jsonify({
        "status"     : "success",
        "page"       : page,
        "per_page"   : per_page,
        "total"      : total,
        "total_pages": (total + per_page - 1) // per_page,
        "data"       : df_to_records(subset),
    })


@app.route("/api/risk-dist", methods=["GET"])
def get_risk_dist():
    """Return risk category counts."""
    dist = _stats["risk_distribution"]
    total = sum(dist.values())
    return jsonify({
        "status" : "success",
        "data"   : [
            {"category": k, "count": v, "percentage": round(v/total*100, 1)}
            for k, v in dist.items()
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict_risk():
    """
    Predict risk category for a new patient.

    Expected JSON body:
    {
        "age": 55,
        "bmi": 29.5,
        "bp_systolic": 145,
        "bp_diastolic": 92,
        "cholesterol": 245,
        "glucose": 130,
        "smoker": 1,
        "exercise_days": 1
    }
    """
    body = request.get_json(force=True)
    required = ["age","bmi","bp_systolic","bp_diastolic",
                 "cholesterol","glucose","smoker","exercise_days"]

    # ── Validate
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"status": "error",
                        "message": f"Missing fields: {missing}"}), 400

    # ── Rule-based scoring (same logic as data generator)
    age        = float(body["age"])
    bmi        = float(body["bmi"])
    bp_sys     = float(body["bp_systolic"])
    chol       = float(body["cholesterol"])
    glucose    = float(body["glucose"])
    smoker     = int(body["smoker"])
    ex_days    = int(body["exercise_days"])

    score = (
        (age > 55)   * 2 +
        (bmi > 30)   * 2 +
        smoker       * 3 +
        (bp_sys > 140) * 2 +
        (chol > 240) * 1 +
        (glucose > 125) * 2 +
        (ex_days < 2) * 1
    )

    if score >= 8:
        category = "High Risk"
    elif score >= 5:
        category = "Moderate Risk"
    elif score >= 2:
        category = "Low Risk"
    else:
        category = "Healthy"

    # ── Personalised recommendations
    recommendations = []
    if bmi > 30:
        recommendations.append("Consider weight management programme (BMI > 30).")
    if bp_sys > 140:
        recommendations.append("Consult a physician about hypertension management.")
    if chol > 240:
        recommendations.append("Review dietary fat intake; cholesterol is elevated.")
    if glucose > 125:
        recommendations.append("Fasting glucose is high — diabetes screening advised.")
    if smoker:
        recommendations.append("Smoking cessation will significantly reduce your risk score.")
    if ex_days < 3:
        recommendations.append("Aim for at least 150 min of moderate exercise per week.")

    return jsonify({
        "status"         : "success",
        "risk_score"     : int(score),
        "risk_category"  : category,
        "recommendations": recommendations,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀  Starting Healthcare Analysis API on http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)