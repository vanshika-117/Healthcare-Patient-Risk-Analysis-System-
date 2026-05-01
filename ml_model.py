"""
================================================================================
  Healthcare Risk Analysis — Machine Learning Module
  Random Forest + Logistic Regression | scikit-learn
================================================================================
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "age", "bmi", "bp_systolic", "bp_diastolic",
    "cholesterol", "glucose", "smoker", "exercise_days"
]

FEATURE_LABELS = {
    "age"           : "Age",
    "bmi"           : "BMI",
    "bp_systolic"   : "Systolic BP",
    "bp_diastolic"  : "Diastolic BP",
    "cholesterol"   : "Cholesterol",
    "glucose"       : "Glucose",
    "smoker"        : "Smoker",
    "exercise_days" : "Exercise Days/Week",
}

RISK_ORDER = ["Healthy", "Low Risk", "Moderate Risk", "High Risk"]

MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Train Random Forest and Logistic Regression classifiers.
    Returns a results dict with models, metrics, and interpretation data.
    """
    X = df[FEATURES].values
    le = LabelEncoder()
    le.fit(RISK_ORDER)                         # fix class order
    y = le.transform(df["risk_category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Random Forest
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=None,
        min_samples_split=4, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred    = rf.predict(X_test)
    rf_proba   = rf.predict_proba(X_test)
    rf_acc     = accuracy_score(y_test, rf_pred)
    rf_cv      = cross_val_score(rf, X, y, cv=StratifiedKFold(5), scoring="accuracy").mean()
    rf_cm      = confusion_matrix(y_test, rf_pred, labels=range(len(RISK_ORDER)))
    rf_report  = classification_report(
        y_test, rf_pred, target_names=le.classes_, output_dict=True
    )

    # ── Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)
    lr_pred    = lr.predict(X_test_sc)
    lr_proba   = lr.predict_proba(X_test_sc)
    lr_acc     = accuracy_score(y_test, lr_pred)
    lr_cv      = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0),
        scaler.fit_transform(X), y,
        cv=StratifiedKFold(5), scoring="accuracy"
    ).mean()
    lr_cm      = confusion_matrix(y_test, lr_pred, labels=range(len(RISK_ORDER)))
    lr_report  = classification_report(
        y_test, lr_pred, target_names=le.classes_, output_dict=True
    )

    # ── Feature importances
    importances = dict(zip(FEATURES, rf.feature_importances_))

    # ── AUC (one-vs-rest, macro)
    y_bin = label_binarize(y_test, classes=range(len(RISK_ORDER)))
    rf_auc = roc_auc_score(y_bin, rf_proba, multi_class="ovr", average="macro")
    lr_auc = roc_auc_score(y_bin, lr_proba, multi_class="ovr", average="macro")

    # ── Save models
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH,   "wb") as f: pickle.dump(rf,     f)
    with open(SCALER_PATH,  "wb") as f: pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le,     f)

    return {
        # Models
        "rf"            : rf,
        "lr"            : lr,
        "scaler"        : scaler,
        "label_encoder" : le,

        # Test split
        "X_test"  : X_test,
        "y_test"  : y_test,

        # Random Forest metrics
        "rf_accuracy"    : round(rf_acc, 4),
        "rf_cv_accuracy" : round(rf_cv,  4),
        "rf_auc"         : round(rf_auc, 4),
        "rf_cm"          : rf_cm,
        "rf_report"      : rf_report,
        "rf_proba"       : rf_proba,

        # Logistic Regression metrics
        "lr_accuracy"    : round(lr_acc, 4),
        "lr_cv_accuracy" : round(lr_cv,  4),
        "lr_auc"         : round(lr_auc, 4),
        "lr_cm"          : lr_cm,
        "lr_report"      : lr_report,
        "lr_proba"       : lr_proba,

        # Interpretation
        "feature_importances" : importances,
        "classes"             : RISK_ORDER,
        "n_train"             : len(X_train),
        "n_test"              : len(X_test),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load saved RF model, scaler, and encoder."""
    with open(MODEL_PATH,   "rb") as f: rf     = pickle.load(f)
    with open(SCALER_PATH,  "rb") as f: scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le     = pickle.load(f)
    return rf, scaler, le


def predict_patient(patient: dict, rf, le) -> dict:
    """
    Predict risk for a single patient dict.
    Returns category, probabilities, and recommendations.
    """
    x = np.array([[patient[f] for f in FEATURES]])
    proba     = rf.predict_proba(x)[0]
    pred_idx  = np.argmax(proba)
    category  = le.inverse_transform([pred_idx])[0]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    prob_dict = {cls: round(float(p) * 100, 1)
                 for cls, p in zip(le.classes_, proba)}

    recs = []
    if patient["bmi"] > 30:
        recs.append("Weight management programme recommended (BMI > 30).")
    if patient["bp_systolic"] > 140:
        recs.append("Consult physician — blood pressure is elevated.")
    if patient["cholesterol"] > 240:
        recs.append("Review dietary fat intake; cholesterol is high.")
    if patient["glucose"] > 125:
        recs.append("Diabetes screening advised (fasting glucose elevated).")
    if patient["smoker"]:
        recs.append("Smoking cessation strongly recommended.")
    if patient["exercise_days"] < 3:
        recs.append("Increase physical activity to at least 3 days/week.")

    return {
        "category"        : category,
        "confidence"      : confidence,
        "probabilities"   : prob_dict,
        "recommendations" : recs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from analysis import generate_patient_data, clean_and_engineer
    print("Training models...")
    df = clean_and_engineer(generate_patient_data(500))
    results = train_models(df)

    print(f"\n{'='*50}")
    print("  MODEL COMPARISON")
    print(f"{'='*50}")
    print(f"  Random Forest   | Acc: {results['rf_accuracy']:.1%}  CV: {results['rf_cv_accuracy']:.1%}  AUC: {results['rf_auc']:.3f}")
    print(f"  Logistic Reg.   | Acc: {results['lr_accuracy']:.1%}  CV: {results['lr_cv_accuracy']:.1%}  AUC: {results['lr_auc']:.3f}")
    print(f"\n  Top features by importance:")
    for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {FEATURE_LABELS[feat]:<20} {bar}  {imp:.3f}")
    print(f"\n  Models saved to ./models/")