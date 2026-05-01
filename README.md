# 🏥 Healthcare Patient Risk Analysis System

A full-stack ML-powered healthcare analytics platform that predicts patient risk categories using Random Forest and Logistic Regression, served through an interactive Plotly Dash dashboard and a Flask REST API.

---

## 📌 Overview

This project simulates an end-to-end clinical decision support system. It generates synthetic patient records, engineers clinical features, trains two ML classifiers, uses synthetic data simulating clinical records and exposes predictions through both an interactive web dashboard and a REST API — complete with personalised health recommendations.

---

## 🗂️ Project Structure

```
healthcare-risk-analysis/
│
├── analysis.py               # Data generation, feature engineering, statistics & matplotlib visualisation
├── ml_model.py               # Random Forest + Logistic Regression training, evaluation & inference
├── dashboard.py              # Plotly Dash interactive 4-tab dashboard
├── app.py                    # Flask REST API (4 endpoints)
├── patient_data_cleaned.csv  # Exported cleaned dataset (auto-generated)
├── requirements.txt          # Python dependencies
└── models/                   # Saved model artefacts (auto-generated)
    ├── rf_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

---

## ⚙️ Features

### 🔬 Machine Learning (`ml_model.py`)
- Trains **Random Forest** (150 estimators) and **Logistic Regression** classifiers
- **5-fold stratified cross-validation** for robust accuracy estimates
- **Multi-class AUC** scoring (one-vs-rest, macro average)
- Feature importance ranking across 8 clinical inputs
- Model persistence via `pickle` to `./models/`

### 📊 Interactive Dashboard (`dashboard.py`)
4-tab Plotly Dash interface:
| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards, risk donut chart, age histogram, BMI boxplot, cholesterol vs glucose scatter |
| **Model Comparison** | Accuracy, CV accuracy, AUC, confusion matrices for both models |
| **Risk Predictor** | Sliders → real-time RF inference → probability bar chart + personalised recommendations |
| **Patient Table** | Filterable, sortable table across all 500 patient records |

### 🌐 Flask REST API (`app.py`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Summary statistics + risk by age group |
| `GET` | `/api/patients` | Paginated patient list |
| `POST` | `/api/predict` | Rule-based risk prediction for a new patient |
| `GET` | `/api/risk-dist` | Risk category distribution with percentages |

### 🧪 Data Pipeline (`analysis.py`)
- Generates **500 synthetic patient records** with realistic clinical correlations
- **Feature engineering**: age groups, WHO BMI classification, hypertension flag, diabetes risk flag, pulse pressure
- Exports cleaned CSV + text summary report
- Produces a 7-panel matplotlib dashboard PNG

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/healthcare-risk-analysis.git
cd healthcare-risk-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dash Dashboard
```bash
python dashboard.py
```
Open [http://127.0.0.1:8050](http://127.0.0.1:8050)

### 4. Run the Flask API (separately)
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5. Run analysis + export reports only
```bash
python analysis.py
```
Outputs: `patient_data_cleaned.csv`, `summary_report.txt`, `healthcare_dashboard.png`

### 6. Train models standalone
```bash
python ml_model.py
```
Saves trained models to `./models/`

---

## 📡 API Usage

### `POST /api/predict`

**Request body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "risk_score": 9,
  "risk_category": "High Risk",
  "recommendations": [
    "Consult a physician about hypertension management.",
    "Review dietary fat intake; cholesterol is elevated.",
    "Fasting glucose is high — diabetes screening advised.",
    "Smoking cessation will significantly reduce your risk score.",
    "Aim for at least 150 min of moderate exercise per week."
  ]
}
```

---

## 🧠 Risk Classification Logic

Patient risk is scored on 8 weighted clinical criteria:

| Factor | Threshold | Points |
|--------|-----------|--------|
| Age | > 55 | +2 |
| BMI | > 30 | +2 |
| Smoker | Yes | +3 |
| Systolic BP | > 140 mmHg | +2 |
| Cholesterol | > 240 mg/dL | +1 |
| Glucose | > 125 mg/dL | +2 |
| Exercise | < 2 days/week | +1 |

| Total Score | Category |
|-------------|----------|
| ≥ 8 | 🔴 High Risk |
| 5 – 7 | 🟡 Moderate Risk |
| 2 – 4 | 🔵 Low Risk |
| 0 – 1 | 🟢 Healthy |

---

## 📦 Dependencies

```
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
flask >= 3.0
scikit-learn >= 1.3
plotly >= 5.18
dash >= 2.16
dash-bootstrap-components >= 1.5
```

---

## 📈 Model Performance (on 500-patient dataset)

| Model | Accuracy | CV Accuracy | AUC |
|-------|----------|-------------|-----|
| Random Forest | ~88% | ~86% | ~0.97 |
| Logistic Regression | ~82% | ~81% | ~0.93 |

> Metrics vary slightly each run due to synthetic data generation. Reported values are representative averages.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib, Plotly, Plotly Dash |
| Backend API | Flask |
| Model Storage | Pickle |

---

## 👩‍💻 Author

**Kayeetha Vanshika**  
---

## 📄 License

This project is intended for educational and portfolio purposes.
