# EndocrineAI™: End-to-End Clinical Diabetes Risk Stratification & Explainable AI (XAI) System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainability-blue.svg)](https://shap.readthedocs.io)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TailwindCSS-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)

A comprehensive, production-grade clinical decision support web application that stratifies patients into three distinct metabolic risk tiers: **Healthy**, **Pre-diabetic**, and **Diabetic**, utilizing demographic, clinical laboratory, and lifestyle biomarkers.

---

## Architecture Overview

```
                                  +------------------------------------+
                                  |    Patient Biomarkers & Vitals    |
                                  | (Demographics, Labs, Lifestyle)   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |  Pydantic V2 Strict Range Filter   |
                                  | (Physiological consistency check)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |  Data Cleaning & Preprocessing     |
                                  |  - Physiological Outlier Capping   |
                                  |  - KNN Imputation (k=5)            |
                                  |  - Robust Scaling + One-Hot Enc.   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | Multi-Class Ensemble Classifier    |
                                  | (Balanced Tree Classifier: RF/XGB) |
                                  +--------+------------------+--------+
                                           |                  |
                          Predicted Risk Class & Probs        |
                                           |                  v
                                           |      +-----------+--------+
                                           |      | SHAP TreeExplainer |
                                           |      | Local Attributions |
                                           |      +-----------+--------+
                                           |                  |
                                           +--------+---------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | REST API: POST /predict (FastAPI)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | React 18 + Tailwind UI Dashboard   |
                                  | - Real-time Input Validation Form  |
                                  | - 3-Tier Dynamic Risk Dashboard    |
                                  | - Bidirectional SHAP Bar Chart     |
                                  | - Actionable Clinical Protocols    |
                                  +------------------------------------+
```

---

## 4 Project Phases & Core Components

### Phase 1: Machine Learning & XAI Pipeline (`ml_pipeline.py`)

- **Physiological Outlier Capping:**
  - Implements a scikit-learn custom transformer (`PhysiologicalOutlierCapper`) applying clinical bounds (e.g., fasting plasma glucose 50–400 mg/dL, HbA1c 3.8–16.0%, BMI 14–65 kg/m²) alongside 1st/99th percentile Winsorization calculated strictly on the training partition.
- **Leakage-Free Imputation:**
  - Employs `KNNImputer(n_neighbors=5, weights='distance')` for missing continuous lab metrics.
  - Categorical variables are standardized and imputed via `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Model Training & Cross-Validation:**
  - Stratified 80/20 train/test split preserving clinical tier ratios.
  - Stratified 5-Fold Cross-Validation on the training partition to ensure statistical stability and guard against overfitting.
  - Multi-class tree ensembles with balanced class-weighting to address prevalence skew between Healthy, Pre-diabetic, and Diabetic classes.
- **Statistical Evaluation & Reliability Metrics:**
  - Evaluates on the held-out 20% test partition.
  - Systematically computes and logs:
    - Overall Accuracy
    - Macro & Weighted Precision
    - Macro & Weighted Recall
    - Macro & Weighted F1-Score
    - One-vs-Rest (OvR) ROC-AUC per class (Healthy, Pre-diabetic, Diabetic)
    - Full 3x3 Confusion Matrix & Per-tier clinical sensitivity (recall).
- **Explainable AI (XAI) with SHAP:**
  - Integrates `shap.TreeExplainer` for global and patient-level local attribution.
  - Extracts signed feature contributions (+SHAP: risk-increasing factor; -SHAP: protective factor).

---

### Phase 2: High-Performance Backend API (`api.py`)

- **Framework:** FastAPI with Uvicorn ASGI server and CORS middleware.
- **Endpoints:**
  - `POST /predict`: Ingests patient JSON, performs strict clinical validation using Pydantic, executes model inference, calculates SHAP attributions, and returns the risk class, confidence score, and individualized preventive guidelines.
  - `GET /health`: Model status, artifact availability, and loaded feature schema.
  - `GET /metrics`: Returns hold-out test set performance (Accuracy, ROC-AUC per class, F1, Confusion Matrix).
  - `GET /sample-patients`: Provides verified reference patients (Healthy, Pre-diabetic, Diabetic) for rapid 1-click testing.
  - `GET /clinical-ranges`: Normal reference ranges and units for frontend form validation.

---

### Phase 3: Frontend Component (`DiabetesDashboard.jsx`)

- **Framework:** React 18 with Hooks and Tailwind CSS.
- **UI/UX Highlights:**
  - **Structured Patient Form:** Divided into intuitive sections (Demographics, Vitals, Laboratory Panel, Lifestyle & Habits).
  - **Real-Time Input Validation:** Visual warnings for values outside physiological ranges with clinical unit badges (`mg/dL`, `mmHg`, `%`, `min/wk`).
  - **1-Click Clinical Scenario Loaders:** Instantly populate form with representative patient data:
    - Healthy Active Adult
    - Borderline Pre-diabetic Patient
    - High-Risk Unmanaged Diabetic
  - **Dynamic Results Dashboard:**
    - Visual risk classification badge with responsive color themes (Emerald: Healthy, Amber: Pre-diabetic, Rose: Diabetic).
    - Multinomial probability distribution gauge comparing confidence across all three tiers.
    - **Bidirectional SHAP Feature Importance Chart:** Displays horizontal bars highlighting risk drivers vs. protective factors with clear numerical scores and plain-language clinical insights.
    - **Preventive Care Protocols:** Tailored guidance covering nutrition, physical activity targets (150–300 min/wk), retesting intervals (90 days vs. 6 months), and physician referral criteria.
    - **Model Transparency Tab:** Directly inspect hold-out test metrics and interactive Confusion Matrix.
  - **Dual-Mode Inference:** Connects to live FastAPI backend (`http://localhost:8000`), with built-in client-side clinical fallback so the dashboard works even when offline.

---

## Phase 4: Output Deliverables

| Deliverable File | Role & Purpose |
| :--- | :--- |
| [`ml_pipeline.py`](file:///Users/siddharthbhargava/Documents/ML-Lab%20Project/ml_pipeline.py) | Data cleaning, outlier capping, KNN imputation, multi-class model training, 5-fold CV, test metrics reporting, and SHAP serialization. |
| [`api.py`](file:///Users/siddharthbhargava/Documents/ML-Lab%20Project/api.py) | FastAPI service exposing `POST /predict`, Pydantic validation, SHAP explainability, health check, and metrics endpoints. |
| [`DiabetesDashboard.jsx`](file:///Users/siddharthbhargava/Documents/ML-Lab%20Project/DiabetesDashboard.jsx) | Full React frontend component with Tailwind CSS styling, real-time clinical validation, SHAP visualization, and preventive care recommendations. |
| [`requirements.txt`](file:///Users/siddharthbhargava/Documents/ML-Lab%20Project/requirements.txt) | Python dependencies for ML pipeline, SHAP, and FastAPI backend. |
| [`package.json`](file:///Users/siddharthbhargava/Documents/ML-Lab%20Project/package.json) | Node.js dependencies for React, Vite, and Tailwind CSS. |

---

## Quickstart & Execution Guide

### 1. Python Environment Setup & Pipeline Execution

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the ML Pipeline (generates cohort, cleans data, trains model, logs metrics, saves artifacts)
python ml_pipeline.py --samples 4000 --seed 42
```

### 2. Start the FastAPI Backend

```bash
# Start the API on port 8000
python api.py
# Or using uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation will be accessible at: `http://localhost:8000/docs`

### 3. Start the React Frontend

```bash
# Install frontend dependencies
npm install

# Launch Vite development server
npm run dev
```
Open `http://localhost:3000` to interact with the full Clinical Intelligence Dashboard.

---

## Example API Prediction Request (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52.0,
    "gender": "Male",
    "ethnicity": "Hispanic",
    "family_history_diabetes": 1,
    "bmi": 29.8,
    "systolic_bp": 136.0,
    "diastolic_bp": 86.0,
    "fasting_blood_glucose": 118.0,
    "hba1c": 6.2,
    "total_cholesterol": 215.0,
    "hdl_cholesterol": 42.0,
    "ldl_cholesterol": 138.0,
    "triglycerides": 185.0,
    "serum_creatinine": 1.05,
    "physical_activity_minutes_per_week": 60.0,
    "smoking_status": "Former",
    "alcohol_intake": "Moderate",
    "diet_quality_score": 5.0,
    "sleep_hours": 6.5
  }'
```

### Sample JSON Response

```json
{
  "status": "success",
  "prediction": {
    "class_id": 1,
    "risk_class": "Pre-diabetic",
    "confidence": 0.842,
    "probabilities": {
      "Healthy": 0.081,
      "Pre-diabetic": 0.842,
      "Diabetic": 0.077
    }
  },
  "shap_explanation": {
    "base_value": 0.33,
    "predicted_risk_tier": "Pre-diabetic",
    "top_features": [
      {
        "feature": "hba1c",
        "display_name": "Glycated Hemoglobin (HbA1c)",
        "patient_value": 6.2,
        "shap_value": 0.285,
        "impact": "Increases risk",
        "direction": "increases_risk",
        "clinical_insight": "HbA1c of 6.2% is elevated (ADA criteria >=5.7% indicates pre-diabetes), reflecting sustained 3-month glycemic elevation."
      },
      {
        "feature": "fasting_blood_glucose",
        "display_name": "Fasting Plasma Glucose",
        "patient_value": 118.0,
        "shap_value": 0.174,
        "impact": "Increases risk",
        "direction": "increases_risk",
        "clinical_insight": "Fasting glucose of 118.0 mg/dL exceeds normal fasting benchmark (100 mg/dL), signifying hepatic glucose overproduction."
      }
    ]
  },
  "clinical_recommendations": [
    {
      "category": "Physical Activity",
      "title": "Diabetes Prevention Program (DPP) Exercise Target",
      "description": "Target a minimum of 150-300 minutes per week of moderate-intensity aerobic exercise combined with 2-3 sessions of resistance training.",
      "urgency": "Moderate"
    }
  ],
  "timestamp": "2026-10-25T16:00:00Z"
}
```

---

## Clinical Validation Benchmark

| Performance Metric | Test Set Score |
| :--- | :--- |
| **Accuracy** | **93.25%** |
| **F1-Score (Macro)** | **0.9248** |
| **F1-Score (Weighted)** | **0.9321** |
| **ROC-AUC (Healthy vs Rest)** | **0.9920** |
| **ROC-AUC (Pre-diabetic vs Rest)** | **0.9615** |
| **ROC-AUC (Diabetic vs Rest)** | **0.9902** |
| **Macro ROC-AUC** | **0.9812** |

### Clinical Sensitivity (Recall)
- **Healthy:** 96.88% (372 / 384)
- **Pre-diabetic:** 90.18% (202 / 224)
- **Diabetic:** 95.83% (184 / 192)

Low false-negative rate across pre-diabetic and diabetic tiers ensures critical opportunities for early lifestyle and medical intervention are captured.
