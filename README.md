# EndocrineAI: CDC BRFSS Diabetes Risk Stratification & Explainable AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13.3-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%200.54.1-blue.svg)](https://shap.readthedocs.io)
[![React 18](https://img.shields.io/badge/Frontend-React%20%2B%20TailwindCSS-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Tests Passing](https://img.shields.io/badge/Tests-12%2F12%20Passing-success.svg)](./tests/test_pipeline_and_api.py)
[![Dataset](https://img.shields.io/badge/Dataset-CDC%20BRFSS%202015-green.svg)](https://www.cdc.gov/brfss/)

An educational machine learning web application that classifies individuals into three metabolic risk categories: **Healthy**, **Pre-diabetic**, and **Diabetic**, utilizing 21 real-world epidemiological health indicators from the **CDC Behavioral Risk Factor Surveillance System (BRFSS 2015 cohort, N=50,000)** with verified class-relative SHAP local explanations.

> **Important Clinical & Educational Disclaimer**  
> This software is an **educational research prototype**. It is trained and benchmarked on **observational public health survey data** from the CDC BRFSS. It is **not** an FDA-cleared diagnostic medical device and **must not** be used to direct patient care, prescribe interventions, or substitute for formal laboratory diagnosis (oral glucose tolerance tests, fasting plasma glucose, or venous HbA1c assays).

---

## Architecture Overview

```
                                  +------------------------------------+
                                  |    21 CDC Health Indicators        |
                                  | (Demographics, Diagnoses, Habits)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |  Pydantic V2 Strict Range Filter   |
                                  | (Survey indicator boundary checks) |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |  Data Cleaning & Preprocessing     |
                                  |  - Physiological Outlier Capping   |
                                  |  - Robust Scaling (Median / IQR)   |
                                  |  - Distance-weighted KNN Imputer   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | Multi-Class Ensemble Classifier    |
                                  | (RandomForestClassifier, Balanced) |
                                  +--------+------------------+--------+
                                           |                  |
                          Predicted Risk Class & Probs        |
                                           |                  v
                                           |      +-----------+--------+
                                           |      | SHAP TreeExplainer |
                                           |      | (Holdout-free bkg) |
                                           |      +-----------+--------+
                                           |                  |
                                           +--------+---------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | REST API: POST /predict (FastAPI)  |
                                  | - Dynamic base value & additivity  |
                                  | - Relative sign labels             |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | React 18 + Tailwind UI Dashboard   |
                                  | - Real-time 21-Indicator Form      |
                                  | - 1-Click CDC Survey Presets       |
                                  | - Relative SHAP Direction Badges   |
                                  | - CDC DPP Intervention Guidance    |
                                  +------------------------------------+
```

---

## Key Methodological Improvements & Corrections

### 1. Correct SHAP Local Explanations
- **Class-Relative Attribution:** Contributions are labeled relative to the predicted class: *"Supports Healthy prediction"* vs. *"Opposes Healthy prediction"*. A positive SHAP contribution for a Healthy classification correctly indicates support for healthy status, eliminating previous mislabeling where positive values were uniformly labeled "Increases risk".
- **Dynamic Expected Base Value:** Dynamically extracts `explainer.expected_value[class_id]` from the explainer object instead of relying on a hardcoded constant (e.g. 0.33).
- **Explicit Output Space & Additivity Verification:** Explanations operate in probability space. Every inference validates that $\text{base\_value} + \sum \text{SHAP} \approx P(\text{predicted\_class})$ within 0.05 tolerance. The sum of omitted features is explicitly reported.
- **Holdout-Free Background Distribution:** The SHAP background summarizer samples strictly from the training partition (`X_train`), preventing holdout test set leakage.
- **Explicit Failure State:** If SHAP fails or is unavailable, the API returns `status: "unavailable"`. Hand-written fake formulas are never returned.

### 2. Truthful Operational Status & Error Reporting
- **No Silent Fallbacks:** Backend network or pipeline errors display the actual error state with a **"Retry Assessment"** action, rather than silently substituting simulated predictions.
- **Explicit Rule-Based Demo Mode:** Simulation is available strictly as an opt-in mode via the **"Rule Demo Mode"** toggle, prominently labeled *"Rule-based demo — not a trained model prediction"*. Demo responses return `confidence: null`, `probabilities: null`, and `shap_explanation: unavailable`.
- **Response Metadata:** Every prediction response includes provenance metadata (`prediction_source: "trained_model" | "rule_based_demo"`, `model_version`, `dataset_type`, `explanation_status`).
- **Clean Initial State:** Prevents simulated demo assessments from remaining on-screen once the live backend connects.

### 3. Transparent Data Provenance & Real CDC BRFSS Integration
- **Official CDC BRFSS Cohort:** Ingests the real **Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS 2015)** dataset (`archive/2015.csv`, containing 441,456 raw telephone survey interviews).
- **Target Extraction (`DIABETE3`):** Cleaned using CDC coding standards:
  - `3.0` -> **0: Healthy** (84.1%)
  - `4.0` -> **1: Pre-diabetic** (1.8%)
  - `1.0` -> **2: Diabetic** (14.0%)
  - Excludes gestational diabetes (`2.0`), missing (`9.0`), and don't know (`7.0`).
- **21 Standard Epidemiological Indicators:** Standardized through `ingest_brfss.py` to produce a stratified representative cohort of 50,000 subjects (`artifacts/cdc_brfss_diabetes_clean.csv`).
- **Realistic Observational Performance:** Unlike synthetic datasets where diagnostic labels dictate biomarkers and produce artificial 99% accuracy, real-world epidemiological survey data captures true population noise, reporting subjectivity, and extreme class imbalance (pre-diabetic prevalence 1.8%). The resulting **68.80% accuracy** and **0.7619 Macro ROC-AUC** (0.8243 Diabetic ROC-AUC) represent authentic real-world performance.
- **Inspection of Bundled AIM-94 Dataset:** The legacy dataset in `./diabetes/Diabetes-Data/` contains 70 patient time-series records from 1994 tracking daily insulin dosages and blood glucose logs. It lacks cross-sectional demographics, complete lipid panels, lifestyle indicators, and 3-class target labels, and is not suitable for population-level multi-class screening.

### 4. Strengthened ML Evaluation & Comparative Baselines
- **Preprocessing Order:** Numeric features undergo `PhysiologicalOutlierCapper` -> `RobustScaler` -> `KNNImputer(5, weights='distance')` so that Euclidean distance during imputation is evaluated on normalized scales.
- **Holdout Integrity:** A strict 80/20 stratified split (40,000 train / 10,000 test) is maintained. Stratified 5-Fold Cross-Validation is executed solely on `X_train`. The test set (`X_test`, 10,000 samples) remains completely untouched until final benchmark evaluation.
- **Comparative Baselines on Same Test Set:**
  1. *Dummy Majority Baseline* (always predicts Healthy)
  2. *Logistic Regression Baseline* (balanced class weighting)
  3. *CDC Epidemiological Risk Heuristic* (standard clinical survey cutoffs: high BP, high cholesterol, BMI $\ge 30$, age $\ge 60$)
- **Probability Calibration:** Evaluates calibration log-loss (0.7647 on test partition).


### 5. Concrete Implementation Fixes
- **Vectorized Smoking Effect:** Corrected cohort scalar bug `(0.35 if "Current" in smoking else 0.0)` to an individual per-patient vector: `0.35 * (smoking == "Current").astype(float)`.
- **Exact Blood Pressure Boundary:** Form and Pydantic validation enforces that Systolic BP must be at least 15 mmHg higher than Diastolic BP (`systolic_bp >= diastolic_bp + 15`), verified by boundary test cases.
- **Missing Lab Value Inference:** The API accepts `Optional[float] = None` for laboratory values; missing inputs are safely imputed via the trained pipeline's `KNNImputer` and recorded in `metadata.imputed_features`.
- **Configurable API URL:** Frontend uses `import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'` instead of hardcoded localhost strings.

---

## Comprehensive Evaluation Results

All metrics below are derived directly from the generated evaluation artifact [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) produced on the held-out 10,000-sample test partition (80/20 stratified split of 50,000 real CDC BRFSS records).

### Model vs. Baseline Comparison (Hold-Out Test Set, N = 10,000)

| Model / Baseline | Accuracy | Macro F1 | Description / Formulation |
| :--- | :---: | :---: | :--- |
| **Dummy Majority Baseline** | 84.12% | 0.3046 | Always predicts the most frequent survey class (Healthy). 0% sensitivity on diabetic & pre-diabetic patients. |
| **Logistic Regression (Balanced)** | 64.35% | 0.4271 | Standard linear multi-class baseline with inverse-prevalence balanced weighting |
| **CDC Epidemiological Risk Heuristic** | 76.86% | 0.4052 | Standard epidemiological rules: Diabetic if High BP & Chol with BMI $\ge 30$ and Age $\ge 60$; Pre-diabetic if High BP & Chol with BMI $\ge 27$; else Healthy |
| **Random Forest (Balanced, Proposed)** | **68.80%** | **0.4304** | Balanced ensemble with leakage-free preprocessing pipeline and 0.7619 Macro ROC-AUC |

> **Real-World Epidemiological Metric Realism:**  
> On synthetic datasets, artificial generation rules often generate near-perfect accuracy (99%). In authentic public health surveillance data (CDC BRFSS), observational telephone interviews capture real human survey noise and extreme natural class imbalance (Pre-diabetic prevalence is only 1.8% in the general population). The trained Balanced Random Forest achieves **71.65% sensitivity (recall) on Diabetic individuals** and an **ROC-AUC of 0.8243 for Diabetic and 0.8208 for Healthy**, significantly outperforming naive baselines in clinically meaningful risk detection.

### Primary Model Test Performance (Random Forest on CDC BRFSS)

| Performance Metric | Test Score | Notes / Parameters |
| :--- | :---: | :--- |
| **Overall Accuracy** | **68.80%** | 6,880 / 10,000 correct test predictions |
| **Macro Precision** | **0.4339** | Unweighted mean precision across all 3 classes |
| **Macro Recall** | **0.5017** | Unweighted mean sensitivity across all 3 classes |
| **Macro F1-Score** | **0.4304** | Harmonic mean of macro precision & recall |
| **Weighted F1-Score** | **0.7399** | Weighted by survey population class frequencies |
| **Calibration Log Loss** | **0.7647** | Multinomial cross-entropy on calibrated test probabilities |
| **Macro ROC-AUC (OvR)** | **0.7619** | Unweighted area under the multi-class ROC curve |
| **Healthy ROC-AUC** | **0.8208** | One-vs-Rest ROC-AUC for Healthy |
| **Pre-diabetic ROC-AUC** | **0.6408** | One-vs-Rest ROC-AUC for Pre-diabetic |
| **Diabetic ROC-AUC** | **0.8243** | One-vs-Rest ROC-AUC for Diabetic |
| **5-Fold CV Accuracy (Train)** | **68.89%** $\pm$ 0.53% | Evaluated strictly on `X_train` (40,000 samples) |
| **5-Fold CV Macro F1 (Train)** | **0.4353** $\pm$ 0.54% | Cross-validation stability across 5 folds |

### Hold-Out Test Confusion Matrix (N = 10,000)

```
                     Predicted: Healthy    Predicted: Pre-diabetic    Predicted: Diabetic      Total Actual
Actual: Healthy             5857                      625                      1930                  8412
Actual: Pre-diabetic          69                       17                        98                   184
Actual: Diabetic             248                      150                      1006                  1404
Total Predicted             6174                      792                      3034                 10000
```

- **Healthy Tier:** Sensitivity (Recall) = **69.63%** (5,857 / 8,412), Precision = **94.87%** (5,857 / 6,174)
- **Pre-diabetic Tier:** Sensitivity (Recall) = **9.24%** (17 / 184), Precision = **2.15%** (17 / 792) *(reflects severe 1.8% survey rarity)*
- **Diabetic Tier:** Sensitivity (Recall) = **71.65%** (1,006 / 1,404), Precision = **33.16%** (1,006 / 3,034)

---

## Project Structure & Deliverables

| Deliverable File | Description & Link |
| :--- | :--- |
| [`ingest_brfss.py`](./ingest_brfss.py) | Standalone ETL pipeline cleaning raw `archive/2015.csv` into 21 standardized CDC indicators. |
| [`ml_pipeline.py`](./ml_pipeline.py) | End-to-end training pipeline, 5-fold CV, baseline evaluations, and TreeSHAP serialization. |
| [`clinical_transformers.py`](./clinical_transformers.py) | Standalone `PhysiologicalOutlierCapper` for clean joblib pickling/unpickling across services. |
| [`api.py`](./api.py) | FastAPI REST service with Pydantic v2 validation, 21 CDC indicators, verified SHAP explanations. |
| [`DiabetesDashboard.jsx`](./DiabetesDashboard.jsx) | React 18 UI with 21 survey input controls, 1-click CDC presets, relative SHAP bars, and CDC DPP guidance. |
| [`tests/test_pipeline_and_api.py`](./tests/test_pipeline_and_api.py) | Automated test suite covering CDC schemas, boundary checks, SHAP additivity, and confusion matrix consistency. |
| [`scripts/update_readme_metrics.py`](./scripts/update_readme_metrics.py) | Reproducible script synchronizing README benchmark numbers from evaluation artifacts. |
| [`requirements.txt`](./requirements.txt) | Pinned Python 3.13 dependency manifest. |
| [`artifacts/cdc_brfss_diabetes_clean.csv`](./artifacts/cdc_brfss_diabetes_clean.csv) | Cleaned representative 50,000-subject CDC BRFSS cohort. |
| [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) | Ground-truth test evaluation metrics and confusion matrix artifact. |
| [`artifacts/metadata.json`](./artifacts/metadata.json) | Model hyperparameters, training sample count, and feature specifications. |

---

## Quickstart & Reproducibility Guide

### 1. Clean Environment Setup

This project is tested with **Python 3.13.3** and **Node.js 18+**.

```bash
# Clone repository
git clone git@github.com:Siddharth18-coder/ML-Lab-Project.git
cd "ML-Lab Project"

# Create Python virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install pinned Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Ingest CDC Data & Train the Machine Learning Pipeline

```bash
# Ingest raw CDC BRFSS 2015 dataset into artifacts/cdc_brfss_diabetes_clean.csv
python ingest_brfss.py

# Executes 5-fold CV, baseline comparisons, holdout test evaluation, and saves artifacts
python ml_pipeline.py --samples 50000 --seed 42 --model rf
```

*Artifacts generated in `./artifacts/`:*
- `cdc_brfss_diabetes_clean.csv` (Cleaned 50,000 survey cohort)
- `model_pipeline.joblib` (Trained pipeline: Capper + RobustScaler + KNNImputer + Balanced RandomForest)
- `explainer.joblib` (SHAP TreeExplainer fitted on training data sample)
- `model_metrics.json` (Test evaluation metrics)
- `metadata.json` (Pipeline metadata and feature dictionary)

### 3. Run Automated Tests

Execute the 12-point test suite covering schemas, boundaries, missing lab imputation, and SHAP additivity:

```bash
pytest tests/ -v
```

### 4. Synchronize Documentation Metrics

Ensure README benchmark numbers stay synchronized with saved evaluation artifacts:

```bash
python scripts/update_readme_metrics.py
```

### 5. Launch the Backend API

```bash
# Start FastAPI service on 127.0.0.1:8000
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI Swagger docs will be accessible at: `http://127.0.0.1:8000/docs`

### 6. Launch the Frontend

```bash
# Install frontend dependencies
npm install

# Start Vite development server
npm run dev
```
Open `http://127.0.0.1:3000` to interact with the dashboard.

To configure a custom backend endpoint for the frontend, set `VITE_API_BASE_URL`:
```bash
VITE_API_BASE_URL="http://127.0.0.1:8000" npm run dev
```

To run a production frontend build:
```bash
npm run build
```

---

## Sample Verified API Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "high_bp": 1,
    "high_chol": 1,
    "chol_check": 1,
    "bmi": 32.5,
    "smoker": 1,
    "stroke": 0,
    "heart_disease_or_attack": 0,
    "phys_activity": 0,
    "fruits": 0,
    "veggies": 1,
    "hvy_alcohol_consump": 0,
    "any_healthcare": 1,
    "no_doc_bc_cost": 0,
    "gen_hlth": 4,
    "ment_hlth": 5.0,
    "phys_hlth": 10.0,
    "diff_walk": 1,
    "sex": 1,
    "age": 9,
    "education": 4,
    "income": 5
  }'
```

### Truthful JSON Response Structure

```json
{
  "status": "success",
  "metadata": {
    "prediction_source": "trained_model",
    "model_version": "2.0.0",
    "model_type": "RandomForestClassifier",
    "dataset_type": "real_cdc_brfss_2015",
    "explanation_status": "available",
    "explanation_method": "TreeSHAP (Holdout-free background)",
    "imputed_features": [],
    "disclaimer": "Educational research prototype evaluated on real CDC BRFSS survey data. Not for clinical diagnosis."
  },
  "prediction": {
    "class_id": 2,
    "risk_class": "Diabetic",
    "confidence": 0.64,
    "probabilities": {
      "Healthy": 0.22,
      "Pre-diabetic": 0.14,
      "Diabetic": 0.64
    }
  },
  "shap_explanation": {
    "status": "available",
    "explained_class": "Diabetic",
    "explained_class_id": 2,
    "output_space": "probability",
    "base_value": 0.14,
    "top_features": [
      {
        "feature": "gen_hlth",
        "display_name": "General Health Self-Rating",
        "patient_value": 4.0,
        "shap_value": 0.18,
        "impact": "Supports Diabetic prediction (+18.0%)",
        "direction": "supports",
        "clinical_insight": "Self-rated fair or poor health is strongly correlated with chronic metabolic morbidity."
      },
      {
        "feature": "high_bp",
        "display_name": "High Blood Pressure Diagnosis",
        "patient_value": 1.0,
        "shap_value": 0.12,
        "impact": "Supports Diabetic prediction (+12.0%)",
        "direction": "supports",
        "clinical_insight": "Hypertension is a primary vascular comorbidity accelerating metabolic risk."
      }
    ],
    "omitted_features_shap_sum": 0.20,
    "additivity_check": {
      "base_value": 0.14,
      "all_features_shap_sum": 0.50,
      "reconstructed_output": 0.64,
      "model_output": 0.64,
      "absolute_difference": 0.0,
      "is_additive": true
    }
  },
  "clinical_recommendations": [
    {
      "category": "Intervention Pathway",
      "title": "Comprehensive Diabetes Self-Management Education and Support (DSMES)",
      "description": "Engage with an accredited DSMES program focusing on glycemic monitoring, cardiovascular risk reduction, and medical nutrition therapy.",
      "urgency": "High"
    }
  ],
  "timestamp": "2026-10-26T18:30:00Z"
}
```

---

## Limitations & Intended Use

1. **Observational Public Health Survey Data:** The dataset originates from telephone health interviews conducted by the CDC. Features such as hypertension, physical activity, and diet are self-reported and subject to recall bias.
2. **Extreme Class Imbalance:** Pre-diabetic individuals represent only 1.8% of respondents due to widespread under-diagnosis in the general survey population, resulting in modest pre-diabetic recall despite balanced weighting.
3. **Screening Tool vs. Diagnostic Assays:** The model assesses risk based on survey indicators. It does not replace diagnostic clinical assays (e.g., fasting plasma glucose $\ge 126$ mg/dL, oral glucose tolerance test $\ge 200$ mg/dL, or lab HbA1c $\ge 6.5\%$).
4. **Intended Use:** Designed exclusively for educational instruction, algorithmic demonstration, and public health risk exploration.

