# EndocrineAI: Clinical Diabetes Risk Stratification & Explainable AI (Educational Prototype)

[![Python 3.13](https://img.shields.io/badge/Python-3.13.3-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%200.54.1-blue.svg)](https://shap.readthedocs.io)
[![React 18](https://img.shields.io/badge/Frontend-React%20%2B%20TailwindCSS-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Tests Passing](https://img.shields.io/badge/Tests-14%2F14%20Passing-success.svg)](./tests/test_pipeline_and_api.py)

An educational machine learning web application that classifies patients into three metabolic risk categories: **Healthy**, **Pre-diabetic**, and **Diabetic**, utilizing demographic, clinical laboratory, and lifestyle biomarkers with verified SHAP local explanations.

> **Important Clinical & Educational Disclaimer**  
> This software is an **educational research prototype**. It is trained and benchmarked on **synthetic data** generated from clinical guideline distributions. It is **not** a diagnostic medical device, has **not** undergone clinical trials or regulatory approval (FDA, CE-IVD), and **must not** be used to direct patient care or clinical decision-making.

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
                                  |  - Robust Scaling (Median / IQR)   |
                                  |  - KNN Imputation (Distance, k=5)  |
                                  |  - One-Hot Categorical Encoding    |
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
                                  | - Real-time Input Validation Form  |
                                  | - Truthful status & Error Retries  |
                                  | - Relative SHAP Direction Badges   |
                                  | - Explicit Rule-Based Demo Toggle  |
                                  +------------------------------------+
```

---

## Key Methodological Improvements & Corrections

### 1. Correct SHAP Local Explanations
- **Class-Relative Attribution:** Contributions are labeled relative to the predicted class: *"Supports Healthy prediction"* vs. *"Opposes Healthy prediction"*. A positive SHAP contribution for a Healthy classification correctly indicates support for healthy status, eliminating previous mislabeling where positive values were uniformly labeled "Increases risk".
- **Dynamic Expected Base Value:** Dynamically extracts `explainer.expected_value[class_id]` from the explainer object instead of relying on a hardcoded constant (e.g. 0.33).
- **Explicit Output Space & Additivity Verification:** Explanations operate in probability space. Every inference validates that $\text{base\_value} + \sum \text{SHAP} \approx P(\text{predicted\_class})$ within 0.05 tolerance. The sum of omitted features is explicitly reported.
- **Holdout-Free Background Distribution:** The SHAP background summarizer samples strictly from the training partition (`X_train`), preventing holdout test set leakage.
- **One-Hot Feature Aggregation:** Attributions across dummy variables (e.g., `ethnicity_Asian`, `ethnicity_Caucasian`) are aggregated back to original clinical feature identities.
- **Explicit Failure State:** If SHAP fails or is unavailable, the API returns `status: "unavailable"`. Hand-written fake formulas are never returned.

### 2. Truthful Operational Status & Error Reporting
- **No Silent Fallbacks:** Backend network or pipeline errors display the actual error state with a **"Retry Assessment"** action, rather than silently substituting simulated predictions.
- **Explicit Rule-Based Demo Mode:** Simulation is available strictly as an opt-in mode via the **"Rule Demo Mode"** toggle, prominently labeled *"Rule-based demo — not a trained model prediction"*. Demo responses return `confidence: null`, `probabilities: null`, and `shap_explanation: unavailable`.
- **Response Metadata:** Every prediction response includes provenance metadata (`prediction_source: "trained_model" | "rule_based_demo"`, `model_version`, `dataset_type`, `explanation_status`).
- **Clean Initial State:** Prevents simulated demo assessments from remaining on-screen once the live backend connects.

### 3. Transparent Data Provenance & Synthetic Framing
- **Synthetic Benchmark Cohort:** The default dataset (`artifacts/diabetes_clinical_dataset.csv`, 4,000 synthetic patient records) is generated using latent physiological risk models guided by American Diabetes Association (ADA) criteria.
- **Label-Biomarker Synthesis Mechanism:** The generator assigns metabolic risk first, and key diagnostic biomarkers (fasting blood glucose and HbA1c) are subsequently generated conditioned on that risk tier. High test accuracy (99.00%) is an inherent property of this synthetic generation procedure and **must not be presented as evidence of real-world clinical performance**.
- **Inspection of Bundled AIM-94 Dataset:** The bundled dataset in `./diabetes/Diabetes-Data/` contains 70 patient time-series records from 1994 tracking daily insulin dosages and blood glucose logs. It lacks cross-sectional demographics, complete lipid panels, renal biomarkers, lifestyle indicators, and 3-class cross-sectional target labels. It was **not** used to train this model.
- **Strict Data Ingestion:** Explicitly supplying an invalid `--data` path raises a `FileNotFoundError` immediately, rather than silently generating synthetic data.

### 4. Strengthened ML Evaluation & Comparative Baselines
- **Preprocessing Order:** Numeric features undergo `PhysiologicalOutlierCapper` -> `RobustScaler` -> `KNNImputer(5, weights='distance')` so that Euclidean distance during KNN imputation is evaluated on normalized scales rather than raw units.
- **Holdout Integrity:** A strict 80/20 stratified split is performed. Cross-validation (Stratified 5-Fold) is executed solely on `X_train`. The final test set (`X_test`, 800 patients) remains untouched until evaluation.
- **Multiple Comparative Baselines:** The primary model is evaluated against three baselines on the same test set:
  1. *Dummy Majority Baseline* (always predicts Healthy)
  2. *Logistic Regression Baseline* (balanced class weighting)
  3. *ADA Threshold Rule Heuristic* (standard clinical cutoffs: FBG $\ge 126$ / HbA1c $\ge 6.5$ for Diabetic; FBG $\ge 100$ / HbA1c $\ge 5.7$ for Pre-diabetic)
- **Probability Calibration:** Reports test calibration log-loss (0.0947).

### 5. Concrete Implementation Fixes
- **Vectorized Smoking Effect:** Corrected cohort scalar bug `(0.35 if "Current" in smoking else 0.0)` to an individual per-patient vector: `0.35 * (smoking == "Current").astype(float)`.
- **Exact Blood Pressure Boundary:** Form and Pydantic validation enforces that Systolic BP must be at least 15 mmHg higher than Diastolic BP (`systolic_bp >= diastolic_bp + 15`), verified by boundary test cases.
- **Missing Lab Value Inference:** The API accepts `Optional[float] = None` for laboratory values; missing inputs are safely imputed via the trained pipeline's `KNNImputer` and recorded in `metadata.imputed_features`.
- **Configurable API URL:** Frontend uses `import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'` instead of hardcoded localhost strings.

---

## Comprehensive Evaluation Results

All metrics below are derived directly from the generated evaluation artifact [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) produced on the held-out 800-sample test partition (80/20 stratified split of 4,000 synthetic records).

### Model vs. Baseline Comparison (Hold-Out Test Set)

| Model / Baseline | Accuracy | Macro F1 | Description / Formulation |
| :--- | :---: | :---: | :--- |
| **Dummy Majority Baseline** | 48.00% | 0.2162 | Always predicts the most frequent training class (Healthy) |
| **ADA Clinical Rule Heuristic** | 93.25% | 0.9239 | Deterministic ADA threshold cutoffs (FBG $\ge 100/126$, HbA1c $\ge 5.7/6.5$) |
| **Logistic Regression** | 96.88% | 0.9658 | Standard linear multi-class model with balanced class weights |
| **Random Forest (Proposed)** | **99.00%** | **0.9880** | Balanced ensemble with leakage-free preprocessing pipeline |

> **Resolving Prior Metric Inconsistencies:**  
> The previous README incorrectly cited 93.25% as the model accuracy (which was actually the performance of the ADA Rule Heuristic baseline) while displaying prediction counts corresponding to 94.75%. The true hold-out test set accuracy of the trained Random Forest pipeline is **99.00%** (792 correct out of 800 test samples).

### Primary Model Test Performance (Random Forest)

| Performance Metric | Test Score | Notes / Parameters |
| :--- | :---: | :--- |
| **Overall Accuracy** | **99.00%** | 792 / 800 correct test predictions |
| **Macro Precision** | **0.9880** | Unweighted mean across all 3 classes |
| **Macro Recall** | **0.9880** | Unweighted mean sensitivity |
| **Macro F1-Score** | **0.9880** | Harmonic mean of macro precision & recall |
| **Weighted F1-Score** | **0.9900** | Weighted by class sample distribution |
| **Calibration Log Loss** | **0.0947** | Multinomial cross-entropy on test probabilities |
| **Macro ROC-AUC (OvR)** | **0.9995** | Area under the multi-class ROC curve |
| **Healthy ROC-AUC** | **0.9999** | One-vs-Rest ROC-AUC |
| **Pre-diabetic ROC-AUC** | **0.9991** | One-vs-Rest ROC-AUC |
| **Diabetic ROC-AUC** | **0.9994** | One-vs-Rest ROC-AUC |
| **5-Fold CV Accuracy (Train)** | **98.84%** $\pm$ 0.21% | Evaluated strictly on `X_train` (3,200 samples) |
| **5-Fold CV Macro F1 (Train)** | **0.9865** $\pm$ 0.25% | Cross-validation stability |

### Hold-Out Test Confusion Matrix (N = 800)

```
                     Predicted: Healthy    Predicted: Pre-diabetic    Predicted: Diabetic      Total Actual
Actual: Healthy              383                      1                         0                   384
Actual: Pre-diabetic           1                    220                         3                   224
Actual: Diabetic               0                      3                       189                   192
Total Predicted              384                    224                       192                   800
```

- **Healthy Tier:** Sensitivity (Recall) = **99.74%** (383/384), Precision = **99.74%** (383/384)
- **Pre-diabetic Tier:** Sensitivity (Recall) = **98.21%** (220/224), Precision = **98.21%** (220/224)
- **Diabetic Tier:** Sensitivity (Recall) = **98.44%** (189/192), Precision = **98.44%** (189/192)

---

## Project Structure & Deliverables

| Deliverable File | Description & Link |
| :--- | :--- |
| [`ml_pipeline.py`](./ml_pipeline.py) | End-to-end training pipeline, data generation, 5-fold CV, baseline evaluations, SHAP serialization. |
| [`clinical_transformers.py`](./clinical_transformers.py) | Standalone `PhysiologicalOutlierCapper` for clean joblib pickling/unpickling across services. |
| [`api.py`](./api.py) | FastAPI REST service with Pydantic v2 validation, verified SHAP explanations, and explicit demo endpoints. |
| [`DiabetesDashboard.jsx`](./DiabetesDashboard.jsx) | React 18 UI with relative SHAP direction badges, truth-in-status indicators, and configurable API URL. |
| [`tests/test_pipeline_and_api.py`](./tests/test_pipeline_and_api.py) | Automated test suite covering schemas, boundaries, SHAP additivity, and KNN missing imputation. |
| [`scripts/update_readme_metrics.py`](./scripts/update_readme_metrics.py) | Reproducible script synchronizing README benchmark numbers from evaluation artifacts. |
| [`requirements.txt`](./requirements.txt) | Pinned Python 3.13 dependency manifest. |
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

### 2. Run the Machine Learning Pipeline

```bash
# Executes synthetic cohort generation, 5-fold CV, baseline comparisons, and saves artifacts
python ml_pipeline.py --samples 4000 --seed 42 --model rf
```

*Artifacts generated in `./artifacts/`:*
- `diabetes_clinical_dataset.csv` (Cohort dataset)
- `model_pipeline.joblib` (Trained pipeline: Capper + RobustScaler + KNNImputer + OneHotEncoder + RandomForest)
- `explainer.joblib` (SHAP TreeExplainer fitted on training data sample)
- `model_metrics.json` (Test evaluation metrics)
- `metadata.json` (Pipeline metadata and feature dictionary)

### 3. Run Automated Tests

Execute the 14-point test suite covering schemas, boundaries, missing lab imputation, and SHAP additivity:

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
    "age": 52.0,
    "gender": "Male",
    "ethnicity": "Hispanic",
    "family_history_diabetes": 1,
    "bmi": 29.8,
    "systolic_bp": 136.0,
    "diastolic_bp": 86.0,
    "fasting_blood_glucose": 116.0,
    "hba1c": 6.1,
    "total_cholesterol": 218.0,
    "hdl_cholesterol": 41.0,
    "ldl_cholesterol": 138.0,
    "triglycerides": 195.0,
    "serum_creatinine": 1.05,
    "physical_activity_minutes_per_week": 60.0,
    "smoking_status": "Former",
    "alcohol_intake": "Moderate",
    "diet_quality_score": 4.8,
    "sleep_hours": 6.2
  }'
```

### Truthful JSON Response Structure

```json
{
  "status": "success",
  "metadata": {
    "prediction_source": "trained_model",
    "model_version": "1.1.0",
    "model_type": "RandomForestClassifier",
    "dataset_type": "synthetic",
    "explanation_status": "available",
    "explanation_method": "TreeSHAP (Holdout-free background)",
    "imputed_features": [],
    "disclaimer": "Educational prototype evaluated on synthetic benchmark data. Not for clinical diagnosis."
  },
  "prediction": {
    "class_id": 1,
    "risk_class": "Pre-diabetic",
    "confidence": 0.88,
    "probabilities": {
      "Healthy": 0.05,
      "Pre-diabetic": 0.88,
      "Diabetic": 0.07
    }
  },
  "shap_explanation": {
    "status": "available",
    "explained_class": "Pre-diabetic",
    "explained_class_id": 1,
    "output_space": "probability",
    "base_value": 0.28,
    "top_features": [
      {
        "feature": "hba1c",
        "display_name": "Glycated Hemoglobin (HbA1c)",
        "patient_value": 6.1,
        "shap_value": 0.32,
        "impact": "Supports Pre-diabetic prediction (+32.0%)",
        "direction": "supports",
        "clinical_insight": "HbA1c of 6.1% is elevated (≥5.7%), supporting the Pre-diabetic classification."
      },
      {
        "feature": "fasting_blood_glucose",
        "display_name": "Fasting Plasma Glucose",
        "patient_value": 116.0,
        "shap_value": 0.22,
        "impact": "Supports Pre-diabetic prediction (+22.0%)",
        "direction": "supports",
        "clinical_insight": "Fasting glucose of 116.0 mg/dL is above normal fasting thresholds, supporting Pre-diabetic."
      }
    ],
    "omitted_features_shap_sum": 0.06,
    "additivity_check": {
      "base_value": 0.28,
      "all_features_shap_sum": 0.60,
      "reconstructed_output": 0.88,
      "model_output": 0.88,
      "absolute_difference": 0.0,
      "is_additive": true
    }
  },
  "clinical_recommendations": [
    {
      "category": "Nutrition & Diet",
      "title": "Medical Nutrition Therapy & Dietary Modification",
      "description": "Emphasize Mediterranean dietary pattern: high dietary fiber (≥14g/1000 kcal), complex carbohydrates with low glycemic index.",
      "urgency": "Moderate"
    }
  ],
  "timestamp": "2026-10-25T16:45:00Z"
}
```

---

## Limitations & Intended Use

1. **Synthetic Data Origin:** The dataset used to train and test this pipeline is synthetic. Labels were assigned by latent metabolic risk simulation, and clinical biomarkers were subsequently derived. While it mirrors ADA clinical distributions, high test accuracy cannot be extrapolated to clinical populations.
2. **Absence of External Clinical Validation:** The model has not been validated on real electronic health records (EHR) from diverse healthcare systems. It has not been benchmarked across demographic sub-populations with varying disease prevalences.
3. **Cross-Sectional Scope:** The classifier categorizes current metabolic tier (Healthy, Pre-diabetic, Diabetic). It does **not** model longitudinal disease progression, rate of beta-cell decline, or time-to-conversion.
4. **Intended Use:** Designed exclusively for educational instruction, algorithmic demonstration, and research into interpretable clinical machine learning.
