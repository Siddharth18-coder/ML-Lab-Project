# EndocrineAI: CDC BRFSS Diabetes Risk Stratification & Explainable AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13.3-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%200.54.1-blue.svg)](https://shap.readthedocs.io)
[![React 18](https://img.shields.io/badge/Frontend-React%20%2B%20TailwindCSS-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Tests Passing](https://img.shields.io/badge/Tests-13%2F13%20Passing-success.svg)](./tests/test_pipeline_and_api.py)
[![Dataset](https://img.shields.io/badge/Dataset-CDC%20BRFSS%202015-green.svg)](https://www.cdc.gov/brfss/)

An educational machine learning web application that stratifies survey respondents into three self-reported glycemic categories: **No Reported Diabetes**, **Pre-diabetic**, and **Diabetic**, utilizing 21 epidemiological health indicators from the **U.S. CDC Behavioral Risk Factor Surveillance System (BRFSS 2015 cohort, N=50,000)** with mathematically verified class-relative TreeSHAP feature attributions.

> [!IMPORTANT]
> **Educational & Methodological Scope Disclaimer**  
> This software is an **academic student portfolio project**. It is trained and benchmarked on **observational cross-sectional survey data** from the CDC BRFSS. It estimates statistical associations with existing self-reported diabetes status, **not** a validated medical prediction of future diabetes onset. It is **not** an FDA-cleared diagnostic medical device and **must not** be used to direct clinical care, prescribe treatments, or substitute for formal laboratory diagnosis (fasting plasma glucose or venous HbA1c assays).

For comprehensive technical specifications, class distributions, and limitation analyses, see the [Model Card](./MODEL_CARD.md).

---

## Architecture Overview

```
                                  +------------------------------------+
                                  |    21 CDC Survey Indicators        |
                                  | (Demographics, Diagnoses, Habits)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |   Pydantic V2 Strict Validator     |
                                  | (Type coercion & boundary checks)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  |  Leakage-Free Preprocessing        |
                                  |  - Physiological Outlier Capping   |
                                  |  - Robust Scaling (Median / IQR)   |
                                  |  - Feature Name Propagation        |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | Multi-Class Ensemble Classifier    |
                                  | (RandomForestClassifier, Balanced) |
                                  +--------+------------------+--------+
                                           |                  |
                          Predicted Class & Probabilities     |
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
                                  | - Exact transformed feature order  |
                                  | - Dynamic base value & additivity  |
                                  | - Non-blocking async threadpool    |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | React 18 + Tailwind UI Dashboard   |
                                  | - Stale Result Invalidation        |
                                  | - Field-Level Validation (422)     |
                                  | - Relative SHAP Direction Badges   |
                                  | - Educational Lifestyle Pathways   |
                                  +------------------------------------+
```

---

## Key Methodological Improvements & Corrections

### 1. Correct SHAP Feature Attribution (Highest Priority Fix)
- **Resolved Feature Permutation:** Scikit-Learn's `ColumnTransformer` emits transformed numerical features first (`bmi`, `phys_hlth`, `ment_hlth`, `age`) followed by binary features (`high_bp`, `high_chol`, ...). Previously, the API mapped SHAP values using the raw input DataFrame order (where `high_bp` was first), mistakenly assigning BMI impacts to blood pressure even while additivity checks passed. This was resolved by dynamically propagating feature names via `preprocessor.get_feature_names_out()` and matching transformed outputs to original input columns.
- **Controlled Regression Test:** Added `test_shap_feature_identity_controlled_regression` in [`tests/test_pipeline_and_api.py`](./tests/test_pipeline_and_api.py) with distinct synthetic contributions, guaranteeing that any permuted feature mapping definitely fails the test suite.
- **Dynamic Expected Base Value:** Dynamically extracts `explainer.expected_value[class_idx]` from the explainer object corresponding to `classifier.classes_`, removing arbitrary fallback constants (e.g. 0.33).
- **Exact Additivity Verification:** Confirms that $\text{base\_value} + \sum \text{SHAP} \approx P(\text{predicted\_class})$ within a documented tolerance of 0.05.
- **Holdout-Free Background Distribution:** The SHAP background summarizer samples strictly from `X_train` (100 k-means clusters), eliminating test set leakage.
- **Explicit Unavailable State:** If explanations cannot be computed or fail additivity, the API returns an explicit `unavailable` status with an actionable reason.

### 2. Dual-Path Reproducibility from a Fresh Clone
The project supports two verified training workflows:
- **Path A (Bundled Cleaned Cohort):** Trains directly from [`artifacts/cdc_brfss_diabetes_clean.csv`](./artifacts/cdc_brfss_diabetes_clean.csv) (50,000 respondents, SHA-256 verified) without needing external raw files.
- **Path B (Raw CDC Survey Ingestion):** Ingests raw CDC BRFSS annual datasets (e.g., `archive/2015.csv`, 441,456 survey interviews) via `ingest_brfss.py`, applying exact row-filtering telemetry and logging rows removed at each cleaning stage.

### 3. Rigorous Evaluation Without Chasing Accuracy
- **Untouched Holdout Partition:** An 80/20 stratified split ($N = 10,000$ test partition) was strictly preserved before model selection.
- **Honest Pre-Diabetes Reporting:** Rather than hiding poor minority class performance, pre-diabetes recall (4.89%) and precision (2.36%) are documented as natural consequences of extreme class imbalance (1.84% prevalence) and symptom profile overlap.
- **Irreducible Bayes Error Analysis:** Discovered **457 duplicate survey rows** in the cohort, with **12 conflicting profile clusters** (identical survey responses with opposing diabetes outcomes), demonstrating that survey questions alone without laboratory blood biomarkers cannot achieve 100% accuracy.
- **Secondary Binary Screening Task:** Evaluated as a separate secondary screening task (0: No Reported Diabetes vs. 1: At-Risk [Pre-diabetic or Diabetic]), achieving **86.96% sensitivity** and **0.8208 ROC-AUC**.

### 4. Probability Calibration vs. TreeSHAP Tradeoff
- Evaluated Platt scaling (`CalibratedClassifierCV(method='sigmoid')`), which reduces multiclass Brier score from 0.3863 to 0.2234 and log loss from 0.6774 to 0.4007.
- Because Platt calibration wraps the ensemble in a non-linear sigmoid layer that **breaks mathematical TreeSHAP probability additivity**, the base ensemble probabilities are served to preserve exact additive explainability.

### 5. API and Frontend Reliability
- **Stale State Management:** When any input or preset changes, displayed results are immediately flagged as stale (`isStale = true`) to prevent users from mistaking old predictions for current inputs.
- **Structured Field-Level Errors:** Handles HTTP 422 validation errors with field-specific messages rendered directly below offending inputs.
- **Non-Blocking Inference:** CPU-intensive inference and TreeSHAP calculations execute in FastAPI's async threadpool via `run_in_threadpool`.

---

## Benchmark Evaluation Results

All metrics below are derived directly from [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) produced on the held-out 10,000-sample test partition (80/20 stratified split of 50,000 CDC BRFSS records).

### Model vs. Baseline Comparison (Hold-Out Test Set, N = 10,000)

| Model Architecture | Accuracy | Balanced Acc | Macro F1 | Brier Score | Log Loss | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Dummy Baseline** | 84.12% | 33.33% | 0.3046 | 0.3176 | 5.7237 | 0.003s | 0.07 µs |
| **Logistic Regression (Balanced)** | 64.35% | 51.64% | 0.4271 | 0.4875 | 0.8776 | 0.424s | 0.06 µs |
| **HistGradientBoosting** | 64.35% | 51.00% | 0.4235 | 0.4529 | 0.7973 | 0.797s | 2.22 µs |
| **Random Forest (Balanced, Selected)** | **71.58%** | **50.43%** | **0.4365** | **0.3863** | **0.6774** | **0.334s** | **2.45 µs** |
| **Project Risk Heuristic** | 76.86% | 42.33% | 0.4052 | 0.4628 | N/A | 0.000s | 0.10 µs |

### Per-Class Test Performance (Random Forest)

| Survey Class | Support | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC (OvR) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Reported Diabetes** | 8,412 | 72.68% | 94.10% | 0.8202 | 0.8208 | 0.9588 |
| **Pre-diabetic** | 184 | **4.89%** | **2.36%** | **0.0319** | 0.6386 | 0.0281 |
| **Diabetic** | 1,404 | **73.72%** | **33.15%** | **0.4574** | 0.8276 | 0.4209 |

### Hold-Out Confusion Matrix (N = 10,000)

```
                               Pred: No Reported Diabetes    Pred: Pre-diabetic    Pred: Diabetic      Total Actual
Actual: No Reported Diabetes                      6,114                       307             1,991              8,412
Actual: Pre-diabetic                                 79                         9                96                184
Actual: Diabetic                                    304                        65             1,035              1,404
Total Predicted                                   6,497                       381             3,122             10,000
```

---

## Project Structure & Deliverables

| Deliverable File | Description & Link |
| :--- | :--- |
| [`ingest_brfss.py`](./ingest_brfss.py) | Dual-path ETL script: validates bundled cohort (Path A) or ingests raw CDC BRFSS dataset (Path B). |
| [`ml_pipeline.py`](./ml_pipeline.py) | Training pipeline with feature name propagation, 5-fold CV, baseline comparison, and calibration assessment. |
| [`clinical_transformers.py`](./clinical_transformers.py) | Standalone schemas and `PhysiologicalOutlierCapper` with explicit `get_feature_names_out()`. |
| [`api.py`](./api.py) | FastAPI service with exact transformed feature mapping, structured 422 error handlers, and threadpool execution. |
| [`DiabetesDashboard.jsx`](./DiabetesDashboard.jsx) | React 18 UI with stale state tracking, field-level validation, and honest benchmark metrics display. |
| [`tests/test_pipeline_and_api.py`](./tests/test_pipeline_and_api.py) | Comprehensive 13-test automated regression suite including controlled SHAP feature identity check. |
| [`MODEL_CARD.md`](./MODEL_CARD.md) | Standardized Model Card detailing task, cohort, evaluation, and technical limitations. |
| [`requirements.txt`](./requirements.txt) | Pinned Python 3.13 dependency manifest. |
| [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) | Ground-truth test evaluation metrics and confusion matrix artifact. |
| [`artifacts/metadata.json`](./artifacts/metadata.json) | Pipeline metadata, hyperparameters, and feature schemas. |

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

### 2. Training Workflows

#### Path A: Train from Bundled Cleaned Cohort
```bash
# Validate bundled cohort (artifacts/cdc_brfss_diabetes_clean.csv)
python ingest_brfss.py --validate-only

# Train model, compute CV and baselines, serialize artifacts
python ml_pipeline.py --samples 50000 --seed 42 --model rf
```

#### Path B: Ingest Raw CDC BRFSS Dataset
```bash
# Obtain raw 2015 CDC BRFSS CSV (e.g. from CDC.gov or Kaggle) into archive/2015.csv
python ingest_brfss.py --raw-file archive/2015.csv --samples 50000 --seed 42

# Train model
python ml_pipeline.py --samples 50000 --seed 42 --model rf
```

### 3. Run Automated Tests

Execute the 13-test regression suite:
```bash
pytest tests/ -v
```

### 4. Launch the Backend API

```bash
# Start FastAPI service on 127.0.0.1:8000
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```
OpenAPI documentation available at `http://127.0.0.1:8000/docs`.

### 5. Launch the Frontend

```bash
# Install frontend dependencies
npm install

# Start Vite development server
npm run dev -- --port 3000

# Build production bundle
npm run build
```
Open `http://127.0.0.1:3000` to interact with the dashboard.

---

## Sample API Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bmi": 32.5,
    "high_bp": 1,
    "high_chol": 1,
    "chol_check": 1,
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
