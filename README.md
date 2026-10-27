# EndocrineAI: CDC BRFSS Diabetes Risk Stratification & Explainable AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13.3-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%200.54.1-blue.svg)](https://shap.readthedocs.io)
[![React 18](https://img.shields.io/badge/Frontend-React%20%2B%20TailwindCSS-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Tests Passing](https://img.shields.io/badge/Tests-21%2F21%20Passing-success.svg)](./tests/)
[![Dataset](https://img.shields.io/badge/Dataset-CDC%20BRFSS%202015-green.svg)](https://www.cdc.gov/brfss/)

An educational machine learning web application that stratifies survey respondents into three self-reported glycemic categories: **No Reported Diabetes**, **Pre-diabetic**, and **Diabetic**, utilizing 21 epidemiological health indicators from the **U.S. CDC Behavioral Risk Factor Surveillance System (BRFSS)** with mathematically verified class-relative TreeSHAP feature attributions and empirical modeling bottleneck investigations.

> [!IMPORTANT]
> **Educational & Methodological Scope Disclaimer**  
> This software is an **academic student portfolio project**. It is trained and benchmarked on **observational cross-sectional survey data** from the CDC BRFSS. It estimates statistical associations with existing self-reported diabetes status, **not** a validated medical prediction of future diabetes onset. It is **not** an FDA-cleared diagnostic medical device and **must not** be used to direct clinical care, prescribe treatments, or substitute for formal laboratory diagnosis (fasting plasma glucose or venous HbA1c assays).

For comprehensive technical specifications, class distributions, and limitation analyses, see the [Model Card](./MODEL_CARD.md). Master experimental records connecting each metrics file to data, configuration, and code versions are stored in [Baseline Manifest](./artifacts/experiments/baseline_manifest.json) and [Master Experiment Manifest](./artifacts/experiments/experiment_manifest.json).

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
                                  | (Type coercion & discrete checks)  |
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
                                  | - Decision Rule (Threshold/Argmax) |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +-----------------+------------------+
                                  | React 18 + Tailwind UI Dashboard   |
                                  | - Stale Result Invalidation        |
                                  | - Field-Level Validation (422)     |
                                  | - Relative SHAP Direction Badges   |
                                  | - 7 Empirical Investigations Tab   |
                                  +------------------------------------+
```

---

## Data Lineage & Attrition Waterfall

The pipeline enforces complete transparency regarding dataset origins and attrition:

| Stage | Record Count | Class 0 (No Diabetes) | Class 1 (Pre-diabetic) | Class 2 (Diabetic) | Lineage Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw BRFSS Survey** | **441,456** | Unmapped | Unmapped | Unmapped | Original CDC 2015 Annual Survey interviews (`archive/2015.csv`) |
| **Post-Cleaning Eligible** | **251,117** | 211,231 (84.11%) | 4,619 (1.84%) | 35,267 (14.04%) | Complete-case cohort surviving 21-feature cleaning (190,339 excluded) |
| **Selected Baseline Cohort** | **50,000** | 42,060 (84.12%) | 923 (1.84%) | 7,017 (14.04%) | Stratified subsample (`artifacts/cdc_brfss_diabetes_clean.csv`) |
| **Training Partition (80%)** | **40,000** | 33,648 (84.12%) | 739 (1.85%) | 5,613 (14.03%) | Strict train partition (`train_indices.npy`, seed 42) |
| **Test Holdout Partition (20%)** | **10,000** | 8,412 (84.12%) | 184 (1.84%) | 1,404 (14.04%) | Untouched development benchmark (`test_indices.npy`) |

### Exclusion Telemetry
- Total records excluded during complete-case cleaning: **190,339**
  - Invalid target responses (`DIABETE3` in [7, 9, blank]): 1,029
  - Missing/refused health status (`GENHLTH` in [7, 9, blank]): 1,208
  - Missing/refused blood pressure (`BPHIGH4` in [7, 9, blank]): 1,438
  - Missing/refused cholesterol check (`TOLDHI2` in [7, 9, blank]): 59,286
  - Missing/refused BMI calculations (`_BMI5` blank): 36,364
  - Missing/refused lifestyle/demographics (`_SMOKER3`, `EXERANY2`, `_AGEG5YR`, `INCOME2`, etc.): 91,014

---

## Empirical Modeling Bottleneck Investigations

All 7 empirical investigations were executed via [`scripts/run_experiments.py`](./scripts/run_experiments.py) and serialized as versioned JSON artifacts in [`artifacts/experiments/`](./artifacts/experiments/):

### 1. Learning Curves (Is 50k Data Volume the Bottleneck?)
Trained across nested stratified training subsets from 10k to the full eligible training pool (200,893 samples), evaluated against the fixed 10,000-sample test holdout:
- **10k Training Samples:** Macro F1 = 0.4349, Balanced Acc = 44.86%, Pre-diab Recall = 2.62%, Pre-diab Precision = 2.43%
- **25k Training Samples:** Macro F1 = 0.4357, Balanced Acc = 45.02%, Pre-diab Recall = 3.14%, Pre-diab Precision = 2.54%
- **40k Training Samples (Baseline):** Macro F1 = **0.4362**, Balanced Acc = 45.18%, Pre-diab Recall = 4.71%, Pre-diab Precision = 2.78%
- **100k Training Samples:** Macro F1 = 0.4358, Balanced Acc = 45.28%, Pre-diab Recall = 5.76%, Pre-diab Precision = 2.82%
- **200,893 Training Samples (Full):** Macro F1 = **0.4347**, Balanced Acc = 45.89%, Pre-diab Recall = 10.60%, Pre-diab Precision = 2.88%

> [!NOTE]
> **Empirical Finding:** Macro F1 remains completely flat (~0.435) despite a 20x increase in training data. While pre-diabetic recall increases from 2.6% to 10.6%, precision remains locked near ~2.9% (barely above the 1.84% base prevalence). The modeling bottleneck is survey feature separation, not dataset size.

### 2. Controlled Preprocessing Ablation
- **Mode A (No Outlier Clipping):** Macro F1 = 0.4360
- **Mode B (BMI-Only Clipping):** Macro F1 = **0.4357** (Selected: preserves valid discrete categories)
- **Mode C (Legacy All-Numerical Clipping):** Macro F1 = 0.4362
- **Mode D (Unscaled Trees):** Macro F1 = 0.4357 (Confirms tree scaling invariance)
- *Conclusion:* Preprocessing variations yield identical performance within ±0.0005 (well within cross-validation std ±0.004). BMI-only clipping was selected to prevent corrupting legitimate ordinal categories (age brackets, education, income).

### 3. Model Search & Class Weighting
- **Unweighted Models:** Predict 0 pre-diabetic cases (0.0% recall).
- **Balanced Class Weights:** Forces minority detection (Pre-diabetic recall: 4.89%, Diabetic recall: 73.72%), Macro F1 = 0.4365.
- **Custom Weights (`{0:1, 1:4, 2:2}`):** Intermediate tradeoff (Macro F1 = 0.4215, Pre-diabetic recall = 2.17%).

### 4. Decision Rules & Threshold Tradeoffs
Separated probability estimation from decision logic:
- **Default Argmax ($\max P_c$):** Macro F1 = 0.4362, Accuracy = 78.50%, Pre-diabetic Recall = 4.71%, Pre-diabetic Precision = 2.78% (307 false alarms).
- **Validation-Tuned Cutoffs ($\tau_{\text{diab}}=0.52, \tau_{\text{prediab}}=0.35$):** Macro F1 = **0.4487**, Accuracy = 77.36%, Pre-diabetic Recall = 8.15%, Pre-diabetic Precision = 2.65% (550 false alarms).
- **Cost-Sensitive Loss Matrix ($C_{1\to 0}=5, C_{2\to 0}=10$):** Pre-diabetic Recall = **29.02%**, but Precision collapses to **1.77%** with **3,101 false alarms**.

### 5. Probability Calibration with Fold-Local Preprocessing
Wrapped the full `Pipeline` inside `CalibratedClassifierCV(cv=5, method='sigmoid')` so fold-local preprocessing is learned independently in each internal fold:
- **Uncalibrated Random Forest:** Brier Score = 0.3863, Log Loss = 0.6774
- **Class-Prior Probability Baseline:** Brier Score = 0.2688, Log Loss = 0.4721
- **Sigmoid Calibrated Pipeline (Platt):** Brier Score = **0.2204**, Log Loss = **0.4009**
- *Serving Strategy:* Platt scaling strictly outperforms the class-prior baseline. Because wrapping the ensemble in CalibratedClassifierCV alters probability additivity for TreeSHAP, uncalibrated probabilities are served by default to preserve verified local XAI attributions, while calibrated probabilities are available as a versioned option.

### 6. Survey Profile Duplication & Cross-Split Overlap
- **Cross-Split Overlap:** 73 records in the holdout test set (0.73%) share exact 21-feature profiles with records in the training set across 72 unique survey profiles.
- **Bayes Error Conflict:** In the 50,000 cohort, 12 profile clusters have identical survey answers but conflicting diabetes diagnoses, illustrating the irreducible error ceiling of survey questionnaires lacking laboratory blood biomarkers.

### 7. Monotonic Runtime & Latency Benchmarks
Measured via monotonic `time.perf_counter()` on Apple Silicon arm64 (macOS 15.4):
- **Preprocessing Fit Duration:** 0.0119s
- **Estimator Fit Duration:** 0.3276s
- **Total Pipeline Fit Duration:** 0.3421s
- **Batch Inference Throughput:** 327,734 samples/sec (on 10,000 holdout records)
- **Single-Record Latency (p50):** 15.93 ms (warmup excluded, repeated 100 times)

---

## Benchmark Evaluation Results

Derived from [`artifacts/model_metrics.json`](./artifacts/model_metrics.json) on the held-out 10,000-sample test partition:

### Baseline Model Comparison (Hold-Out Test Set, N = 10,000)

| Model Architecture | Accuracy | Balanced Acc | Macro F1 | Brier Score | Log Loss | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Dummy Baseline** | 84.12% | 33.33% | 0.3046 | 0.3176 | 5.7237 | 0.003s | 0.07 µs |
| **Logistic Regression (Balanced)** | 64.35% | 51.64% | 0.4271 | 0.4875 | 0.8776 | 0.424s | 0.06 µs |
| **HistGradientBoosting** | 64.35% | 51.00% | 0.4235 | 0.4529 | 0.7973 | 0.797s | 2.22 µs |
| **Random Forest (Balanced, Selected)** | **71.58%** | **50.43%** | **0.4365** | **0.3863** | **0.6774** | **0.334s** | **2.45 µs** |
| **Project Risk Heuristic** | 76.86% | 42.33% | 0.4052 | 0.4628 | N/A | 0.000s | 0.10 µs |

### Per-Class Test Performance (Random Forest)

| Survey Class | Support | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC (OvR) | Average Precision (PR-AUC) | Prevalence Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Reported Diabetes** | 8,412 | 72.68% | 94.10% | 0.8202 | 0.8208 | 0.9588 | 84.12% |
| **Pre-diabetic** | 184 | **4.89%** | **2.36%** | **0.0319** | 0.6386 | 0.0281 | 1.84% |
| **Diabetic** | 1,404 | **73.72%** | **33.15%** | **0.4574** | 0.8276 | 0.4209 | 14.04% |

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

| File | Description & Link |
| :--- | :--- |
| [`ingest_brfss.py`](./ingest_brfss.py) | Ingestion CLI supporting `--samples all` or integer, discrete domain validation, and exclusion telemetry. |
| [`ml_pipeline.py`](./ml_pipeline.py) | Training pipeline with BMI-only outlier capping, fold-local calibration, decision rules, and monotonic timing. |
| [`clinical_transformers.py`](./clinical_transformers.py) | Domain definitions (`DISCRETE_ALLOWED_VALUES`), discrete validator, and `PhysiologicalOutlierCapper`. |
| [`scripts/run_experiments.py`](./scripts/run_experiments.py) | Reproducible runner executing all 7 empirical investigations and saving JSON artifacts. |
| [`scripts/preserve_baseline_manifest.py`](./scripts/preserve_baseline_manifest.py) | Creates cryptographic baseline manifest linking metrics, dataset hashes, and code versions. |
| [`artifacts/experiments/`](./artifacts/experiments/) | Complete experiment directory containing learning curves, ablation, calibration, and benchmark JSONs. |
| [`api.py`](./api.py) | FastAPI service with exact feature order, decision rule support, artifact compatibility checks, and `/experiments`. |
| [`DiabetesDashboard.jsx`](./DiabetesDashboard.jsx) | React 18 UI with decision rule toggle, accurate probability terminology, and Empirical Investigations tab. |
| [`tests/test_pipeline_and_api.py`](./tests/test_pipeline_and_api.py) | 13 integration and regression tests for ingestion, pipeline, and API endpoints. |
| [`tests/test_modeling_improvements.py`](./tests/test_modeling_improvements.py) | 8 targeted regression tests covering discrete domains, loading modes, calibration, rules, and compatibility. |
| [`MODEL_CARD.md`](./MODEL_CARD.md) | Standardized Model Card with full experimental findings, data lineage, and limitations. |

---

## Quickstart & Reproducibility Guide

### 1. Clean Environment Setup
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
# Validate bundled cohort format
python ingest_brfss.py --validate-only

# Train model on bundled 50k cohort
python ml_pipeline.py --samples 50000 --seed 42 --model rf
```

#### Path B: Ingest Raw CDC BRFSS Dataset & Train Full Data
```bash
# Clean raw CDC BRFSS 2015 CSV and extract all 251,117 eligible records
python ingest_brfss.py --data-mode raw --raw-path archive/2015.csv --samples all --seed 42

# Train model on full available eligible cohort
python ml_pipeline.py --data-mode cleaned --clean-path artifacts/cdc_brfss_diabetes_clean_full.csv --samples all --seed 42 --model rf
```

#### Path C: Reproduce All Empirical Bottleneck Experiments
```bash
# Executes learning curves, preprocessing ablation, model search, calibration, and runtime benchmarks
python scripts/run_experiments.py
```

### 3. Run Automated Tests
```bash
# Run all 21 automated regression tests
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
    "income": 5,
    "decision_rule": "threshold_tuned"
  }'
```
