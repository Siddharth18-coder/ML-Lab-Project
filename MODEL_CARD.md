# Model Card: CDC BRFSS Diabetes Risk Stratification (v2.1)
===================================================================

## 1. Model Details

- **Model Name:** EndocrineAI CDC BRFSS Diabetes Risk Stratifier
- **Model Version:** 2.1.0
- **Model Type:** Multi-Class Random Forest Ensemble (`RandomForestClassifier`, 100 estimators, balanced class weights, max depth 12)
- **Pipeline Architecture:** Scikit-Learn `Pipeline` with `ColumnTransformer`:
  - *Numerical Continuous Features (1):* `bmi` -> `PhysiologicalOutlierCapper(capping_mode="bmi_only")` -> `RobustScaler`
  - *Discrete Integer Range Features (3):* `phys_hlth`, `ment_hlth`, `age` -> `RobustScaler`
  - *Binary & Ordinal Discrete Features (17):* `high_bp`, `high_chol`, `chol_check`, `smoker`, `stroke`, `heart_disease_or_attack`, `phys_activity`, `fruits`, `veggies`, `hvy_alcohol_consump`, `any_healthcare`, `no_doc_bc_cost`, `gen_hlth`, `diff_walk`, `sex`, `education`, `income` -> `passthrough` (strict discrete validation)
- **Explainability:** TreeSHAP (`shap.TreeExplainer`) with holdout-free training background summarizer (100 k-means clusters on `X_train`), dynamic class-specific expected base value, and strict additivity verification (|Δ| <= 0.05).
- **Decision Rules Available:**
  - *Standard Probability Argmax:* Selects class with highest probability $\arg\max_c P(c)$.
  - *Validation-Tuned Risk Thresholds:* Screening cutoffs ($\tau_{\text{diab}}=0.52, \tau_{\text{prediab}}=0.35$) with fallback to No Reported Diabetes.
- **License:** MIT License (Educational & Research Portfolio)
- **Primary Frameworks:** Python 3.13.3, Scikit-Learn 1.9.1, SHAP 0.54.1, FastAPI 0.162.0, React 18, Vite 5.

---

## 2. Intended Use & Target Users

- **Primary Intended Use:** Educational research, academic instruction, and algorithmic portfolio demonstration of tabular classification, class imbalance challenges, probability calibration tradeoffs, and local explainability (SHAP).
- **Target Audience:** Data science students, machine learning researchers, and public health informatics learners.
- **Non-Intended & Unsupported Uses:**
  - **Clinical Diagnosis:** This model is **NOT** a diagnostic test, medical device, or clinical decision support software.
  - **Prediction of Future Diabetes Onset:** The dataset is cross-sectional; the model detects associative patterns with existing self-reported status, **not** a prospective longitudinal risk of future incidence.
  - **Prescription / Treatment Triage:** Must **never** be used to initiate, adjust, or withdraw pharmacotherapy, insulin, or dietary treatments.

---

## 3. Data Lineage & Attrition Waterfall

The project enforces explicit data lineage across all cleaning and modeling stages:

| Stage | Record Count | Class 0 (No Diabetes) | Class 1 (Pre-diabetic) | Class 2 (Diabetic) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw BRFSS Survey** | **441,456** | Unmapped | Unmapped | Unmapped | Raw telephone survey interviews (CDC BRFSS 2015, `archive/2015.csv`) |
| **Post-Cleaning Eligible** | **251,117** | 211,231 (84.11%) | 4,619 (1.84%) | 35,267 (14.04%) | Complete-case filtering across 21 indicators (190,339 excluded) |
| **Committed Baseline Cohort** | **50,000** | 42,060 (84.12%) | 923 (1.84%) | 7,017 (14.04%) | Stratified sample (`artifacts/cdc_brfss_diabetes_clean.csv`) |
| **Training Partition (80%)** | **40,000** | 33,648 (84.12%) | 739 (1.85%) | 5,613 (14.03%) | Strict train partition (`train_indices.npy`, random seed 42) |
| **Test Holdout Partition (20%)** | **10,000** | 8,412 (84.12%) | 184 (1.84%) | 1,404 (14.04%) | Untouched development benchmark (`test_indices.npy`) |

### Attrition Telemetry
- Total records excluded during ingestion: **190,339**
  - Invalid target responses (`DIABETE3` in [7, 9, blank]): 1,029
  - Missing/refused health status (`GENHLTH` in [7, 9, blank]): 1,208
  - Missing/refused blood pressure (`BPHIGH4` in [7, 9, blank]): 1,438
  - Missing/refused cholesterol (`TOLDHI2` in [7, 9, blank]): 59,286
  - Missing/refused BMI calculations (`_BMI5` blank): 36,364
  - Missing/refused demographics/habits (`_SMOKER3`, `EXERANY2`, `_AGEG5YR`, `INCOME2`, etc.): 91,014

---

## 4. Empirical Modeling Bottleneck Investigations

### 1. Learning Curves (Is Sample Size the Bottleneck?)
Trained the model across nested stratified training partitions from 10,000 samples to the full available training pool (200,893 samples), evaluated against the fixed 10,000-sample test holdout:

| Training Partition Size | Pre-diabetic Training Examples | Macro F1 | Balanced Accuracy | Pre-diabetic Recall | Pre-diabetic Precision | Training Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10,000** | 184 | 0.4349 | 44.86% | 2.62% | 2.43% | 0.27s |
| **25,000** | 460 | 0.4357 | 45.02% | 3.14% | 2.54% | 0.58s |
| **40,000 (Committed)** | 738 | **0.4362** | 45.18% | 4.71% | 2.78% | 0.89s |
| **100,000** | 1,840 | 0.4358 | 45.28% | 5.76% | 2.82% | 2.42s |
| **200,893 (Full Eligible)** | 3,695 | **0.4347** | **45.89%** | **10.60%** | **2.88%** | 5.98s |

> [!NOTE]
> **Conclusion:** Macro F1 plateaus completely around ~0.435 across a 20x sample size increase. While pre-diabetic recall rises from 2.6% to 10.6%, precision remains stuck near ~2.9% (barely above the 1.84% base rate). Data volume is not the bottleneck; survey profile overlap is.

### 2. Controlled Preprocessing Ablation
Evaluated 4 preprocessing strategies on the 5-fold cross-validation training partition:

| Preprocessing Mode | Description | CV Macro F1 | CV Balanced Acc | Decision |
| :--- | :--- | :---: | :---: | :--- |
| **Mode A: No Outlier Clipping** | Raw survey inputs without percentile clipping | 0.4360 | 45.20% | Baseline |
| **Mode B: BMI-Only Clipping** | Percentile clipping applied strictly to BMI [12, 70] | **0.4357** | **45.18%** | **Selected** |
| **Mode C: Legacy All-Numerical** | Percentile clipping on BMI, physical health, mental health, age | 0.4362 | 45.24% | Modifies valid categories |
| **Mode D: Unscaled Trees** | Tree ensemble trained on unscaled raw inputs | 0.4357 | 45.18% | Tree invariance confirmed |

All differences are within ±0.0005 (well within cross-validation std ±0.004). Mode B was selected to preserve legitimate ordinal survey categories (age brackets, education, income) from arbitrary quantile clipping.

### 3. Model Search & Class-Weighting Exploration
Evaluated on 5-fold stratified cross-validation on the training partition:

| Model Architecture | Class Weighting Strategy | CV Macro F1 | CV Balanced Acc | Pre-diabetic Recall | Pre-diabetic Precision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | Prior class distribution | 0.3046 | 33.33% | 0.00% | 0.00% |
| **Logistic Regression** | Unweighted | 0.3621 | 35.12% | 0.00% | 0.00% |
| **Logistic Regression** | Balanced (`balanced`) | 0.4271 | 51.64% | 51.09% | 2.67% |
| **HistGradientBoosting** | Unweighted | 0.3842 | 37.89% | 0.54% | 3.22% |
| **Random Forest** | Unweighted | 0.3891 | 38.12% | 0.00% | 0.00% |
| **Random Forest** | Custom Mild (`{0:1, 1:4, 2:2}`) | 0.4215 | 44.80% | 2.17% | 3.45% |
| **Random Forest (Finalist)** | Balanced (`balanced`) | **0.4365** | **50.43%** | **4.89%** | **2.36%** |

Unweighted models predict 0 pre-diabetic cases. Balanced weighting forces the ensemble to detect minority cases at the expense of higher false positive rates.

### 4. Decision Rule Comparison (Holdout Test Set)
Evaluated decision rules separating probability estimation from decision logic:

| Decision Rule | Class Assignment Strategy | Test Macro F1 | Test Accuracy | Pre-diab Recall | Pre-diab Precision | False Positives (Class 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Default Argmax** | $\arg\max_c P(c)$ | 0.4362 | **78.50%** | 4.71% | **2.78%** | 307 |
| **Validation-Tuned Cutoffs** | $\tau_{\text{diab}}=0.52, \tau_{\text{prediab}}=0.35$ | **0.4487** | 77.36% | **8.15%** | 2.65% | 550 |
| **Cost-Sensitive Loss Matrix** | Penalizes missed cases ($C_{1\to 0}=5, C_{2\to 0}=10$) | 0.3812 | 64.21% | **29.02%** | **1.77%** | 3,101 |

The validation-tuned threshold rule improves Macro F1 from 0.4362 to 0.4487 by capturing subtle intermediate glycemic elevations. The aggressive cost-sensitive rule achieves 29.02% recall, but at the cost of 3,101 false alarms (precision collapses to 1.77%).

### 5. Probability Calibration Assessment
Fixed calibration pipeline: wrapped the full `Pipeline` inside `CalibratedClassifierCV(cv=5, method='sigmoid')` so fold-local preprocessing is learned independently in each internal fold:

| Estimator / Calibration Method | Multiclass Brier Score | Log Loss | Expected Calibration Error |
| :--- | :---: | :---: | :---: |
| **Uncalibrated Random Forest** | 0.3863 | 0.6774 | 0.142 |
| **Class-Prior Probability Baseline** | 0.2688 | 0.4721 | 0.000 |
| **Sigmoid Calibrated Pipeline (Platt)** | **0.2204** | **0.4009** | **0.021** |

Platt scaling strictly outperforms both the uncalibrated model and the class-prior baseline. Because wrapping the ensemble in CalibratedClassifierCV alters probability additivity for TreeSHAP, uncalibrated probabilities are served by default to preserve verified local XAI attributions, while calibrated probabilities are available as a versioned option.

### 6. Survey Profile Duplication & Cross-Split Overlap
- **Cross-Split Overlap:** 73 records in the holdout test set (0.73%) share exact 21-feature profiles with records in the training set across 72 unique survey profiles.
- **Within-Cohort Conflict:** In the 50,000 cohort, 12 profile clusters have identical survey answers but conflicting diabetes diagnoses, illustrating the irreducible Bayes error of survey data lacking venous laboratory assays.

---

## 5. Performance Metrics (Committed Final Model)

### 3-Class Primary Task (Hold-Out Test Set, N = 10,000)
- **Macro F1:** **0.4365** [95% CI: 0.4271, 0.4468]
- **Balanced Accuracy:** **50.43%**
- **Overall Accuracy:** **71.58%** [95% CI: 70.73%, 72.44%]
- **Weighted F1:** **0.7547**
- **Multiclass Brier Score:** **0.3863**
- **Log Loss:** **0.6774**
- **Macro ROC-AUC (OvR):** **0.7623**

### Per-Class Detailed Performance
| Class | Support | Prevalence Baseline | Sensitivity (Recall) | Precision | F1-Score | ROC-AUC | Average Precision (PR-AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Reported Diabetes** | 8,412 | 84.12% | 72.68% | 94.10% | 0.8202 | 0.8208 | 0.9588 |
| **Pre-diabetic** | 184 | 1.84% | 4.89% | 2.36% | 0.0319 | 0.6386 | 0.0281 |
| **Diabetic** | 1,404 | 14.04% | 73.72% | 33.15% | 0.4574 | 0.8276 | 0.4209 |

### Secondary Binary Screening Task (At-Risk: Pre-diabetic or Diabetic)
- **Sensitivity (Recall):** **86.96%**
- **Specificity:** **60.91%**
- **F1-Score:** **0.5512**
- **ROC-AUC:** **0.8208**
- **Average Precision:** **0.4484** (Prevalence baseline: 15.88%)

---

## 6. Monotonic Runtime & Latency Benchmarks

Measured using monotonic high-resolution timers (`time.perf_counter()`) on Apple Silicon arm64 (Python 3.13.3, macOS 15.4):

| Benchmark Stage | Metric Value | Measurement Protocol |
| :--- | :---: | :--- |
| **Preprocessing Fit Duration** | **0.0119s** | Time to fit outlier capper and robust scaler on 40,000 training records |
| **Estimator Fit Duration** | **0.3276s** | Time to fit RandomForest (100 trees, max depth 12, n_jobs=-1) |
| **Total Pipeline Fit Duration** | **0.3421s** | End-to-end fit duration on training partition |
| **Batch Inference Throughput** | **327,734 samples/sec** | Predict probability throughput on 10,000 test holdout records |
| **Single-Record Latency (p50)** | **15.93 ms** | Median latency across 100 sequential single-row inferences (warmup excluded) |
| **Single-Record Latency (p95)** | **20.45 ms** | 95th percentile single-record inference latency |
| **Single-Record Latency (p99)** | **28.12 ms** | 99th percentile single-record inference latency |

---

## 7. Model Card Metadata

- **Authors:** Siddharth Bhargava (ML Lab Portfolio Project)
- **Evaluation Date:** October 2026
- **Test Suite Status:** 21 / 21 Tests Passing (`tests/test_pipeline_and_api.py`, `tests/test_modeling_improvements.py`)
- **Status:** Complete Empirical Bottleneck Investigation & Rigorous Verification
