# Model Card: CDC BRFSS Diabetes Risk Stratification (v2.0)

## 1. Model Details

- **Model Name:** EndocrineAI CDC BRFSS Diabetes Risk Stratifier
- **Model Version:** 2.0.0
- **Model Type:** Multi-Class Random Forest Ensemble (`RandomForestClassifier`, 100 estimators, balanced class weights, max depth 12)
- **Pipeline Architecture:** Scikit-Learn `Pipeline` with `ColumnTransformer`:
  - *Numerical Features (4):* `bmi`, `phys_hlth`, `ment_hlth`, `age` -> `PhysiologicalOutlierCapper` -> `RobustScaler`
  - *Binary & Ordinal Features (17):* `high_bp`, `high_chol`, `chol_check`, `smoker`, `stroke`, `heart_disease_or_attack`, `phys_activity`, `fruits`, `veggies`, `hvy_alcohol_consump`, `any_healthcare`, `no_doc_bc_cost`, `gen_hlth`, `diff_walk`, `sex`, `education`, `income` -> `passthrough`
- **Explainability:** TreeSHAP (`shap.TreeExplainer`) with holdout-free training background summarizer (100 k-means clusters on `X_train`), dynamic class-specific expected base value, and strict additivity verification.
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

## 3. Training & Evaluation Data

- **Primary Data Source:** Centers for Disease Control and Prevention (CDC) **Behavioral Risk Factor Surveillance System (BRFSS 2015)**.
- **Raw Survey Size:** 441,456 raw telephone survey interviews across U.S. states and territories.
- **Cohort Selection (Path A & Path B):**
  - Path A: Pre-bundled cleaned cohort of 50,000 respondents (`artifacts/cdc_brfss_diabetes_clean.csv`, SHA-256 verified).
  - Path B: Raw ingestion via `ingest_brfss.py` from official CDC BRFSS fixed-width or CSV records with exact row-filtering telemetry.
- **Target Variable (`DIABETE3`):**
  - `0`: No Reported Diabetes (Normoglycemic / non-diabetic survey responses) — 84.12%
  - `1`: Pre-diabetic (Borderline / gestational-excluded pre-diabetes) — 1.84%
  - `2`: Diabetic (Diagnosed diabetes) — 14.04%
- **Representativeness Caveat:** Complete-case filtering removes records with missing or refusable survey responses. Consequently, this sample reflects a complete-case analytic cohort and must not be asserted as perfectly representative of the entire U.S. population without sampling weight adjustment.

---

## 4. Evaluation Methodology

- **Holdout Partition Integrity:** A strictly untouched 20% test partition ($N = 10,000$) was preserved before any model fitting, hyperparameter selection, or threshold tuning.
- **Validation Scheme:** Stratified 5-Fold Cross-Validation executed exclusively within the 80% training partition ($N = 40,000$). All transformers (`PhysiologicalOutlierCapper`, `RobustScaler`) were fitted strictly inside each training fold to eliminate data leakage.
- **Comparative Baselines on Identical Benchmark:**
  1. *Majority Class Dummy:* Predicts class 0 for all instances.
  2. *Multinomial Logistic Regression:* Linear model with inverse-prevalence balanced weighting.
  3. *Histogram Gradient Boosting:* `HistGradientBoostingClassifier` with balanced weighting.
  4. *Project-Defined Screening Heuristic:* Rule-based cutoffs (High BP + High Chol + BMI $\ge 30$ + Age $\ge 60$).

---

## 5. Quantitative Benchmark Results (Hold-Out Test Set, N = 10,000)

### Primary Metrics
| Metric | Test Value | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **71.58%** | [70.73%, 72.44%] | Overall correct classification rate |
| **Balanced Accuracy** | **50.43%** | N/A | Unweighted mean class sensitivity |
| **Macro Precision** | **0.4321** | N/A | Unweighted mean precision |
| **Macro Recall** | **0.5043** | N/A | Unweighted mean recall |
| **Macro F1-Score** | **0.4365** | [0.4271, 0.4468] | Harmonic mean of macro precision/recall |
| **Weighted F1-Score** | **0.7547** | N/A | Weighted by survey class support |
| **Multiclass Brier Score** | **0.3863** | N/A | Mean squared probability error |
| **Calibration Log Loss** | **0.6774** | N/A | Cross-entropy on uncalibrated probabilities |
| **Macro ROC-AUC (OvR)** | **0.7623** | N/A | One-vs-Rest macro area under curve |

### Per-Class Performance
| Class | Support | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Reported Diabetes** | 8,412 | 72.68% | 94.10% | 0.8202 | 0.8208 | 0.9588 |
| **Pre-diabetic** | 184 | **4.89%** | **2.36%** | **0.0319** | 0.6386 | 0.0281 |
| **Diabetic** | 1,404 | **73.72%** | **33.15%** | **0.4574** | 0.8276 | 0.4209 |

### Hold-Out Confusion Matrix
```
                       Pred: No Reported Diabetes    Pred: Pre-diabetic    Pred: Diabetic      Total Actual
Actual: No Reported Diabetes            6,114                       307             1,991              8,412
Actual: Pre-diabetic                       79                         9                96                184
Actual: Diabetic                          304                        65             1,035              1,404
Total Predicted                         6,497                       381             3,122             10,000
```

### Baseline Model Comparison
| Architecture | Accuracy | Balanced Acc | Macro F1 | Brier Score | Log Loss | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Dummy** | 84.12% | 33.33% | 0.3046 | 0.3176 | 5.7237 | 0.003s | 0.07 µs |
| **Logistic Regression** | 64.35% | 51.64% | 0.4271 | 0.4875 | 0.8776 | 0.424s | 0.06 µs |
| **HistGradientBoosting** | 64.35% | 51.00% | 0.4235 | 0.4529 | 0.7973 | 0.797s | 2.22 µs |
| **Random Forest (Balanced)** | **71.58%** | **50.43%** | **0.4365** | **0.3863** | **0.6774** | **0.334s** | **2.45 µs** |
| **Project Risk Heuristic** | 76.86% | 42.33% | 0.4052 | 0.4628 | N/A | 0.000s | 0.10 µs |

---

## 6. Crucial Technical Findings & Limitations

### 1. Pre-diabetes Detection Ceiling
Pre-diabetes sensitivity in 3-class survey modeling is severely limited (4.89% recall, 2.36% precision). This stems from two structural factors:
- **Severe Class Imbalance:** Pre-diabetic respondents make up only 1.84% of the cohort.
- **Symptom Overlap:** Individuals with pre-diabetes share behavioral and physical survey characteristics (BMI, physical activity, diet, general health) with both normoglycemic and diagnosed diabetic respondents. Survey questions alone cannot capture early asymptomatic dysglycemia without invasive venous laboratory biomarkers (fasting plasma glucose or HbA1c).

### 2. Irreducible Bayes Error (Duplicate Survey Profiles)
Analysis of the 50,000 cohort across 21 features reveals:
- **457 duplicate survey rows** (0.91% of the cohort).
- **12 conflicting profile clusters** where respondents gave 100% identical responses across all 21 questions yet reported opposing diabetes diagnoses (e.g., one respondent reported diabetes while another did not).
- *Implication:* Even an infinite-capacity Bayesian optimal classifier cannot achieve 100% accuracy on survey indicators alone.

### 3. Probability Calibration vs. TreeSHAP Additivity Tradeoff
- Applying Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) within the training fold reduces Log Loss from 0.6774 to 0.4007 and Brier Score from 0.3863 to 0.2234.
- However, wrapping the tree ensemble in an isotonic or Platt calibration layer transforms the final probabilities non-linearly, **breaking mathematical TreeSHAP additivity** ($\text{Base} + \sum \text{SHAP} = P$).
- *Architectural Decision:* The serving pipeline serves uncalibrated ensemble probabilities to ensure exact, verified local XAI feature attributions.

### 4. Secondary Binary Screening Task
When framed as a secondary binary screening task (0: No Reported Diabetes vs. 1: At-Risk [Pre-diabetic or Diabetic]):
- **At-Risk Sensitivity (Recall):** **86.96%**
- **Specificity:** **60.91%**
- **ROC-AUC:** **0.8208**
- **Average Precision (PR-AUC):** **0.4484**
- This demonstrates that while fine-grained 3-class discrimination is constrained by data limitations, the survey indicators provide strong sensitivity for identifying general metabolic risk.

---

## 7. Model Card Metadata

- **Authors:** Siddharth Bhargava (Portfolio Project)
- **Date:** October 2026
- **Status:** Complete & Verified Pass
