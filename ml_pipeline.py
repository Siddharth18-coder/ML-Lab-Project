"""
ml_pipeline.py - Clinical Diabetes Risk Machine Learning Pipeline (Educational Prototype)
========================================================================================
Scope & Purpose:
This pipeline implements data cleaning, leakage-free preprocessing, multi-class model training,
stratified 5-fold cross-validation, baseline comparisons, test-set metrics evaluation, and
patient-level SHAP explainability for an educational diabetes classification prototype.

Dataset Provenance & Clinical Framing:
- Default Data: Purely synthetic benchmark cohort generated via American Diabetes Association
  (ADA) physiological thresholds with realistic noise, physiological outliers, and missingness.
- Label-Biomarker Mechanism: Labels are assigned first by latent metabolic risk, and key diagnostic
  biomarkers (fasting glucose, HbA1c) are subsequently generated from those labels.
  THEREFORE: High test set accuracy on this synthetic data is an inherent artifact of data synthesis
  and MUST NOT be claimed or interpreted as evidence of real-world clinical performance.
- Bundled AIM-94 Dataset: The dataset in ./diabetes/Diabetes-Data/ contains 1994 time-series insulin
  and glucose logs for 70 patients; it lacks cross-sectional lipid/vital/lifestyle features and 3-class
  labels, so it was not used to train this model.
- Classification Scope: Categorizes current glycemic status (Healthy, Pre-diabetic, Diabetic).
  It does NOT predict future longitudinal disease progression.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)

from clinical_transformers import PhysiologicalOutlierCapper, PHYSIOLOGICAL_BOUNDS

# Optional heavy ML libraries with explicit dependency reporting
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MLPipeline")

# -----------------------------------------------------------------------------
# Clinical Constants & Target Mapping
# -----------------------------------------------------------------------------
CLASS_NAMES = ["Healthy", "Pre-diabetic", "Diabetic"]
CLASS_MAP = {0: "Healthy", 1: "Pre-diabetic", 2: "Diabetic"}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "fasting_blood_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "systolic_bp",
    "diastolic_bp",
    "serum_creatinine",
    "physical_activity_minutes_per_week",
    "diet_quality_score",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "family_history_diabetes",
    "smoking_status",
    "alcohol_intake",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# -----------------------------------------------------------------------------
# Module 1: Synthetic Cohort Data Generation (Educational Benchmark)
# -----------------------------------------------------------------------------
def generate_synthetic_clinical_cohort(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic clinical cohort reflecting ADA physiological guidelines.
    IMPORTANT: This is synthetic data where labels drive the generation of several
    biomarkers; high classification accuracy is expected and not proof of clinical efficacy.
    """
    logger.info("Generating synthetic clinical benchmark cohort (%d samples, seed=%d)...", n_samples, random_state)
    rng = np.random.default_rng(random_state)

    # Demographic distributions
    age = rng.normal(52, 14, size=n_samples).clip(18, 95)
    gender = rng.choice(["Male", "Female", "Other"], size=n_samples, p=[0.48, 0.49, 0.03])
    ethnicity = rng.choice(
        ["Caucasian", "African American", "Asian", "Hispanic", "Other"],
        size=n_samples,
        p=[0.55, 0.16, 0.11, 0.14, 0.04]
    )
    family_history = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40])

    # Lifestyle distributions
    smoking = rng.choice(["Never", "Former", "Current"], size=n_samples, p=[0.52, 0.30, 0.18])
    alcohol = rng.choice(["None", "Moderate", "Heavy"], size=n_samples, p=[0.45, 0.45, 0.10])
    activity_min = rng.gamma(shape=2.5, scale=60, size=n_samples).clip(0, 700)
    diet_score = rng.normal(5.8, 2.0, size=n_samples).clip(1.0, 10.0)
    sleep = rng.normal(7.1, 1.3, size=n_samples).clip(3.5, 11.0)

    # Anthropometrics & Blood Pressure
    bmi_base = 24.5 + (age - 40) * 0.08 + (family_history * 2.2) - (activity_min / 120.0)
    bmi = (bmi_base + rng.normal(0, 3.8, size=n_samples)).clip(16.0, 58.0)

    systolic_bp = (112 + (age * 0.38) + (bmi * 0.45) + rng.normal(0, 11, size=n_samples)).clip(85, 210)
    diastolic_bp = (72 + (age * 0.15) + (bmi * 0.28) + rng.normal(0, 7, size=n_samples)).clip(50, 125)

    # Latent metabolic risk score
    # Note: Corrected vectorized per-patient smoking indicator
    smoking_effect = 0.35 * (smoking == "Current").astype(float)
    risk_score = (
        0.035 * (age - 45)
        + 0.12 * (bmi - 25)
        + 0.85 * family_history
        - 0.003 * (activity_min - 150)
        - 0.15 * (diet_score - 5.0)
        + 0.015 * (systolic_bp - 120)
        + smoking_effect
        + rng.normal(0, 0.95, size=n_samples)
    )

    # Class assignment based on clinical risk score percentiles
    labels = np.zeros(n_samples, dtype=int)
    pre_cutoff = np.percentile(risk_score, 48)
    diab_cutoff = np.percentile(risk_score, 76)

    labels[(risk_score >= pre_cutoff) & (risk_score < diab_cutoff)] = 1
    labels[risk_score >= diab_cutoff] = 2

    # Laboratory Panel - Fasting Glucose & HbA1c coupled to assigned class
    fbg = np.zeros(n_samples)
    hba1c = np.zeros(n_samples)

    for i in range(n_samples):
        c = labels[i]
        if c == 0:  # Healthy
            fbg[i] = rng.normal(87, 7.5) + (risk_score[i] * 1.5)
            hba1c[i] = rng.normal(5.05, 0.28) + (risk_score[i] * 0.05)
        elif c == 1:  # Pre-diabetic
            fbg[i] = rng.normal(111, 7.0) + (risk_score[i] * 1.8)
            hba1c[i] = rng.normal(6.05, 0.22) + (risk_score[i] * 0.06)
        else:  # Diabetic
            fbg[i] = rng.normal(156, 32.0) + (risk_score[i] * 3.5)
            hba1c[i] = rng.normal(8.10, 1.45) + (risk_score[i] * 0.15)

    fbg = fbg.clip(55.0, 390.0)
    hba1c = hba1c.clip(4.0, 15.5)

    # Lipid Panel & Renal markers
    tot_chol = (180 + (labels * 18) + (bmi * 0.6) + rng.normal(0, 25, size=n_samples)).clip(105, 380)
    hdl = (58 - (labels * 7.5) - (bmi * 0.35) + (activity_min / 60.0) + rng.normal(0, 9, size=n_samples)).clip(18, 110)
    ldl = (tot_chol - hdl - (fbg * 0.15) + rng.normal(0, 15, size=n_samples)).clip(40, 270)
    trig = (120 + (labels * 45) + (bmi * 1.8) + rng.normal(0, 35, size=n_samples)).clip(45, 550)
    creatinine = (0.85 + (labels * 0.25) + (age * 0.005) + rng.normal(0, 0.18, size=n_samples)).clip(0.4, 4.5)

    df = pd.DataFrame({
        "age": age.round(1),
        "gender": gender,
        "ethnicity": ethnicity,
        "family_history_diabetes": family_history,
        "bmi": bmi.round(1),
        "fasting_blood_glucose": fbg.round(1),
        "hba1c": hba1c.round(2),
        "total_cholesterol": tot_chol.round(1),
        "hdl_cholesterol": hdl.round(1),
        "ldl_cholesterol": ldl.round(1),
        "triglycerides": trig.round(1),
        "systolic_bp": systolic_bp.round(0),
        "diastolic_bp": diastolic_bp.round(0),
        "serum_creatinine": creatinine.round(2),
        "physical_activity_minutes_per_week": activity_min.round(0),
        "smoking_status": smoking,
        "alcohol_intake": alcohol,
        "diet_quality_score": diet_score.round(1),
        "sleep_hours": sleep.round(1),
        "diabetes_stage": labels
    })

    # Realistic missingness (~5% missing lab values)
    logger.info("Injecting missing clinical values (~5%% in selected labs) and physiological outliers...")
    for col in ["fasting_blood_glucose", "hba1c", "hdl_cholesterol", "triglycerides", "serum_creatinine"]:
        mask_missing = rng.random(size=n_samples) < 0.05
        df.loc[mask_missing, col] = np.nan

    # Extreme data-entry outliers to exercise capper
    outlier_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df.loc[outlier_idx[: len(outlier_idx) // 2], "fasting_blood_glucose"] = rng.uniform(420, 650)
    df.loc[outlier_idx[len(outlier_idx) // 2:], "hba1c"] = rng.uniform(16.5, 22.0)

    return df


# -----------------------------------------------------------------------------
# Module 2: Preprocessing Pipeline Architecture
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs a leakage-free Scikit-Learn preprocessing pipeline.
    Justified ordering for numerical features:
    1. PhysiologicalOutlierCapper: bounds extreme data-entry errors.
    2. RobustScaler: centers and scales features by median and IQR (ignores NaNs).
    3. KNNImputer: calculates distance in the normalized metric space, ensuring
       large-magnitude features (e.g. activity min) do not dominate subtle biomarkers.
    """
    numerical_transformer = Pipeline([
        ("capper", PhysiologicalOutlierCapper()),
        ("scaler", RobustScaler()),
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


# -----------------------------------------------------------------------------
# Module 3: Model Instantiation with Explicit Dependency Handling
# -----------------------------------------------------------------------------
def get_classifier(model_type: str = "rf", random_state: int = 42) -> BaseEstimator:
    """
    Instantiates requested classifier. Does NOT silently substitute dependencies.
    """
    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError(
                "XGBoost was explicitly requested (--model xgb), but cannot be loaded. "
                "Ensure xgboost is installed and OpenMP runtime (libomp.dylib / libgomp.so) is present."
            )
        logger.info("Selected XGBoost multi-class classifier.")
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    elif model_type == "lgb":
        if not HAS_LGB:
            raise RuntimeError(
                "LightGBM was explicitly requested (--model lgb), but cannot be loaded. "
                "Ensure lightgbm is installed and available."
            )
        logger.info("Selected LightGBM multi-class classifier.")
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            random_state=random_state,
        )
    elif model_type == "rf":
        logger.info("Selected Random Forest multi-class classifier with balanced class weights.")
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from 'rf', 'xgb', 'lgb'.")


# -----------------------------------------------------------------------------
# Module 4: Baselines Evaluation
# -----------------------------------------------------------------------------
def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates baseline models on the exact same hold-out test set:
    1. Majority-class DummyClassifier
    2. LogisticRegression (with balanced class weights)
    3. Transparent ADA clinical rule heuristic (FBG >= 126 or HbA1c >= 6.5 -> Diabetic)
    """
    logger.info("Evaluating comparative baseline models on hold-out test set...")

    # Baseline 1: Dummy Majority
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_acc = float(accuracy_score(y_test, y_pred_dummy))
    dummy_f1 = float(f1_score(y_test, y_pred_dummy, average="macro", zero_division=0))

    # Baseline 2: Logistic Regression (inside full preprocessor pipeline)
    log_reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
    ])
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_lr = log_reg_pipeline.predict(X_test)
    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_f1 = float(f1_score(y_test, y_pred_lr, average="macro", zero_division=0))

    # Baseline 3: Deterministic ADA Rule Heuristic
    y_pred_rule = np.zeros(len(X_test), dtype=int)
    for idx, (_, row) in enumerate(X_test.iterrows()):
        fbg = row.get("fasting_blood_glucose")
        hba1c = row.get("hba1c")
        if (hba1c is not None and not pd.isna(hba1c) and hba1c >= 6.5) or (
            fbg is not None and not pd.isna(fbg) and fbg >= 126.0
        ):
            y_pred_rule[idx] = 2  # Diabetic
        elif (hba1c is not None and not pd.isna(hba1c) and hba1c >= 5.7) or (
            fbg is not None and not pd.isna(fbg) and fbg >= 100.0
        ):
            y_pred_rule[idx] = 1  # Pre-diabetic
        else:
            y_pred_rule[idx] = 0  # Healthy

    rule_acc = float(accuracy_score(y_test, y_pred_rule))
    rule_f1 = float(f1_score(y_test, y_pred_rule, average="macro", zero_division=0))

    baselines = {
        "majority_class_dummy": {
            "model": "DummyClassifier(strategy='most_frequent')",
            "accuracy": round(dummy_acc, 4),
            "f1_macro": round(dummy_f1, 4),
            "description": "Always predicts the most frequent class in the training partition."
        },
        "logistic_regression": {
            "model": "LogisticRegression(class_weight='balanced')",
            "accuracy": round(lr_acc, 4),
            "f1_macro": round(lr_f1, 4),
            "description": "Standard linear multi-class baseline with balanced weighting."
        },
        "ada_rule_heuristic": {
            "model": "ADA Standard Threshold Heuristic",
            "accuracy": round(rule_acc, 4),
            "f1_macro": round(rule_f1, 4),
            "description": "Deterministic rule: Diabetic if FBG>=126 or HbA1c>=6.5; Pre-diabetic if FBG>=100 or HbA1c>=5.7; else Healthy."
        }
    }
    return baselines


# -----------------------------------------------------------------------------
# Module 5: Model Training, K-Fold CV & Hold-out Evaluation
# -----------------------------------------------------------------------------
def train_and_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Executes strict train/test split, k-fold cross validation, and baseline comparisons.
    """
    logger.info("Performing Stratified 80/20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    logger.info("Train partition: %d samples | Test partition: %d samples", len(X_train), len(X_test))
    preprocessor = build_preprocessing_pipeline()
    classifier = get_classifier(model_type=model_type, random_state=random_state)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # 5-Fold Stratified Cross Validation on Training Data Only
    logger.info("Executing Stratified %d-Fold Cross-Validation on training data only...", cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv_results = cross_validate(full_pipeline, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=-1)

    cv_metrics = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "cv_precision_macro_mean": float(np.mean(cv_results["test_precision_macro"])),
        "cv_recall_macro_mean": float(np.mean(cv_results["test_recall_macro"])),
    }
    logger.info(
        "K-Fold CV Results -> Accuracy: %.4f (+/- %.4f), F1-Macro: %.4f (+/- %.4f)",
        cv_metrics["cv_accuracy_mean"],
        cv_metrics["cv_accuracy_std"],
        cv_metrics["cv_f1_macro_mean"],
        cv_metrics["cv_f1_macro_std"],
    )

    # Train final model on complete training partition
    logger.info("Fitting complete pipeline on X_train...")
    full_pipeline.fit(X_train, y_train)

    # Evaluate comparative baselines
    baselines = evaluate_baselines(X_train, y_train, X_test, y_test, build_preprocessing_pipeline(), random_state=random_state)

    # Holdout inference
    logger.info("Running inference on hold-out test set...")
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)

    return full_pipeline, cv_metrics, y_pred, y_prob, X_test, y_test, X_train, y_train, baselines


def evaluate_test_performance(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cv_metrics: Dict[str, Any],
    baselines: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes hold-out test metrics and calibration loss.
    """
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Multi-class log loss as probability calibration indicator
    try:
        test_log_loss = float(log_loss(y_test, y_prob))
    except Exception:
        test_log_loss = None

    # One-vs-Rest ROC-AUC
    roc_auc_per_class = roc_auc_score(y_test, y_prob, multi_class="ovr", average=None)
    roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    roc_auc_weighted = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_dict = {
        "matrix": cm.tolist(),
        "classes": CLASS_NAMES,
        "breakdown": {
            CLASS_NAMES[i]: {
                "total_actual": int(np.sum(cm[i, :])),
                "correct_predictions": int(cm[i, i]),
                "sensitivity_recall": float(cm[i, i] / np.sum(cm[i, :])) if np.sum(cm[i, :]) > 0 else 0.0,
            }
            for i in range(len(CLASS_NAMES))
        }
    }

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "dataset_type": "synthetic",
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_macro": round(float(rec_macro), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "calibration_log_loss": round(test_log_loss, 4) if test_log_loss is not None else None,
        "roc_auc_macro": round(float(roc_auc_macro), 4),
        "roc_auc_weighted": round(float(roc_auc_weighted), 4),
        "roc_auc_per_class": {
            CLASS_NAMES[i]: round(float(roc_auc_per_class[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm_dict,
        "classification_report": report,
        "cross_validation": cv_metrics,
        "baselines_comparison": baselines,
    }

    return metrics


def display_metrics_dashboard(metrics: Dict[str, Any]):
    """Pretty-prints evaluation report with explicit synthetic data framing."""
    print("\n" + "=" * 78)
    print("   MODEL EVALUATION REPORT (EDUCATIONAL PROTOTYPE - SYNTHETIC DATASET)")
    print("=" * 78)
    print("NOTICE: Evaluated on synthetic benchmark data. Performance metrics reflect")
    print("synthetic cohort distributions and DO NOT denote real-world clinical efficacy.")
    print("-" * 78)
    print(f"Overall Hold-Out Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score (Macro Average):     {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted Average):  {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro / Weighted): {metrics['precision_macro']:.4f} / {metrics['precision_weighted']:.4f}")
    print(f"Recall (Macro / Weighted):    {metrics['recall_macro']:.4f} / {metrics['recall_weighted']:.4f}")
    if metrics.get("calibration_log_loss") is not None:
        print(f"Test Multi-Class Log Loss:    {metrics['calibration_log_loss']:.4f}")
    print("-" * 78)
    print("Comparative Baseline Models (Hold-Out Test Set):")
    for name, b in metrics["baselines_comparison"].items():
        print(f"  * {b['model']:<46} -> Acc: {b['accuracy']:.4f} | F1-Macro: {b['f1_macro']:.4f}")
    print("-" * 78)
    print("One-vs-Rest (OvR) ROC-AUC Scores:")
    for cls_name, score in metrics["roc_auc_per_class"].items():
        print(f"  * {cls_name:<16} ROC-AUC: {score:.4f}")
    print(f"  * Macro Average ROC-AUC:         {metrics['roc_auc_macro']:.4f}")
    print(f"  * Weighted Average ROC-AUC:      {metrics['roc_auc_weighted']:.4f}")
    print("-" * 78)
    print("Confusion Matrix (Actual Rows vs Predicted Columns):")
    cm = metrics["confusion_matrix"]["matrix"]
    headers = f"{'Actual \\ Pred':<16} | {'Healthy':<10} | {'Pre-diabetic':<12} | {'Diabetic':<10}"
    print(headers)
    print("-" * len(headers))
    for i, row in enumerate(cm):
        cls_name = CLASS_NAMES[i]
        print(f"{cls_name:<16} | {row[0]:<10} | {row[1]:<12} | {row[2]:<10}")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# Module 6: Explainability (XAI) with SHAP
# -----------------------------------------------------------------------------
def setup_shap_explainer(pipeline: Pipeline, X_train_sample: pd.DataFrame) -> Tuple[Any, List[str]]:
    """
    Initializes SHAP TreeExplainer using background data sampled STRICTLY from X_train.
    """
    logger.info("Initializing SHAP TreeExplainer with %d training background samples...", len(X_train_sample))
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform background dataset
    X_train_trans = preprocessor.transform(X_train_sample)

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + encoded_cat_names

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model, data=X_train_trans)
            logger.info("SHAP TreeExplainer successfully initialized.")
            return explainer, feature_names
        except Exception as e:
            logger.warning("SHAP TreeExplainer notice: %s. Using KernelExplainer fallback.", e)
            explainer = shap.KernelExplainer(model.predict_proba, X_train_trans[:50])
            return explainer, feature_names
    else:
        logger.warning("SHAP library not installed or failed to initialize.")
        return None, feature_names


def explain_single_patient(
    patient_dict: Dict[str, Any],
    pipeline: Pipeline,
    explainer: Any,
    feature_names: List[str],
    top_k: int = 6
) -> Dict[str, Any]:
    """
    Computes local SHAP attributions labeled RELATIVE TO THE EXPLAINED CLASS.
    Extracts actual base value from explainer, checks additivity, and handles
    unavailable explanations truthfully.
    """
    patient_df = pd.DataFrame([patient_dict])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    X_patient_trans = preprocessor.transform(patient_df)
    probs = model.predict_proba(X_patient_trans)[0]
    pred_class_id = int(np.argmax(probs))
    pred_class_name = CLASS_NAMES[pred_class_id]

    if not HAS_SHAP or explainer is None:
        return {
            "status": "unavailable",
            "reason": "SHAP explainer artifact is not available.",
            "explained_class": pred_class_name,
            "explained_class_id": pred_class_id,
            "output_space": "probability",
            "base_value": None,
            "confidence": round(float(probs[pred_class_id]), 4),
            "class_probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "top_features": [],
            "omitted_features_shap_sum": None,
            "additivity_check": None,
        }

    try:
        # Extract actual expected value from the explainer for pred_class_id
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(ev[pred_class_id])
        else:
            base_value = float(ev)

        raw_shap = explainer.shap_values(X_patient_trans)
        if isinstance(raw_shap, list):
            class_shap_vals = raw_shap[pred_class_id][0]
        elif hasattr(raw_shap, "values"):
            class_shap_vals = raw_shap.values[0, :, pred_class_id]
        elif raw_shap.ndim == 3:
            class_shap_vals = raw_shap[0, :, pred_class_id]
        else:
            class_shap_vals = raw_shap[0]

        # Aggregate one-hot dummy features back into parent feature groups
        feature_impacts: Dict[str, float] = {}
        for name, s_val in zip(feature_names, class_shap_vals):
            matched_cat = next((cat for cat in CATEGORICAL_FEATURES if name.startswith(cat)), None)
            root = matched_cat if matched_cat else name
            feature_impacts[root] = feature_impacts.get(root, 0.0) + float(s_val)

        # Validate additivity: base_value + sum(all_features) ≈ model output
        all_shap_sum = float(sum(feature_impacts.values()))
        reconstructed = base_value + all_shap_sum
        model_prob = float(probs[pred_class_id])
        additivity_diff = abs(reconstructed - model_prob)

        # Sort features by absolute contribution magnitude
        sorted_feats = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for feat, val in sorted_feats[:top_k]:
            patient_val = patient_dict.get(feat, "N/A")
            # Label relative to explained class
            if val > 0:
                impact_label = f"Supports {pred_class_name} prediction"
                direction = "supports"
            elif val < 0:
                impact_label = f"Opposes {pred_class_name} prediction"
                direction = "opposes"
            else:
                impact_label = f"Neutral regarding {pred_class_name}"
                direction = "neutral"

            top_features.append({
                "feature": feat,
                "patient_value": patient_val,
                "shap_value": round(float(val), 4),
                "impact": impact_label,
                "direction": direction,
            })

        top_sum = float(sum(f["shap_value"] for f in top_features))
        omitted_sum = round(all_shap_sum - top_sum, 4)

        return {
            "status": "available",
            "explained_class": pred_class_name,
            "explained_class_id": pred_class_id,
            "output_space": "probability",
            "base_value": round(base_value, 4),
            "confidence": round(float(probs[pred_class_id]), 4),
            "class_probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "top_features": top_features,
            "omitted_features_shap_sum": omitted_sum,
            "additivity_check": {
                "base_value": round(base_value, 4),
                "all_features_shap_sum": round(all_shap_sum, 4),
                "reconstructed_output": round(reconstructed, 4),
                "model_output": round(model_prob, 4),
                "absolute_difference": round(additivity_diff, 5),
                "is_additive": bool(additivity_diff < 0.05),
            }
        }
    except Exception as err:
        logger.warning("SHAP explanation failed: %s", err)
        return {
            "status": "unavailable",
            "reason": f"Calculation error: {str(err)}",
            "explained_class": pred_class_name,
            "explained_class_id": pred_class_id,
            "output_space": "probability",
            "base_value": None,
            "confidence": round(float(probs[pred_class_id]), 4),
            "class_probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "top_features": [],
            "omitted_features_shap_sum": None,
            "additivity_check": None,
        }


# -----------------------------------------------------------------------------
# Module 7: Full Pipeline Execution and Artifact Persistence
# -----------------------------------------------------------------------------
def run_pipeline(
    data_path: Optional[str] = None,
    output_dir: str = "artifacts",
    model_type: str = "rf",
    n_samples: int = 4000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Main orchestration routine.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ingest Data or explicitly fail if missing
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Explicitly supplied dataset path '{data_path}' was not found.")
        logger.info("Loading patient cohort dataset from supplied path: %s", data_path)
        df = pd.read_csv(data_path)
        dataset_source = f"external_file: {data_path}"
    else:
        logger.info("No external dataset specified. Generating ADA synthetic clinical benchmark cohort...")
        df = generate_synthetic_clinical_cohort(n_samples=n_samples, random_state=random_state)
        csv_save_path = os.path.join(output_dir, "diabetes_clinical_dataset.csv")
        df.to_csv(csv_save_path, index=False)
        logger.info("Synthetic benchmark dataset saved to %s", csv_save_path)
        dataset_source = "synthetic_ada_benchmark"

    target_col = "diabetes_stage"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Columns: {df.columns.tolist()}")

    X = df[ALL_FEATURES]
    y = df[target_col].astype(int)

    # Train, Cross-Validate, and Evaluate Baselines
    (
        pipeline,
        cv_metrics,
        y_pred,
        y_prob,
        X_test,
        y_test,
        X_train,
        y_train,
        baselines
    ) = train_and_validate_pipeline(
        X, y, model_type=model_type, cv_folds=5, random_state=random_state
    )

    # Evaluate Test Metrics
    metrics = evaluate_test_performance(y_test, y_pred, y_prob, cv_metrics, baselines)
    display_metrics_dashboard(metrics)

    # Setup SHAP Explainer (Sample strictly from X_train to prevent leakage)
    X_train_bg = X_train.sample(n=min(150, len(X_train)), random_state=random_state)
    explainer, feature_names = setup_shap_explainer(pipeline, X_train_bg)

    # Verify Single Patient Explanation Demo
    sample_patient = {
        "age": 56.0,
        "gender": "Male",
        "ethnicity": "Hispanic",
        "family_history_diabetes": 1,
        "bmi": 31.4,
        "fasting_blood_glucose": 138.0,
        "hba1c": 6.8,
        "total_cholesterol": 235.0,
        "hdl_cholesterol": 38.0,
        "ldl_cholesterol": 145.0,
        "triglycerides": 210.0,
        "systolic_bp": 142.0,
        "diastolic_bp": 88.0,
        "serum_creatinine": 1.1,
        "physical_activity_minutes_per_week": 45.0,
        "smoking_status": "Former",
        "alcohol_intake": "Moderate",
        "diet_quality_score": 4.5,
        "sleep_hours": 6.0,
    }
    demo_explanation = explain_single_patient(sample_patient, pipeline, explainer, feature_names)
    logger.info("Sample Patient Inference Demo: Class=%s (Conf: %.2f)",
                demo_explanation["explained_class"], demo_explanation["confidence"])
    for f in demo_explanation["top_features"]:
        logger.info("   -> Feature '%s' (value=%s, SHAP=%s: %s)", f["feature"], f["patient_value"], f["shap_value"], f["impact"])

    # Persist Artifacts
    pipeline_path = os.path.join(output_dir, "model_pipeline.joblib")
    explainer_path = os.path.join(output_dir, "explainer.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    metadata_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(pipeline, pipeline_path)
    if explainer is not None:
        try:
            joblib.dump(explainer, explainer_path)
        except Exception as e:
            logger.warning("Could not serialize explainer: %s", e)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "model_version": "1.1.0-educational",
        "dataset_source": dataset_source,
        "dataset_type": "synthetic",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": {CLASS_NAMES[k]: int(v) for k, v in y.value_counts().items()},
        "split_strategy": "stratified_80_20",
        "random_seed": random_state,
        "model_type": model_type,
        "estimator": type(pipeline.named_steps["classifier"]).__name__,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "encoded_feature_names": feature_names,
        "class_mapping": CLASS_MAP,
        "classes": CLASS_NAMES,
        "physiological_bounds": PHYSIOLOGICAL_BOUNDS,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Pipeline execution complete! Artifacts saved in %s", output_dir)
    return metrics


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Diabetes Risk ML Pipeline (Educational Prototype)")
    parser.add_argument("--data", type=str, default=None, help="Path to clinical CSV dataset")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory for saved artifacts")
    parser.add_argument("--model", type=str, choices=["rf", "xgb", "lgb"], default="rf", help="Classifier algorithm")
    parser.add_argument("--samples", type=int, default=4000, help="Number of cohort records if generating synthetic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    run_pipeline(
        data_path=args.data,
        output_dir=args.out,
        model_type=args.model,
        n_samples=args.samples,
        random_state=args.seed
    )
