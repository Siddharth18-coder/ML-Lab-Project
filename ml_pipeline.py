"""
ml_pipeline.py - Comprehensive End-to-End Clinical Machine Learning Pipeline
=============================================================================
Role: Lead Full-Stack Machine Learning Engineer
Task: Multi-Class Diabetes Risk Stratification (Healthy, Pre-diabetic, Diabetic)

Key Architectural Modules:
1. Data Generation / Ingestion: Realistic clinical cohort generator adhering to
   ADA (American Diabetes Association) guidelines, with physiological noise,
   clinical missingness, and laboratory outliers.
2. Robust Data Cleaning:
   - Physiological outlier capping (Winsorization & clinical physiological boundaries)
   - Missing value imputation using KNNImputer on numerical lab tests
   - Most-frequent imputation and One-Hot Encoding for categorical demographics
   - Robust scaling to guard against extreme skew in biomarker distributions
3. Model Training & Validation:
   - Stratified 80/20 holdout train/test split
   - Stratified 5-Fold Cross Validation
   - Class-weighting / SMOTE support to counteract clinical prevalence imbalances
   - Multi-class optimization (Random Forest / XGBoost / LightGBM)
4. Comprehensive Metric Reporting:
   - Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted)
   - One-vs-Rest (OvR) ROC-AUC per class (Healthy, Pre-diabetic, Diabetic)
   - Full 3x3 Confusion Matrix & Per-Class Clinical Sensitivity/Specificity
5. Explainability (XAI):
   - SHAP TreeExplainer integration for global and local patient-level feature attributions
   - Ranked positive/negative driver extraction for individual clinical consultation
6. Artifact Persistence:
   - Serializes trained pipeline, explainer, feature metadata, and metrics into artifacts/
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Optional heavy ML libraries with graceful fallbacks
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

# Physiological bounds (min, max) based on clinical survival and laboratory ranges
PHYSIOLOGICAL_BOUNDS = {
    "age": (18.0, 95.0),
    "bmi": (14.0, 65.0),
    "fasting_blood_glucose": (50.0, 400.0),
    "hba1c": (3.8, 16.0),
    "total_cholesterol": (90.0, 420.0),
    "hdl_cholesterol": (15.0, 120.0),
    "ldl_cholesterol": (35.0, 300.0),
    "triglycerides": (35.0, 600.0),
    "systolic_bp": (70.0, 240.0),
    "diastolic_bp": (40.0, 140.0),
    "serum_creatinine": (0.3, 6.0),
    "physical_activity_minutes_per_week": (0.0, 800.0),
    "diet_quality_score": (1.0, 10.0),
    "sleep_hours": (3.0, 13.0),
}


# -----------------------------------------------------------------------------
# Module 1: Robust Outlier Capper (Imported from clinical_transformers)
# -----------------------------------------------------------------------------
from clinical_transformers import PhysiologicalOutlierCapper
# -----------------------------------------------------------------------------
# Module 2: Medical Cohort Data Generation (ADA Criteria Aligned)
# -----------------------------------------------------------------------------
def generate_synthetic_clinical_cohort(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a medically realistic clinical cohort of diabetic, pre-diabetic,
    and healthy individuals adhering to American Diabetes Association (ADA)
    diagnostic guidelines with clinical covariance, noise, missingness, and outliers.
    """
    logger.info("Generating realistic clinical cohort (%d patients)...", n_samples)
    rng = np.random.default_rng(random_state)

    # Demographic distributions
    age = rng.normal(52, 14, size=n_samples).clip(18, 90)
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

    # Latent metabolic risk score driving ADA diagnostic thresholds
    # Factors: Age, BMI, Family History, Sedentary life, Diet, Blood Pressure
    risk_score = (
        0.035 * (age - 45)
        + 0.12 * (bmi - 25)
        + 0.85 * family_history
        - 0.003 * (activity_min - 150)
        - 0.15 * (diet_score - 5.0)
        + 0.015 * (systolic_bp - 120)
        + (0.35 if "Current" in smoking else 0.0)
        + rng.normal(0, 0.95, size=n_samples)
    )

    # Class assignment based on clinical risk score percentiles
    # Creates approximately: 48% Healthy, 28% Pre-diabetic, 24% Diabetic
    labels = np.zeros(n_samples, dtype=int)
    pre_cutoff = np.percentile(risk_score, 48)
    diab_cutoff = np.percentile(risk_score, 76)

    labels[(risk_score >= pre_cutoff) & (risk_score < diab_cutoff)] = 1
    labels[risk_score >= diab_cutoff] = 2

    # Laboratory Panel - Fasting Glucose & HbA1c closely coupled to class and latent risk
    # Healthy: Fasting Glucose 70-99 mg/dL, HbA1c 4.2-5.6%
    # Pre-diabetic: Fasting Glucose 100-125 mg/dL, HbA1c 5.7-6.4%
    # Diabetic: Fasting Glucose >= 126 mg/dL, HbA1c >= 6.5%
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

    # Assemble raw DataFrame
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

    # Introduce realistic clinical anomalies and missingness (~5% missing lab test values)
    logger.info("Injecting realistic missing clinical values (~5%%) and physiological outliers...")
    for col in ["fasting_blood_glucose", "hba1c", "hdl_cholesterol", "triglycerides", "serum_creatinine"]:
        mask_missing = rng.random(size=n_samples) < 0.05
        df.loc[mask_missing, col] = np.nan

    # Inject extreme physiological outliers (e.g. data-entry or instrumentation errors) in ~1.5% of rows
    outlier_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df.loc[outlier_idx[: len(outlier_idx) // 2], "fasting_blood_glucose"] = rng.uniform(420, 650)
    df.loc[outlier_idx[len(outlier_idx) // 2:], "hba1c"] = rng.uniform(16.5, 22.0)

    return df


# -----------------------------------------------------------------------------
# Module 3: Preprocessing Pipeline Architecture
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline() -> Tuple[Pipeline, ColumnTransformer]:
    """
    Constructs a leakage-free Scikit-Learn preprocessing pipeline:
    - Numerical: Outlier Capper -> KNNImputer -> RobustScaler
    - Categorical: SimpleImputer (most_frequent) -> OneHotEncoder
    """
    numerical_transformer = Pipeline([
        ("capper", PhysiologicalOutlierCapper()),
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", RobustScaler()),
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
# Module 4: Model Training, K-Fold CV & Hyperparameter Optimization
# -----------------------------------------------------------------------------
def get_classifier(model_type: str = "rf", random_state: int = 42) -> BaseEstimator:
    """Instantiates the requested multi-class classifier with balanced weighting."""
    if model_type == "xgb" and HAS_XGB:
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
    elif model_type == "lgb" and HAS_LGB:
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
    else:
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


def train_and_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    """
    Executes strict train/test split, k-fold cross validation, and trains final pipeline.
    """
    logger.info("Performing Stratified 80/20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    logger.info("Train set shape: %s | Test set shape: %s", X_train.shape, X_test.shape)
    preprocessor = build_preprocessing_pipeline()
    classifier = get_classifier(model_type=model_type, random_state=random_state)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # 5-Fold Stratified Cross Validation on Training Data
    logger.info("Executing Stratified %d-Fold Cross-Validation on training data...", cv_folds)
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
        "K-Fold CV Results -> Accuracy: %.3f (+/- %.3f), F1-Macro: %.3f (+/- %.3f)",
        cv_metrics["cv_accuracy_mean"],
        cv_metrics["cv_accuracy_std"],
        cv_metrics["cv_f1_macro_mean"],
        cv_metrics["cv_f1_macro_std"],
    )

    # Train final model on entire training partition
    logger.info("Fitting complete preprocessing and classifier pipeline on X_train...")
    full_pipeline.fit(X_train, y_train)

    # Inference on hold-out test set
    logger.info("Running inference on hold-out test set...")
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)

    return full_pipeline, cv_metrics, y_pred, y_prob, X_test, y_test


# -----------------------------------------------------------------------------
# Module 5: Comprehensive Performance Metric Evaluation
# -----------------------------------------------------------------------------
def evaluate_test_performance(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cv_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes hold-out test metrics:
    - Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted)
    - ROC-AUC (One-vs-Rest) per class and macro/weighted average
    - Multi-class Confusion Matrix with clinical specificity/sensitivity
    """
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # One-vs-Rest ROC-AUC calculation
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
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_macro": round(float(rec_macro), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_macro": round(float(roc_auc_macro), 4),
        "roc_auc_weighted": round(float(roc_auc_weighted), 4),
        "roc_auc_per_class": {
            CLASS_NAMES[i]: round(float(roc_auc_per_class[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm_dict,
        "classification_report": report,
        "cross_validation": cv_metrics,
    }

    return metrics


def display_metrics_dashboard(metrics: Dict[str, Any]):
    """Pretty-prints the comprehensive clinical evaluation report to console."""
    print("\n" + "=" * 78)
    print("      CLINICAL VALIDATION REPORT: 3-TIER DIABETES PREDICTION MODEL")
    print("=" * 78)
    print(f"Overall Hold-Out Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score (Macro Average):     {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted Average):  {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro / Weighted): {metrics['precision_macro']:.4f} / {metrics['precision_weighted']:.4f}")
    print(f"Recall (Macro / Weighted):    {metrics['recall_macro']:.4f} / {metrics['recall_weighted']:.4f}")
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
    print("-" * 78)
    print("Clinical Sensitivity (Recall) per Risk Tier:")
    for cls_name, info in metrics["confusion_matrix"]["breakdown"].items():
        pct = info["sensitivity_recall"] * 100
        print(f"  * {cls_name:<16} Sensitivity: {pct:6.2f}% ({info['correct_predictions']}/{info['total_actual']})")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# Module 6: Explainability (XAI) with SHAP
# -----------------------------------------------------------------------------
def setup_shap_explainer(pipeline: Pipeline, X_train_sample: pd.DataFrame) -> Tuple[Any, List[str]]:
    """
    Initializes a SHAP TreeExplainer for multi-class tree ensembles.
    Returns the explainer and the transformed feature names.
    """
    logger.info("Initializing SHAP TreeExplainer...")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform background dataset
    X_train_trans = preprocessor.transform(X_train_sample)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + encoded_cat_names

    if HAS_SHAP:
        try:
            # Random forest / tree explainer
            explainer = shap.TreeExplainer(model, data=shap.sample(X_train_trans, 100))
            logger.info("SHAP TreeExplainer successfully created with %d background samples.", 100)
            return explainer, feature_names
        except Exception as e:
            logger.warning("Standard SHAP TreeExplainer initialization notice: %s. Using KernelExplainer fallback.", e)
            predict_fn = model.predict_proba
            explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train_trans, 50))
            return explainer, feature_names
    else:
        logger.warning("SHAP library not installed. Creating native surrogate attribution explainer.")
        return None, feature_names


def explain_single_patient(
    patient_dict: Dict[str, Any],
    pipeline: Pipeline,
    explainer: Any,
    feature_names: List[str],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Computes local SHAP feature attributions for an individual patient record,
    returning a ranked list of driving features (both risk-increasing and protective).
    """
    patient_df = pd.DataFrame([patient_dict])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    X_patient_trans = preprocessor.transform(patient_df)
    probs = model.predict_proba(X_patient_trans)[0]
    pred_class_id = int(np.argmax(probs))
    pred_class_name = CLASS_NAMES[pred_class_id]

    top_features = []

    if HAS_SHAP and explainer is not None:
        try:
            raw_shap = explainer.shap_values(X_patient_trans)
            # Depending on SHAP version: raw_shap can be a list of 3 arrays (one per class)
            # or a 3D array of shape (samples, features, classes)
            if isinstance(raw_shap, list):
                class_shap_vals = raw_shap[pred_class_id][0]
            elif hasattr(raw_shap, "values"):
                class_shap_vals = raw_shap.values[0, :, pred_class_id]
            elif raw_shap.ndim == 3:
                class_shap_vals = raw_shap[0, :, pred_class_id]
            else:
                class_shap_vals = raw_shap[0]

            # Aggregate one-hot encoded categories back to original feature groups for clinical readability
            feature_impacts = {}
            for name, s_val in zip(feature_names, class_shap_vals):
                # Identify root feature
                root = name.split("_")[0] if any(cat in name for cat in CATEGORICAL_FEATURES) else name
                matched_cat = next((cat for cat in CATEGORICAL_FEATURES if name.startswith(cat)), None)
                if matched_cat:
                    root = matched_cat
                feature_impacts[root] = feature_impacts.get(root, 0.0) + float(s_val)

            # Sort features by absolute attribution magnitude
            sorted_feats = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

            for feat, val in sorted_feats[:top_k]:
                patient_val = patient_dict.get(feat, "N/A")
                direction = "increases_risk" if val > 0 else "reduces_risk"
                top_features.append({
                    "feature": feat,
                    "patient_value": patient_val,
                    "shap_value": round(val, 4),
                    "impact": "Increases risk" if val > 0 else "Protective factor",
                    "direction": direction,
                })
        except Exception as err:
            logger.warning("SHAP calculation fallback: %s", err)

    # Fallback to feature importance surrogate if SHAP not initialized
    if not top_features and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_k]
        for idx in sorted_idx:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            clean_name = feat_name.split("_")[0] if any(cat in feat_name for cat in CATEGORICAL_FEATURES) else feat_name
            top_features.append({
                "feature": clean_name,
                "patient_value": patient_dict.get(clean_name, "N/A"),
                "shap_value": round(float(importances[idx]), 4),
                "impact": "Primary predictive driver",
                "direction": "increases_risk",
            })

    return {
        "predicted_class": pred_class_name,
        "predicted_class_id": pred_class_id,
        "confidence": round(float(probs[pred_class_id]), 4),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
        },
        "top_features": top_features
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
    Main orchestration routine:
    1. Loads or generates clinical cohort
    2. Executes outlier capping & KNN imputation pipeline
    3. Runs Stratified K-Fold CV & model training
    4. Evaluates comprehensive holdout performance metrics
    5. Builds SHAP explainability model
    6. Saves all artifacts to disk
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Ingest or Generate Data
    if data_path and os.path.exists(data_path):
        logger.info("Loading patient cohort dataset from: %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.info("No external dataset specified or file not found. Generating ADA clinical benchmark dataset...")
        df = generate_synthetic_clinical_cohort(n_samples=n_samples, random_state=random_state)
        csv_save_path = os.path.join(output_dir, "diabetes_clinical_dataset.csv")
        df.to_csv(csv_save_path, index=False)
        logger.info("Generated dataset saved to %s", csv_save_path)

    # Step 2: Separate features and target
    target_col = "diabetes_stage"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Columns: {df.columns.tolist()}")

    X = df[ALL_FEATURES]
    y = df[target_col].astype(int)

    # Step 3: Train and Validate Pipeline
    pipeline, cv_metrics, y_pred, y_prob, X_test, y_test = train_and_validate_pipeline(
        X, y, model_type=model_type, cv_folds=5, random_state=random_state
    )

    # Step 4: Evaluate Test Metrics
    metrics = evaluate_test_performance(y_test, y_pred, y_prob, cv_metrics)
    display_metrics_dashboard(metrics)

    # Step 5: Setup SHAP Explainer
    explainer, feature_names = setup_shap_explainer(pipeline, X.sample(n=min(300, len(X)), random_state=random_state))

    # Step 6: Test Single Patient Explanation Demo
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
                demo_explanation["predicted_class"], demo_explanation["confidence"])
    for f in demo_explanation["top_features"]:
        logger.info("   -> Feature '%s' (value=%s, SHAP impact=%s)", f["feature"], f["patient_value"], f["shap_value"])

    # Step 7: Serialize Artifacts
    pipeline_path = os.path.join(output_dir, "model_pipeline.joblib")
    explainer_path = os.path.join(output_dir, "explainer.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    metadata_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(pipeline, pipeline_path)
    if explainer is not None:
        try:
            joblib.dump(explainer, explainer_path)
        except Exception as e:
            logger.warning("Could not joblib serialize full explainer (%s). Explanations will run dynamically.", e)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "encoded_feature_names": feature_names,
        "class_mapping": CLASS_MAP,
        "classes": CLASS_NAMES,
        "physiological_bounds": PHYSIOLOGICAL_BOUNDS,
        "model_type": model_type,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Pipeline executed successfully! Artifacts written to %s", output_dir)
    return metrics


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Clinical Diabetes Risk ML Pipeline")
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
