"""
ml_pipeline.py - Real-World CDC BRFSS Diabetes Risk Stratification Pipeline
=============================================================================
Scope & Purpose:
This pipeline implements data ingestion, leakage-free preprocessing, multi-class model
training, stratified 5-fold cross-validation, comparative baselines, hold-out test
evaluation, and SHAP explainability on the real-world CDC Behavioral Risk Factor
Surveillance System (BRFSS) multi-class dataset.

Data Provenance & Ethical Framing:
- Source: U.S. Centers for Disease Control and Prevention (CDC) BRFSS 2015 Survey.
- Target Definition: 3-Class Glycemic Status (DIABETE3)
  * Class 0 (Healthy / No diabetes): DIABETE3 == 3.0
  * Class 1 (Pre-diabetic / Borderline): DIABETE3 == 4.0
  * Class 2 (Diabetic): DIABETE3 == 1.0
- Features: 21 genuine demographic, socioeconomic, and lifestyle health indicators.
- Real-World Validation: Replaces synthetic benchmark with genuine public health surveillance data.
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
from sklearn.preprocessing import RobustScaler
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

from clinical_transformers import PhysiologicalOutlierCapper, BRFSS_FEATURE_BOUNDS
from ingest_brfss import (
    load_brfss_cohort,
    BRFSS_CLEAN_FEATURES,
    TARGET_COL,
    clean_brfss_dataframe,
    BRFSS_RAW_COLS
)

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
    "bmi",
    "gen_hlth",
    "ment_hlth",
    "phys_hlth",
    "age",
    "education",
    "income",
]

BINARY_FEATURES = [
    "high_bp",
    "high_chol",
    "chol_check",
    "smoker",
    "stroke",
    "heart_disease_or_attack",
    "phys_activity",
    "fruits",
    "veggies",
    "hvy_alcohol_consump",
    "any_healthcare",
    "no_doc_bc_cost",
    "diff_walk",
    "sex",
]

ALL_FEATURES = BINARY_FEATURES + NUMERICAL_FEATURES

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "high_bp": "High Blood Pressure",
    "high_chol": "High Blood Cholesterol",
    "chol_check": "Cholesterol Checked in Past 5 Years",
    "bmi": "Body Mass Index (BMI)",
    "smoker": "Smoking History (>=100 Cigarettes)",
    "stroke": "History of Stroke",
    "heart_disease_or_attack": "Coronary Heart Disease or Heart Attack",
    "phys_activity": "Physical Activity in Past 30 Days",
    "fruits": "Daily Fruit Consumption (>=1/day)",
    "veggies": "Daily Vegetable Consumption (>=1/day)",
    "hvy_alcohol_consump": "Heavy Alcohol Consumption",
    "any_healthcare": "Healthcare Coverage",
    "no_doc_bc_cost": "Cost Barrier to Medical Care",
    "gen_hlth": "Self-Rated General Health",
    "ment_hlth": "Days of Poor Mental Health (0-30)",
    "phys_hlth": "Days of Poor Physical Health (0-30)",
    "diff_walk": "Difficulty Walking or Climbing Stairs",
    "sex": "Biological Sex",
    "age": "Age Category (18-24 to 80+)",
    "education": "Education Level",
    "income": "Household Income Bracket",
}


# -----------------------------------------------------------------------------
# Module 2: Preprocessing Pipeline Architecture
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs a leakage-free Scikit-Learn preprocessing pipeline for CDC BRFSS data.
    - Numerical / Ordinal Features: Outlier Capper + RobustScaler.
    - Binary Features: Direct passthrough (features are already standard 0/1 integers).
    """
    numerical_transformer = Pipeline([
        ("capper", PhysiologicalOutlierCapper(bounds=BRFSS_FEATURE_BOUNDS)),
        ("scaler", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


# -----------------------------------------------------------------------------
# Module 3: Model Instantiation
# -----------------------------------------------------------------------------
def get_classifier(model_type: str = "rf", random_state: int = 42) -> BaseEstimator:
    """Returns the requested classifier with balanced class weighting."""
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose 'rf' or 'lr'.")


# -----------------------------------------------------------------------------
# Module 4: Comparative Baselines
# -----------------------------------------------------------------------------
class CDCEpidemiologicalHeuristicBaseline(BaseEstimator):
    """
    Transparent rule-based epidemiological heuristic:
    - Diabetic (2) if High BP == 1, High Chol == 1, BMI >= 30, and Age >= 9 (60+).
    - Pre-diabetic (1) if High BP == 1, High Chol == 1, and BMI >= 27.
    - Otherwise Healthy (0).
    """
    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=ALL_FEATURES)
        preds = np.zeros(len(X), dtype=int)
        high_bp = X["high_bp"].values
        high_chol = X["high_chol"].values
        bmi = X["bmi"].values
        age = X["age"].values

        prediab_mask = (high_bp == 1) & (high_chol == 1) & (bmi >= 27.0)
        diab_mask = (high_bp == 1) & (high_chol == 1) & (bmi >= 30.0) & (age >= 9)

        preds[prediab_mask] = 1
        preds[diab_mask] = 2
        return preds


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluates comparative baselines on the hold-out test set."""
    baselines_results = {}

    # 1. Majority Class Dummy Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    baselines_results["majority_class_dummy"] = {
        "model": "DummyClassifier(strategy='most_frequent')",
        "accuracy": float(np.round(accuracy_score(y_test, dummy_pred), 4)),
        "f1_macro": float(np.round(f1_score(y_test, dummy_pred, average="macro", zero_division=0), 4)),
        "description": "Always predicts the most frequent training class (Healthy)."
    }

    # 2. Logistic Regression Baseline
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    baselines_results["logistic_regression"] = {
        "model": "LogisticRegression(class_weight='balanced')",
        "accuracy": float(np.round(accuracy_score(y_test, lr_pred), 4)),
        "f1_macro": float(np.round(f1_score(y_test, lr_pred, average="macro", zero_division=0), 4)),
        "description": "Standard linear multi-class baseline with balanced weighting."
    }

    # 3. CDC Epidemiological Rule Heuristic
    heuristic = CDCEpidemiologicalHeuristicBaseline()
    h_pred = heuristic.predict(X_test)
    baselines_results["cdc_rule_heuristic"] = {
        "model": "CDC Epidemiological Risk Heuristic",
        "accuracy": float(np.round(accuracy_score(y_test, h_pred), 4)),
        "f1_macro": float(np.round(f1_score(y_test, h_pred, average="macro", zero_division=0), 4)),
        "description": "Deterministic rule: Diabetic if High BP & Chol with BMI>=30 & Age>=60; Pre-diabetic if High BP & Chol with BMI>=27; else Healthy."
    }

    return baselines_results


# -----------------------------------------------------------------------------
# Module 5: Training and Cross-Validation
# -----------------------------------------------------------------------------
def train_and_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Executes strict 80/20 train/test split, 5-fold CV, and baseline benchmarking."""
    logger.info("Executing Stratified 80/20 Train/Test Split...")
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

    # 5-Fold Stratified Cross-Validation on Training Partition Only
    logger.info("Running Stratified %d-Fold Cross-Validation on X_train only...", cv_folds)
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

    # Fit complete pipeline on complete training partition
    logger.info("Fitting complete pipeline on X_train...")
    full_pipeline.fit(X_train, y_train)

    # Evaluate comparative baselines
    baselines = evaluate_baselines(X_train, y_train, X_test, y_test, build_preprocessing_pipeline(), random_state=random_state)

    # Holdout test inference
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
    """Computes test set performance metrics, calibration log loss, and confusion matrix."""
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Calibration loss
    calib_loss = float(log_loss(y_test, y_prob))

    # Multi-class ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
        roc_per_class = {}
        for idx, cls_name in enumerate(CLASS_NAMES):
            y_binary = (y_test == idx).astype(int)
            roc_per_class[cls_name] = float(np.round(roc_auc_score(y_binary, y_prob[:, idx]), 4))
    except Exception as e:
        logger.warning("ROC-AUC calculation failed: %s", e)
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0
        roc_per_class = {cls: 0.0 for cls in CLASS_NAMES}

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    cm_breakdown = {}
    for idx, cls_name in enumerate(CLASS_NAMES):
        total_actual = int(cm[idx, :].sum())
        correct = int(cm[idx, idx])
        sens = float(correct / total_actual) if total_actual > 0 else 0.0
        cm_breakdown[cls_name] = {
            "total_actual": total_actual,
            "correct_predictions": correct,
            "sensitivity_recall": sens
        }

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return {
        "dataset_type": "real_cdc_brfss_2015",
        "sample_count": len(y_test),
        "accuracy": float(np.round(acc, 4)),
        "precision_macro": float(np.round(prec_macro, 4)),
        "precision_weighted": float(np.round(prec_weighted, 4)),
        "recall_macro": float(np.round(rec_macro, 4)),
        "recall_weighted": float(np.round(rec_weighted, 4)),
        "f1_macro": float(np.round(f1_m, 4)),
        "f1_weighted": float(np.round(f1_w, 4)),
        "calibration_log_loss": float(np.round(calib_loss, 4)),
        "roc_auc_macro": float(np.round(roc_auc_macro, 4)),
        "roc_auc_weighted": float(np.round(roc_auc_weighted, 4)),
        "roc_auc_per_class": roc_per_class,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "classes": CLASS_NAMES,
            "breakdown": cm_breakdown
        },
        "classification_report": report,
        "cross_validation": cv_metrics,
        "baselines_comparison": baselines
    }


# -----------------------------------------------------------------------------
# Module 6: SHAP Explainability Architecture
# -----------------------------------------------------------------------------
def setup_shap_explainer(pipeline: Pipeline, X_train_sample: pd.DataFrame) -> Tuple[Any, List[str]]:
    """Fits TreeExplainer on holdout-free training background sample."""
    if not HAS_SHAP:
        logger.warning("SHAP is not installed. Explainer setup skipped.")
        return None, []

    logger.info("Initializing SHAP TreeExplainer on holdout-free training background...")
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train_trans = preprocessor.transform(X_train_sample)
    feature_names = ALL_FEATURES

    explainer = shap.TreeExplainer(
        classifier,
        data=shap.sample(X_train_trans, 100),
        model_output="probability",
        feature_perturbation="interventional"
    )
    return explainer, feature_names


# -----------------------------------------------------------------------------
# Module 7: Full Pipeline Orchestration
# -----------------------------------------------------------------------------
def run_pipeline(
    data_path: Optional[str] = None,
    output_dir: str = "artifacts",
    model_type: str = "rf",
    sample_size: int = 50000,
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs data loading, training, baseline comparisons, evaluation, and persists artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingest Data
    raw_path = data_path or "archive/2015.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"CDC BRFSS dataset '{raw_path}' was not found.")

    logger.info("Ingesting CDC BRFSS dataset from %s (sample size: %d)...", raw_path, sample_size)
    df = load_brfss_cohort(raw_csv_path=raw_path, sample_size=sample_size, random_state=random_state)

    clean_csv_path = os.path.join(output_dir, "cdc_brfss_diabetes_clean.csv")
    df.to_csv(clean_csv_path, index=False)
    logger.info("Cleaned cohort saved to %s", clean_csv_path)

    X = df[ALL_FEATURES]
    y = df[TARGET_COL].astype(int)

    # 2. Train and Validate
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

    # 3. Evaluate Holdout Metrics
    metrics = evaluate_test_performance(y_test, y_pred, y_prob, cv_metrics, baselines)

    # 4. Fit SHAP Explainer
    explainer, feature_names = setup_shap_explainer(pipeline, X_train.sample(n=200, random_state=random_state))

    # 5. Persist Artifacts
    pipeline_path = os.path.join(output_dir, "model_pipeline.joblib")
    explainer_path = os.path.join(output_dir, "explainer.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    metadata_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved pipeline to %s", pipeline_path)

    if explainer is not None:
        joblib.dump(explainer, explainer_path)
        logger.info("Saved SHAP explainer to %s", explainer_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)

    metadata = {
        "model_version": "2.0.0-brfss",
        "estimator": type(pipeline.named_steps["classifier"]).__name__,
        "dataset_name": "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015",
        "dataset_type": "real_cdc_brfss_2015",
        "provenance": "Centers for Disease Control and Prevention (CDC), Public Domain",
        "n_samples_total": len(df),
        "n_features": len(ALL_FEATURES),
        "features": ALL_FEATURES,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "classes": CLASS_NAMES,
        "cv_folds": 5,
        "random_state": random_state
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDC BRFSS Diabetes Risk Stratification Pipeline")
    parser.add_argument("--data", type=str, default="archive/2015.csv", help="Path to raw or cleaned BRFSS CSV")
    parser.add_argument("--samples", type=int, default=50000, help="Number of records to sample")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"], help="Estimator type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    metrics = run_pipeline(
        data_path=args.data,
        output_dir=args.outdir,
        model_type=args.model,
        sample_size=args.samples,
        random_state=args.seed
    )
    print("Pipeline Execution Completed Successfully.")
    print("Hold-Out Test Accuracy:", metrics["accuracy"])
    print("Hold-Out Test Macro F1:", metrics["f1_macro"])
