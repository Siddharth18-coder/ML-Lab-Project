"""
api.py - CDC BRFSS Diabetes Risk Stratification & Explainability REST API
==========================================================================
Data Provenance & Scope:
Delivers multi-class diabetes risk classification (No Reported Diabetes, Pre-diabetic, Diabetic)
trained directly on the real-world U.S. Centers for Disease Control and Prevention (CDC)
Behavioral Risk Factor Surveillance System (BRFSS) dataset.

Key Architectural Guarantees:
1. Strict Pydantic v2 validation adhering to CDC BRFSS discrete allowed values and bounds.
2. Verified TreeSHAP explainability:
   - Features mapped using exact transformed order from ColumnTransformer.get_feature_names_out().
   - Classes mapped dynamically using classifier.classes_.
   - Base value dynamically extracted from explainer for the chosen class (no fabricated fallbacks).
   - Attributions labeled relative to the explained class ('Supports X' vs 'Opposes X').
   - Additivity check: base_value + sum(shap_values) == model probability output (tol=0.05).
   - If additivity verification fails, returns explicit 'failed_verification' state.
3. Decision Rules & Calibration Transparency:
   - Supports 'argmax' (default max-probability) and 'threshold_tuned' (validation-tuned screening cutoffs).
   - When threshold rule triggers a non-maximal class, confidence represents the class's assigned probability.
   - Explanations explicitly identify the output space and base estimator.
4. Truthful operational status:
   - Returns HTTP 503 if model artifacts are missing, corrupt, or incompatible.
   - Structured field-level error messages on validation failure (HTTP 422).
   - Non-blocking async inference via threadpool execution.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from clinical_transformers import (
    RAW_INPUT_FEATURES,
    FEATURE_DISPLAY_NAMES,
    CLASS_DEFINITIONS,
    CLASS_NAMES,
    BRFSS_FEATURE_BOUNDS,
    PhysiologicalOutlierCapper,
    validate_survey_feature_values
)
from ml_pipeline import apply_decision_rule

# -----------------------------------------------------------------------------
# Logging and Path Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DiabetesAPI")

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
PIPELINE_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

pipeline: Optional[Any] = None
explainer: Optional[Any] = None
metadata: Optional[Dict[str, Any]] = None
metrics_cache: Optional[Dict[str, Any]] = None


def validate_artifact_compatibility(
    loaded_pipeline: Any,
    loaded_explainer: Any = None,
    metadata_dict: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Strict validation of artifact compatibility between model pipeline, TreeSHAP explainer,
    and metadata feature schemas. Prevents serving mismatched or stale artifacts.
    """
    if loaded_pipeline is None:
        raise RuntimeError("Artifact compatibility check failed: pipeline artifact is None.")

    if not hasattr(loaded_pipeline, "named_steps") or "preprocessor" not in loaded_pipeline.named_steps or "classifier" not in loaded_pipeline.named_steps:
        raise RuntimeError("Artifact compatibility check failed: Pipeline must contain 'preprocessor' and 'classifier' named steps.")

    if metadata_dict is not None and "raw_features_canonical_order" in metadata_dict:
        raw_order = metadata_dict.get("raw_features_canonical_order", [])
        if set(raw_order) != set(RAW_INPUT_FEATURES):
            raise RuntimeError(f"Artifact compatibility check failed: Metadata features {set(raw_order)} do not match canonical {set(RAW_INPUT_FEATURES)}.")

    if loaded_explainer is not None:
        if hasattr(loaded_explainer, "feature_names") and loaded_explainer.feature_names is not None:
            if len(loaded_explainer.feature_names) != len(RAW_INPUT_FEATURES):
                raise RuntimeError(
                    f"Artifact compatibility check failed: Explainer feature count ({len(loaded_explainer.feature_names)}) "
                    f"does not match pipeline feature count ({len(RAW_INPUT_FEATURES)})."
                )

    return True


def load_artifacts():
    """Loads and validates trained model pipeline, explainer, metadata, and compatibility."""
    global pipeline, explainer, metadata, metrics_cache
    pipeline = None
    explainer = None
    metadata = None
    metrics_cache = None

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info("Loaded feature metadata.")
        except Exception as e:
            logger.warning("Could not load metadata: %s", e)

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_cache = json.load(f)
            logger.info("Loaded test set metrics.")
        except Exception as e:
            logger.warning("Could not load metrics: %s", e)

    if PIPELINE_PATH.exists():
        try:
            loaded_pipeline = joblib.load(PIPELINE_PATH)
            validate_artifact_compatibility(loaded_pipeline, None, metadata)
            pipeline = loaded_pipeline
            logger.info("Successfully loaded ML pipeline from %s", PIPELINE_PATH)
        except Exception as e:
            logger.error("Failed to load or validate pipeline: %s", e)
            pipeline = None

    if EXPLAINER_PATH.exists():
        try:
            loaded_explainer = joblib.load(EXPLAINER_PATH)
            if pipeline is not None:
                validate_artifact_compatibility(pipeline, loaded_explainer, metadata)
            explainer = loaded_explainer
            logger.info("Successfully loaded SHAP explainer from %s", EXPLAINER_PATH)
        except Exception as e:
            logger.warning("Could not load SHAP explainer: %s", e)
            explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated @app.on_event handlers."""
    logger.info("Starting Diabetes Risk Stratification API...")
    load_artifacts()
    yield
    logger.info("Shutting down Diabetes Risk Stratification API.")


# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CDC BRFSS Diabetes Risk Stratification API",
    description=(
        "Production-grade, explainable machine learning API trained on CDC BRFSS public health survey data. "
        "Evaluates 21 population health indicators with verified TreeSHAP local attribution and configurable decision rules."
    ),
    version="2.2.0-brfss",
    lifespan=lifespan
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Structured validation error handler providing clear, field-level diagnostics
    rather than unformatted internal errors.
    """
    errors = []
    for error in exc.errors():
        loc = error.get("loc", [])
        field = str(loc[-1]) if loc else "general"
        msg = error.get("msg", "Invalid value")
        err_type = error.get("type", "value_error")

        # Clinical boundary clarification for common fields
        if field == "bmi":
            friendly_msg = f"BMI must be between 12.0 and 70.0 kg/m² (received: {error.get('input')})."
        elif field in ["ment_hlth", "phys_hlth"]:
            friendly_msg = f"Health days must be an integer between 0 and 30 days (received: {error.get('input')})."
        elif field in ["high_bp", "high_chol", "smoker", "stroke", "heart_disease_or_attack", "phys_activity", "diff_walk", "sex"]:
            friendly_msg = f"Field '{field}' requires a binary indicator: 1 (Yes) or 0 (No)."
        elif field == "gen_hlth":
            friendly_msg = "General health must be rated 1 (Excellent), 2 (Very Good), 3 (Good), 4 (Fair), or 5 (Poor)."
        elif field == "age":
            friendly_msg = "Age category must be between 1 (18-24 years) and 13 (80+ years)."
        else:
            friendly_msg = msg

        errors.append({
            "field": field,
            "message": friendly_msg,
            "error_type": err_type
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "validation_error",
            "detail": "Input validation failed. Please check the following survey indicators.",
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# -----------------------------------------------------------------------------
# Data Models and Schemas
# -----------------------------------------------------------------------------
class PatientData(BaseModel):
    """
    Patient epidemiological and lifestyle profile conforming to CDC BRFSS indicators.
    """
    # Vitals & Anthropometrics
    bmi: float = Field(..., ge=12.0, le=70.0, description="Body Mass Index in kg/m² (12.0 - 70.0)")
    high_bp: Literal[0, 1] = Field(..., description="High Blood Pressure diagnosed by a doctor (1=Yes, 0=No)")
    high_chol: Literal[0, 1] = Field(..., description="High Blood Cholesterol diagnosed by a doctor (1=Yes, 0=No)")
    chol_check: Literal[0, 1] = Field(default=1, description="Cholesterol checked within past 5 years (1=Yes, 0=No)")

    # Medical History & Chronic Conditions
    stroke: Literal[0, 1] = Field(default=0, description="History of stroke (1=Yes, 0=No)")
    heart_disease_or_attack: Literal[0, 1] = Field(
        default=0, description="History of coronary heart disease or myocardial infarction (1=Yes, 0=No)"
    )
    diff_walk: Literal[0, 1] = Field(
        default=0, description="Serious difficulty walking or climbing stairs (1=Yes, 0=No)"
    )

    # Lifestyle & Behavioral
    smoker: Literal[0, 1] = Field(..., description="Smoked at least 100 cigarettes in entire life (1=Yes, 0=No)")
    phys_activity: Literal[0, 1] = Field(
        ..., description="Any physical activity or exercise in past 30 days other than regular job (1=Yes, 0=No)"
    )
    fruits: Literal[0, 1] = Field(default=1, description="Consumes fruit 1 or more times per day (1=Yes, 0=No)")
    veggies: Literal[0, 1] = Field(default=1, description="Consumes vegetables 1 or more times per day (1=Yes, 0=No)")
    hvy_alcohol_consump: Literal[0, 1] = Field(
        default=0, description="Heavy alcohol consumption: Men >14 drinks/wk, Women >7 drinks/wk (1=Yes, 0=No)"
    )

    # Healthcare Access
    any_healthcare: Literal[0, 1] = Field(default=1, description="Any health insurance or healthcare plan (1=Yes, 0=No)")
    no_doc_bc_cost: Literal[0, 1] = Field(
        default=0, description="Could not see doctor in past 12 months due to cost (1=Yes, 0=No)"
    )

    # Self-Reported Health Metrics
    gen_hlth: int = Field(
        ..., ge=1, le=5, description="Self-rated general health (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)"
    )
    ment_hlth: float = Field(
        default=0.0, ge=0.0, le=30.0, description="Days of poor mental health in past 30 days (0 - 30)"
    )
    phys_hlth: float = Field(
        default=0.0, ge=0.0, le=30.0, description="Days of poor physical health or injury in past 30 days (0 - 30)"
    )

    # Demographics & Socioeconomic Status
    sex: Literal[0, 1] = Field(..., description="Biological sex (1=Male, 0=Female)")
    age: int = Field(
        ..., ge=1, le=13, description="Age category: 1=18-24, 2=25-29, 3=30-34 ... 13=80+"
    )
    education: int = Field(
        default=4, ge=1, le=6, description="Education level: 1=Never/Kindergarten ... 4=High school grad ... 6=College grad"
    )
    income: int = Field(
        default=5, ge=1, le=8, description="Annual household income bracket: 1=<$10k ... 5=<$35k ... 8=>=$75k"
    )

    # Optional decision rule preference
    decision_rule: Optional[Literal["argmax", "threshold_tuned"]] = Field(
        default="argmax",
        description="Decision rule: 'argmax' (max probability) or 'threshold_tuned' (validation-tuned screening thresholds)"
    )


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    patient_value: Any
    shap_value: float
    impact: str
    direction: Literal["supports", "opposes", "neutral"]
    clinical_insight: str


class AdditivityCheck(BaseModel):
    base_value: float
    all_features_shap_sum: float
    reconstructed_output: float
    model_output: float
    absolute_difference: float
    is_additive: bool


class SHAPExplanation(BaseModel):
    status: Literal["available", "unavailable", "failed_verification", "not_applicable"]
    reason: Optional[str] = None
    explained_class: Optional[str] = None
    explained_class_id: Optional[int] = None
    output_space: Optional[str] = None
    base_value: Optional[float] = None
    top_features: List[FeatureContribution] = []
    omitted_features_shap_sum: Optional[float] = None
    additivity_check: Optional[AdditivityCheck] = None


class PredictionResult(BaseModel):
    class_id: int
    risk_class: Literal["No Reported Diabetes", "Pre-diabetic", "Diabetic"]
    confidence: Optional[float] = None
    probability_assigned: Optional[float] = None
    decision_rule_applied: Literal["argmax_max_probability", "validation_tuned_thresholds"] = "argmax_max_probability"
    probabilities: Optional[Dict[str, float]] = None


class ClinicalRecommendation(BaseModel):
    category: Literal["Nutrition & Diet", "Physical Activity", "Clinical Monitoring", "Medical Consultation"]
    title: str
    description: str
    urgency: Literal["Routine", "Moderate", "Immediate"]


class ResponseMetadata(BaseModel):
    prediction_source: Literal["trained_model", "rule_based_demo"]
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    dataset_type: Optional[str] = None
    explanation_status: Literal["available", "unavailable", "failed_verification", "not_applicable"]
    explanation_method: Optional[str] = None
    probability_calibration: str = "uncalibrated_ensemble_vote"
    imputed_features: List[str] = []
    disclaimer: str


class PredictionResponse(BaseModel):
    status: str
    metadata: ResponseMetadata
    prediction: PredictionResult
    shap_explanation: SHAPExplanation
    clinical_recommendations: List[ClinicalRecommendation]
    timestamp: str


def generate_feature_clinical_insight(feature: str, value: Any, shap_val: float, explained_class: str) -> str:
    """Generates class-relative non-causal statistical insight for CDC BRFSS indicators."""
    supports = shap_val > 0
    verb = "supports" if supports else "opposes"

    if feature == "high_bp":
        status_txt = "presence of diagnosed hypertension" if value == 1 else "no reported hypertension"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "high_chol":
        status_txt = "diagnosed hypercholesterolemia" if value == 1 else "normal cholesterol history"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "bmi":
        return f"BMI of {value} kg/m²; statistically {verb} the '{explained_class}' prediction."
    elif feature == "gen_hlth":
        ratings = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
        r_txt = ratings.get(int(value), str(value))
        return f"General health self-rated as {r_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "age":
        age_ranges = {
            1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
            6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
            11: "70-74", 12: "75-79", 13: "80+"
        }
        range_txt = age_ranges.get(int(value), f"bracket {value}")
        return f"Individual in age bracket {range_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "phys_activity":
        status_txt = "active in past 30 days" if value == 1 else "sedentary lifestyle (no past 30 days activity)"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "diff_walk":
        status_txt = "mobility limitation (difficulty walking/stairs)" if value == 1 else "no mobility difficulty"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "smoker":
        status_txt = "history of smoking (>=100 cigarettes)" if value == 1 else "never smoked >=100 cigarettes"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    else:
        disp_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        return f"{disp_name} value of {value}; statistically {verb} the '{explained_class}' prediction."


def generate_clinical_recommendations(risk_class: str, patient: PatientData) -> List[ClinicalRecommendation]:
    """Generates tailored educational care pathways based on CDC National Diabetes Prevention Program guidelines."""
    recs = []

    if risk_class == "Diabetic":
        recs.append(ClinicalRecommendation(
            category="Medical Consultation",
            title="Formal Diagnostic Evaluation & Glycemic Confirmation",
            description="Consult a healthcare clinician for diagnostic venous laboratory testing (fasting plasma glucose >=126 mg/dL or venous HbA1c >=6.5%).",
            urgency="Immediate"
        ))
        if patient.high_bp == 1 or patient.high_chol == 1:
            recs.append(ClinicalRecommendation(
                category="Clinical Monitoring",
                title="Concomitant Cardiovascular Risk Factor Surveillance",
                description="Manage concomitant hypertension and hyperlipidemia with target blood pressure and lipid monitoring.",
                urgency="Immediate"
            ))

    elif risk_class == "Pre-diabetic":
        recs.append(ClinicalRecommendation(
            category="Physical Activity",
            title="CDC National Diabetes Prevention Program (DPP) Exercise Target",
            description="Aim for at least 150 minutes per week of moderate-intensity aerobic physical activity (e.g. brisk walking) combined with resistance training.",
            urgency="Moderate"
        ))
        recs.append(ClinicalRecommendation(
            category="Nutrition & Diet",
            title="Dietary Modification & 5-7% Weight Reduction Goal",
            description="Follow Mediterranean or DASH dietary patterns rich in dietary fiber, reducing refined carbohydrates and sugar-sweetened beverages.",
            urgency="Moderate"
        ))

    else:
        recs.append(ClinicalRecommendation(
            category="Physical Activity",
            title="Sustained Physical Health & Active Living",
            description="Maintain at least 150 minutes of weekly aerobic exercise and daily physical activity to protect metabolic insulin sensitivity.",
            urgency="Routine"
        ))
        recs.append(ClinicalRecommendation(
            category="Clinical Monitoring",
            title="Periodic Preventive Health Checkups",
            description="Continue routine preventive health checkups every 1 to 2 years.",
            urgency="Routine"
        ))

    return recs


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint exposing readiness and loaded artifact metadata."""
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "service": "CDC BRFSS Diabetes Risk Stratification API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_ready": pipeline is not None,
        "explainer_ready": explainer is not None,
        "artifacts_loaded": {
            "pipeline": pipeline is not None,
            "explainer": explainer is not None,
            "metadata": metadata is not None,
            "metrics": metrics_cache is not None
        },
        "model_info": {
            "version": metadata.get("model_version", "unknown") if metadata else "unknown",
            "model_type": type(pipeline.named_steps["classifier"]).__name__ if pipeline else "none",
            "dataset_type": "real_cdc_brfss_2015"
        },
        "supported_classes": CLASS_NAMES
    }


@app.get("/metrics")
async def get_metrics():
    """Returns test set evaluation metrics and comparative baselines."""
    if metrics_cache is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation metrics artifact (artifacts/model_metrics.json) is not loaded."
        )
    return metrics_cache


@app.get("/experiments")
async def list_experiments():
    """Returns master experiment manifest and registered empirical investigation files."""
    manifest_path = EXPERIMENTS_DIR / "experiment_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Experiment manifest not generated yet.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/experiments/{experiment_name}")
async def get_experiment_details(experiment_name: str):
    """Retrieves specific experiment results JSON (learning_curves, preprocessing_ablation, etc.)."""
    safe_name = f"{experiment_name}.json" if not experiment_name.endswith(".json") else experiment_name
    exp_file = EXPERIMENTS_DIR / safe_name
    if not exp_file.exists():
        raise HTTPException(status_code=404, detail=f"Experiment artifact '{safe_name}' not found.")
    with open(exp_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/sample-patients")
async def get_sample_patients():
    """Returns curated representative survey profiles for exploration."""
    return {
        "healthy_adult": {
            "name": "Healthy Young Adult (Low Risk Profile)",
            "description": "Young active individual with normal BMI, healthy diet, and no chronic diagnoses.",
            "data": {
                "high_bp": 0, "high_chol": 0, "chol_check": 1, "bmi": 22.4, "smoker": 0,
                "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1, "fruits": 1,
                "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
                "gen_hlth": 1, "ment_hlth": 0.0, "phys_hlth": 0.0, "diff_walk": 0,
                "sex": 0, "age": 3, "education": 6, "income": 8
            }
        },
        "prediabetic_risk": {
            "name": "Sedentary Middle-Aged Adult (Pre-diabetic Lifestyle Risk)",
            "description": "Middle-aged respondent with elevated BMI, sedentary habits, and cost barrier to care.",
            "data": {
                "high_bp": 1, "high_chol": 0, "chol_check": 1, "bmi": 28.5, "smoker": 0,
                "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 0, "fruits": 0,
                "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 1,
                "gen_hlth": 3, "ment_hlth": 4.0, "phys_hlth": 2.0, "diff_walk": 0,
                "sex": 1, "age": 7, "education": 4, "income": 5
            }
        },
        "diabetic_multimorbid": {
            "name": "Older Adult with Comorbidities (High Risk Diabetic)",
            "description": "Older respondent with diagnosed hypertension, hyperlipidemia, obesity, and mobility limits.",
            "data": {
                "high_bp": 1, "high_chol": 1, "chol_check": 1, "bmi": 34.2, "smoker": 1,
                "stroke": 0, "heart_disease_or_attack": 1, "phys_activity": 0, "fruits": 0,
                "veggies": 0, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
                "gen_hlth": 5, "ment_hlth": 10.0, "phys_hlth": 20.0, "diff_walk": 1,
                "sex": 1, "age": 10, "education": 3, "income": 3
            }
        }
    }


def _run_inference(patient_data: PatientData) -> PredictionResponse:
    """Synchronous inference helper executed inside run_in_threadpool."""
    raw_dict = patient_data.model_dump()
    req_rule = raw_dict.pop("decision_rule", "argmax") or "argmax"
    input_df = pd.DataFrame([raw_dict])[RAW_INPUT_FEATURES]

    classifier = pipeline.named_steps["classifier"]
    classes_ = list(classifier.classes_)

    probs = pipeline.predict_proba(input_df)[0]

    # Map probability dictionary
    prob_dict = {
        CLASS_DEFINITIONS[int(cls_id)]: float(np.round(probs[c_idx], 4))
        for c_idx, cls_id in enumerate(classes_)
    }

    # Evaluate decision rule
    pred_classes, assigned_probs = apply_decision_rule(
        probs,
        classes=classes_,
        rule_type=req_rule,
        threshold_diabetic=0.52,
        threshold_prediabetic=0.35
    )
    predicted_class_id = int(pred_classes[0])
    assigned_prob = float(assigned_probs[0])
    confidence = assigned_prob  # Never display maximum probability for a different class!
    rule_applied = "validation_tuned_thresholds" if req_rule == "threshold_tuned" else "argmax_max_probability"

    risk_name = CLASS_DEFINITIONS[predicted_class_id]

    # SHAP local attribution using exact transformed feature order
    top_features: List[FeatureContribution] = []
    shap_status = "unavailable"
    shap_reason = None
    base_value = None
    omitted_sum = None
    additivity_check = None

    if explainer is not None:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            X_trans = preprocessor.transform(input_df)

            # Transformed feature names order from ColumnTransformer
            transformed_feature_names = [col.split("__")[-1] for col in preprocessor.get_feature_names_out()]

            shap_raw = explainer.shap_values(X_trans)
            if isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
                shap_cls = shap_raw[0, :, col_idx]
            elif isinstance(shap_raw, list):
                shap_cls = np.array(shap_raw[col_idx])[0]
            else:
                shap_cls = np.array(shap_raw)[0]

            if hasattr(explainer, "expected_value") and explainer.expected_value is not None:
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base_value = float(ev[col_idx])
                else:
                    base_value = float(ev)
            else:
                raise ValueError("SHAP explainer does not contain an expected_value.")

            feature_shaps = []
            for idx, feat_name in enumerate(transformed_feature_names):
                s_val = float(shap_cls[idx])
                p_val = raw_dict[feat_name]
                disp = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
                direction = "supports" if s_val > 0 else ("opposes" if s_val < 0 else "neutral")
                prefix = "Supports" if s_val > 0 else "Opposes"
                pp_val = s_val * 100.0
                impact_str = f"{prefix} {risk_name} prediction ({pp_val:+.1f} percentage points)"
                insight = generate_feature_clinical_insight(feat_name, p_val, s_val, risk_name)

                feature_shaps.append(FeatureContribution(
                    feature=feat_name,
                    display_name=disp,
                    patient_value=p_val,
                    shap_value=float(np.round(s_val, 4)),
                    impact=impact_str,
                    direction=direction,
                    clinical_insight=insight
                ))

            feature_shaps.sort(key=lambda f: abs(f.shap_value), reverse=True)
            top_features = feature_shaps[:8]

            all_shap_sum = float(np.sum(shap_cls))
            top_shap_sum = float(sum(f.shap_value for f in top_features))
            omitted_sum = float(np.round(all_shap_sum - top_shap_sum, 4))
            reconstructed = float(base_value + all_shap_sum)
            model_out = float(probs[col_idx])
            abs_diff = float(abs(reconstructed - model_out))

            is_additive = (abs_diff < 0.05)
            additivity_check = AdditivityCheck(
                base_value=float(np.round(base_value, 4)),
                all_features_shap_sum=float(np.round(all_shap_sum, 4)),
                reconstructed_output=float(np.round(reconstructed, 4)),
                model_output=float(np.round(model_out, 4)),
                absolute_difference=float(np.round(abs_diff, 4)),
                is_additive=is_additive
            )

            if is_additive:
                shap_status = "available"
            else:
                shap_status = "failed_verification"
                shap_reason = f"Additivity difference ({abs_diff:.4f}) exceeds tolerance (0.05)."

        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            shap_status = "unavailable"
            shap_reason = str(e)

    recs = generate_clinical_recommendations(risk_name, patient_data)

    return PredictionResponse(
        status="success",
        metadata=ResponseMetadata(
            prediction_source="trained_model",
            model_version=metadata.get("model_version", "2.2.0-brfss") if metadata else "2.2.0-brfss",
            model_type=type(pipeline.named_steps["classifier"]).__name__,
            dataset_type="real_cdc_brfss_2015",
            explanation_status=shap_status,
            explanation_method="TreeSHAP (Holdout-free background)" if shap_status == "available" else None,
            probability_calibration="uncalibrated_ensemble_vote",
            imputed_features=[],
            disclaimer="Trained on CDC BRFSS public health surveillance data. For educational and research demonstration."
        ),
        prediction=PredictionResult(
            class_id=predicted_class_id,
            risk_class=risk_name,
            confidence=float(np.round(confidence, 4)),
            probability_assigned=float(np.round(assigned_prob, 4)),
            decision_rule_applied=rule_applied,
            probabilities=prob_dict
        ),
        shap_explanation=SHAPExplanation(
            status=shap_status,
            reason=shap_reason,
            explained_class=risk_name,
            explained_class_id=predicted_class_id,
            output_space="probability",
            base_value=float(np.round(base_value, 4)) if base_value is not None else None,
            top_features=top_features,
            omitted_features_shap_sum=omitted_sum,
            additivity_check=additivity_check
        ),
        clinical_recommendations=recs,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes_risk(patient: PatientData):
    """
    Evaluates trained CDC BRFSS machine learning pipeline and returns class prediction,
    assigned class probability, applied decision rule, and verified TreeSHAP feature attributions.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model pipeline is not loaded or artifact is missing (artifacts/model_pipeline.joblib). Run ml_pipeline.py."
        )

    try:
        return await run_in_threadpool(_run_inference, patient)
    except Exception as e:
        logger.error("Inference execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/demo-predict", response_model=PredictionResponse)
async def demo_predict_diabetes_risk(patient: PatientData):
    """
    Explicit Opt-In Rule-Based Demo Endpoint.
    Evaluates project-defined epidemiological heuristics rather than the ML model.
    Omits confidence, probabilities, and TreeSHAP attributions.
    """
    score = 0
    if patient.high_bp == 1: score += 2
    if patient.high_chol == 1: score += 2
    if patient.bmi >= 30.0: score += 3
    elif patient.bmi >= 25.0: score += 1
    if patient.age >= 9: score += 2
    elif patient.age >= 6: score += 1
    if patient.gen_hlth >= 4: score += 2
    if patient.phys_activity == 0: score += 1

    if score >= 6:
        risk_name = "Diabetic"
        predicted_class_id = 2
    elif score >= 3:
        risk_name = "Pre-diabetic"
        predicted_class_id = 1
    else:
        risk_name = "No Reported Diabetes"
        predicted_class_id = 0

    recs = generate_clinical_recommendations(risk_name, patient)

    return PredictionResponse(
        status="demo",
        metadata=ResponseMetadata(
            prediction_source="rule_based_demo",
            model_version="heuristic-demo",
            model_type="EpidemiologicalRuleHeuristic",
            dataset_type="heuristic_rules",
            explanation_status="unavailable",
            explanation_method=None,
            probability_calibration="none",
            imputed_features=[],
            disclaimer="Rule-based illustrative demo mode. Not an ML model prediction."
        ),
        prediction=PredictionResult(
            class_id=predicted_class_id,
            risk_class=risk_name,
            confidence=None,
            probability_assigned=None,
            decision_rule_applied="argmax_max_probability",
            probabilities=None
        ),
        shap_explanation=SHAPExplanation(
            status="unavailable",
            reason="SHAP feature attribution is omitted in rule-based demo mode.",
            explained_class=risk_name,
            explained_class_id=predicted_class_id,
            output_space=None,
            base_value=None,
            top_features=[],
            omitted_features_shap_sum=None,
            additivity_check=None
        ),
        clinical_recommendations=recs,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
