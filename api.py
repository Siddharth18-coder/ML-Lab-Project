"""
api.py - CDC BRFSS Diabetes Risk Stratification & Explainability REST API
==========================================================================
Data Provenance & Scope:
Delivers multi-class diabetes risk classification (No Reported Diabetes, Pre-diabetic, Diabetic)
trained directly on the real-world U.S. Centers for Disease Control and Prevention (CDC)
Behavioral Risk Factor Surveillance System (BRFSS) dataset (50,000 surveyed cohort).

Key Architectural Guarantees:
1. Strict Pydantic v2 validation adhering to CDC BRFSS health indicator codes and bounds.
2. Verified TreeSHAP explainability:
   - Features mapped using exact transformed order from ColumnTransformer.get_feature_names_out().
   - Classes mapped dynamically using classifier.classes_.
   - Base value dynamically extracted from explainer for the predicted class (no fabricated fallbacks).
   - Attributions labeled relative to the explained class ('Supports X' vs 'Opposes X').
   - Additivity check: base_value + sum(shap_values) == model probability output (tol=0.05).
   - Associations framed scientifically without making unsupported causal medical claims.
3. Explicit Demo Mode:
   - POST /predict: strictly evaluates trained real-world ML model pipeline.
   - POST /demo-predict: rule-based epidemiological demo returning confidence=None, probabilities=None.
4. Truthful operational status:
   - If model artifacts are missing or corrupt, returns HTTP 503.
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
    PhysiologicalOutlierCapper
)

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
PIPELINE_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

pipeline: Optional[Any] = None
explainer: Optional[Any] = None
metadata: Optional[Dict[str, Any]] = None
metrics_cache: Optional[Dict[str, Any]] = None


def load_artifacts():
    """Loads and validates trained model pipeline, explainer, and metadata."""
    global pipeline, explainer, metadata, metrics_cache
    pipeline = None
    explainer = None
    metadata = None
    metrics_cache = None

    if PIPELINE_PATH.exists():
        try:
            loaded_pipeline = joblib.load(PIPELINE_PATH)
            # Compatibility sanity check
            if hasattr(loaded_pipeline, "named_steps") and "preprocessor" in loaded_pipeline.named_steps:
                pipeline = loaded_pipeline
                logger.info("Successfully loaded ML pipeline from %s", PIPELINE_PATH)
            else:
                logger.error("Loaded artifact at %s is not a valid scikit-learn pipeline.", PIPELINE_PATH)
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)

    if EXPLAINER_PATH.exists():
        try:
            loaded_explainer = joblib.load(EXPLAINER_PATH)
            explainer = loaded_explainer
            logger.info("Successfully loaded SHAP explainer from %s", EXPLAINER_PATH)
        except Exception as e:
            logger.warning("Could not load SHAP explainer: %s", e)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="CDC BRFSS Diabetes Risk Stratification API",
    description=(
        "Public health decision support service for multi-class diabetes risk "
        "screening and SHAP explainability trained on real CDC BRFSS epidemiological data."
    ),
    version="2.1.0",
    lifespan=lifespan
)

# CORS configuration supporting environment variable or local development
allowed_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]
if "*" in allowed_origins or not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Provides structured, human-readable field-specific validation errors."""
    error_list = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        error_list.append({
            "field": field_path,
            "message": err.get("msg", "Validation failed"),
            "type": err.get("type", "value_error")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "validation_error",
            "message": "Input validation failed. Please check the supplied survey indicators.",
            "errors": error_list
        }
    )


# -----------------------------------------------------------------------------
# Pydantic Schemas: CDC BRFSS 21 Health Indicators
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
    status: Literal["available", "unavailable"]
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
    explanation_status: Literal["available", "unavailable", "not_applicable"]
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
    elif feature == "smoker":
        status_txt = "history of smoking (>=100 cigarettes)" if value == 1 else "non-smoker"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "diff_walk":
        status_txt = "mobility limitation (difficulty walking/stairs)" if value == 1 else "no walking limitations"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "heart_disease_or_attack":
        status_txt = "cardiovascular comorbidity (CHD/MI)" if value == 1 else "no cardiovascular comorbidity"
        return f"Reported {status_txt}; statistically {verb} the '{explained_class}' prediction."
    elif feature == "stroke":
        status_txt = "history of stroke" if value == 1 else "no stroke history"
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
    """Health check endpoint exposing readiness and artifact status."""
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
    input_df = pd.DataFrame([raw_dict])[RAW_INPUT_FEATURES]

    classifier = pipeline.named_steps["classifier"]
    classes_ = list(classifier.classes_)

    probs = pipeline.predict_proba(input_df)[0]
    col_idx = int(np.argmax(probs))
    predicted_class_id = int(classes_[col_idx])
    confidence = float(np.max(probs))
    risk_name = CLASS_DEFINITIONS[predicted_class_id]

    prob_dict = {
        CLASS_DEFINITIONS[int(cls_id)]: float(np.round(probs[c_idx], 4))
        for c_idx, cls_id in enumerate(classes_)
    }

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
                # Shape: (n_samples, n_features, n_classes)
                shap_cls = shap_raw[0, :, col_idx]
            elif isinstance(shap_raw, list):
                # List of arrays per class: shap_raw[col_idx] has shape (n_samples, n_features)
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
                # Percentage point description
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

            # Sort by absolute contribution magnitude
            feature_shaps.sort(key=lambda f: abs(f.shap_value), reverse=True)
            top_features = feature_shaps[:8]

            all_shap_sum = float(np.sum(shap_cls))
            top_shap_sum = float(sum(f.shap_value for f in top_features))
            omitted_sum = float(np.round(all_shap_sum - top_shap_sum, 4))
            reconstructed = float(base_value + all_shap_sum)
            model_out = float(probs[col_idx])
            abs_diff = float(abs(reconstructed - model_out))

            additivity_check = AdditivityCheck(
                base_value=float(np.round(base_value, 4)),
                all_features_shap_sum=float(np.round(all_shap_sum, 4)),
                reconstructed_output=float(np.round(reconstructed, 4)),
                model_output=float(np.round(model_out, 4)),
                absolute_difference=float(np.round(abs_diff, 4)),
                is_additive=(abs_diff < 0.05)
            )
            shap_status = "available"

        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            shap_status = "unavailable"
            shap_reason = str(e)

    recs = generate_clinical_recommendations(risk_name, patient_data)

    return PredictionResponse(
        status="success",
        metadata=ResponseMetadata(
            prediction_source="trained_model",
            model_version=metadata.get("model_version", "2.1.0-brfss") if metadata else "2.1.0-brfss",
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
            probabilities=prob_dict
        ),
        shap_explanation=SHAPExplanation(
            status=shap_status,
            reason=shap_reason,
            explained_class=risk_name if shap_status == "available" else None,
            explained_class_id=predicted_class_id if shap_status == "available" else None,
            output_space="probability" if shap_status == "available" else None,
            base_value=float(np.round(base_value, 4)) if base_value is not None else None,
            top_features=top_features,
            omitted_features_shap_sum=omitted_sum,
            additivity_check=additivity_check
        ),
        clinical_recommendations=recs,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(patient: PatientData):
    """
    Main inference endpoint evaluating the trained Scikit-Learn pipeline and TreeSHAP.
    Runs non-blocking in threadpool.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained ML pipeline artifact (artifacts/model_pipeline.joblib) is not loaded."
        )

    try:
        response = await run_in_threadpool(_run_inference, patient)
        return response
    except Exception as e:
        logger.error("Inference pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/demo-predict", response_model=PredictionResponse)
async def predict_demo_rules(patient: PatientData):
    """
    Explicit Rule-based epidemiological demonstration.
    Returns confidence=None and probabilities=None with explicit demo labeling.
    """
    # Project-defined Epidemiological Threshold Heuristic
    if patient.high_bp == 1 and patient.high_chol == 1 and patient.bmi >= 30.0 and patient.age >= 9:
        cls_id = 2
        risk_name = "Diabetic"
    elif patient.high_bp == 1 and patient.high_chol == 1 and patient.bmi >= 27.0:
        cls_id = 1
        risk_name = "Pre-diabetic"
    else:
        cls_id = 0
        risk_name = "No Reported Diabetes"

    recs = generate_clinical_recommendations(risk_name, patient)

    return PredictionResponse(
        status="demo",
        metadata=ResponseMetadata(
            prediction_source="rule_based_demo",
            model_version=None,
            model_type="Project-Defined Epidemiological Heuristic",
            dataset_type="none",
            explanation_status="not_applicable",
            explanation_method=None,
            probability_calibration="none",
            imputed_features=[],
            disclaimer="Explicit rule-based demonstration mode. Not a trained model prediction."
        ),
        prediction=PredictionResult(
            class_id=cls_id,
            risk_class=risk_name,
            confidence=None,
            probabilities=None
        ),
        shap_explanation=SHAPExplanation(
            status="unavailable",
            reason="SHAP explanations are not applicable to the rule-based demonstration mode."
        ),
        clinical_recommendations=recs,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
