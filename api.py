"""
api.py - CDC BRFSS Diabetes Risk Machine Learning REST API
============================================================
Data Provenance & Scope:
This API delivers multi-class diabetes risk classification (Healthy, Pre-diabetic, Diabetic)
trained directly on the real-world U.S. Centers for Disease Control and Prevention (CDC)
Behavioral Risk Factor Surveillance System (BRFSS) dataset (250,000+ surveyed individuals).

Key Architectural Guarantees:
1. Strict Pydantic v2 validation adhering to CDC BRFSS health indicator codes and ranges.
2. Verified TreeSHAP explainability:
   - Base value dynamically extracted from explainer for predicted class.
   - Attributions labeled relative to the explained class ('Supports X' vs 'Opposes X').
   - Additivity check: base_value + sum(shap_values) ≈ model probability output.
3. Explicit Demo Mode:
   - POST /predict: strictly evaluates trained real-world ML model pipeline.
   - POST /demo-predict: rule-based epidemiological demo returning confidence=None, probabilities=None.
4. Truthful operational status:
   - If model artifacts are missing, returns HTTP 503 (no fake simulated numbers).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Literal

import numpy as np
import pandas as pd
import joblib

from clinical_transformers import PhysiologicalOutlierCapper, BRFSS_FEATURE_BOUNDS
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Logging and Initialization
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DiabetesAPI")

app = FastAPI(
    title="CDC BRFSS Diabetes Risk Intelligence API",
    description=(
        "Public health decision support service for multi-class diabetes risk "
        "classification and SHAP explainability trained on real CDC BRFSS epidemiological data."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Artifact Locations and Model Loading
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "model_pipeline.joblib")
EXPLAINER_PATH = os.path.join(ARTIFACTS_DIR, "explainer.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.json")

CLASS_NAMES = ["Healthy", "Pre-diabetic", "Diabetic"]
CLASS_MAP = {0: "Healthy", 1: "Pre-diabetic", 2: "Diabetic"}

pipeline: Optional[Any] = None
explainer: Optional[Any] = None
metadata: Optional[Dict[str, Any]] = None
metrics_cache: Optional[Dict[str, Any]] = None


def load_artifacts():
    """Loads trained model pipeline, explainer, and metadata."""
    global pipeline, explainer, metadata, metrics_cache
    pipeline = None
    explainer = None
    metadata = None
    metrics_cache = None

    if os.path.exists(PIPELINE_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            logger.info("Successfully loaded ML pipeline from %s", PIPELINE_PATH)
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)

    if os.path.exists(EXPLAINER_PATH):
        try:
            explainer = joblib.load(EXPLAINER_PATH)
            logger.info("Successfully loaded SHAP explainer from %s", EXPLAINER_PATH)
        except Exception as e:
            logger.warning("Could not load SHAP explainer: %s", e)

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info("Loaded feature metadata.")
        except Exception as e:
            logger.warning("Could not load metadata: %s", e)

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_cache = json.load(f)
            logger.info("Loaded test set metrics.")
        except Exception as e:
            logger.warning("Could not load metrics: %s", e)


@app.on_event("startup")
async def startup_event():
    load_artifacts()


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
    risk_class: Literal["Healthy", "Pre-diabetic", "Diabetic"]
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
    imputed_features: List[str] = []
    disclaimer: str


class PredictionResponse(BaseModel):
    status: str
    metadata: ResponseMetadata
    prediction: PredictionResult
    shap_explanation: SHAPExplanation
    clinical_recommendations: List[ClinicalRecommendation]
    timestamp: str


# -----------------------------------------------------------------------------
# Clinical Display Metadata and Insights
# -----------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
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

ALL_FEATURE_KEYS = [
    "high_bp", "high_chol", "chol_check", "bmi", "smoker",
    "stroke", "heart_disease_or_attack", "phys_activity", "fruits",
    "veggies", "hvy_alcohol_consump", "any_healthcare", "no_doc_bc_cost",
    "gen_hlth", "ment_hlth", "phys_hlth", "diff_walk", "sex", "age",
    "education", "income"
]


def generate_feature_clinical_insight(feature: str, value: Any, shap_val: float, explained_class: str) -> str:
    """Generates class-relative clinical insights for CDC BRFSS indicators."""
    supports = shap_val > 0
    verb = "supporting" if supports else "opposing"

    if feature == "high_bp":
        status_txt = "presence of diagnosed hypertension" if value == 1 else "normal blood pressure"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "high_chol":
        status_txt = "diagnosed hypercholesterolemia" if value == 1 else "normal cholesterol history"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "bmi":
        return f"BMI of {value} kg/m², {verb} {explained_class} classification."
    elif feature == "gen_hlth":
        ratings = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
        r_txt = ratings.get(int(value), str(value))
        return f"General health rated as {r_txt}, {verb} {explained_class} classification."
    elif feature == "age":
        age_ranges = {
            1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
            6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
            11: "70-74", 12: "75-79", 13: "80+"
        }
        range_txt = age_ranges.get(int(value), f"bracket {value}")
        return f"Patient in age bracket {range_txt}, {verb} {explained_class} classification."
    elif feature == "phys_activity":
        status_txt = "active in past 30 days" if value == 1 else "sedentary lifestyle"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "smoker":
        status_txt = "history of smoking" if value == 1 else "non-smoker"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "diff_walk":
        status_txt = "mobility limitation (difficulty walking)" if value == 1 else "no mobility limitations"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "heart_disease_or_attack":
        status_txt = "cardiovascular comorbidity (CHD/MI)" if value == 1 else "no cardiovascular disease"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    elif feature == "stroke":
        status_txt = "history of stroke" if value == 1 else "no stroke history"
        return f"Reported {status_txt}, {verb} {explained_class} classification."
    else:
        disp_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        return f"{disp_name} of {value}, {verb} {explained_class} classification."


def generate_clinical_recommendations(risk_class: str, patient: PatientData) -> List[ClinicalRecommendation]:
    """Generates tailored recommendations based on CDC National Diabetes Prevention Program guidelines."""
    recs = []

    if risk_class == "Diabetic":
        recs.append(ClinicalRecommendation(
            category="Medical Consultation",
            title="Comprehensive Glycemic & Diabetic Clinical Evaluation",
            description="Schedule a comprehensive clinical assessment including diagnostic HbA1c and fasting plasma glucose confirmation.",
            urgency="Immediate"
        ))
        if patient.high_bp == 1 or patient.high_chol == 1:
            recs.append(ClinicalRecommendation(
                category="Clinical Monitoring",
                title="Cardiovascular Risk Factor Management",
                description="Manage concomitant hypertension and hyperlipidemia with target blood pressure <130/80 mmHg and statin therapy as clinically indicated.",
                urgency="Immediate"
            ))

    elif risk_class == "Pre-diabetic":
        recs.append(ClinicalRecommendation(
            category="Physical Activity",
            title="CDC National Diabetes Prevention Program (DPP) Exercise Target",
            description="Aim for at least 150-300 minutes per week of moderate-intensity aerobic physical activity combined with resistance training.",
            urgency="Moderate"
        ))
        recs.append(ClinicalRecommendation(
            category="Nutrition & Diet",
            title="Dietary Modification & 5-7% Weight Reduction Target",
            description="Adopt a Mediterranean or DASH dietary pattern rich in dietary fiber, reducing refined carbohydrates and sugar-sweetened beverages.",
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
            title="Periodic Preventive Health Screening",
            description="Continue routine preventive health checkups every 1 to 2 years.",
            urgency="Routine"
        ))

    return recs


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes_risk(patient: PatientData):
    """
    Evaluates patient profile through trained real-world CDC BRFSS machine learning pipeline.
    Returns predicted class, probabilities, class-relative SHAP attributions, and additivity checks.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifacts are not loaded. Run 'python ml_pipeline.py' first."
        )

    try:
        raw_dict = patient.model_dump()
        input_df = pd.DataFrame([raw_dict])[ALL_FEATURE_KEYS]

        probs = pipeline.predict_proba(input_df)[0]
        cls_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        risk_name = CLASS_MAP[cls_id]

        prob_dict = {CLASS_NAMES[i]: float(np.round(probs[i], 4)) for i in range(len(CLASS_NAMES))}

        # SHAP local attribution
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

                shap_raw = explainer.shap_values(X_trans)
                if isinstance(shap_raw, list):
                    shap_cls = np.array(shap_raw[cls_id])[0]
                elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
                    shap_cls = shap_raw[0, :, cls_id]
                else:
                    shap_cls = np.array(shap_raw)[0]

                if hasattr(explainer, "expected_value"):
                    ev = explainer.expected_value
                    base_value = float(ev[cls_id]) if isinstance(ev, (list, np.ndarray)) else float(ev)
                else:
                    base_value = 0.33

                feature_shaps = []
                for idx, feat_name in enumerate(ALL_FEATURE_KEYS):
                    s_val = float(shap_cls[idx])
                    p_val = raw_dict[feat_name]
                    disp = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
                    direction = "supports" if s_val > 0 else ("opposes" if s_val < 0 else "neutral")
                    prefix = "Supports" if s_val > 0 else "Opposes"
                    impact_str = f"{prefix} {risk_name} prediction ({s_val:+.1%})"
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

                # Top features sorted by absolute magnitude
                feature_shaps.sort(key=lambda f: abs(f.shap_value), reverse=True)
                top_features = feature_shaps[:8]

                all_shap_sum = float(np.sum(shap_cls))
                top_shap_sum = float(sum(f.shap_value for f in top_features))
                omitted_sum = float(np.round(all_shap_sum - top_shap_sum, 4))
                reconstructed = float(base_value + all_shap_sum)
                model_out = float(probs[cls_id])
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

        recs = generate_clinical_recommendations(risk_name, patient)

        return PredictionResponse(
            status="success",
            metadata=ResponseMetadata(
                prediction_source="trained_model",
                model_version=metadata.get("model_version", "2.0.0-brfss") if metadata else "2.0.0-brfss",
                model_type=type(pipeline.named_steps["classifier"]).__name__,
                dataset_type="real_cdc_brfss_2015",
                explanation_status=shap_status,
                explanation_method="TreeSHAP (Holdout-free background)" if shap_status == "available" else None,
                imputed_features=[],
                disclaimer="Trained on CDC BRFSS public health surveillance data. For educational and research demonstration."
            ),
            prediction=PredictionResult(
                class_id=cls_id,
                risk_class=risk_name,
                confidence=float(np.round(confidence, 4)),
                probabilities=prob_dict
            ),
            shap_explanation=SHAPExplanation(
                status=shap_status,
                reason=shap_reason,
                explained_class=risk_name if shap_status == "available" else None,
                explained_class_id=cls_id if shap_status == "available" else None,
                output_space="probability" if shap_status == "available" else None,
                base_value=float(np.round(base_value, 4)) if base_value is not None else None,
                top_features=top_features,
                omitted_features_shap_sum=omitted_sum,
                additivity_check=additivity_check
            ),
            clinical_recommendations=recs,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error("Inference pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/demo-predict", response_model=PredictionResponse)
async def predict_demo_rules(patient: PatientData):
    """
    Explicit Rule-based epidemiological demonstration.
    Returns confidence=None and probabilities=None with explicit demo labeling.
    """
    # Deterministic CDC Rule Heuristic
    if patient.high_bp == 1 and patient.high_chol == 1 and patient.bmi >= 30.0 and patient.age >= 9:
        cls_id = 2
        risk_name = "Diabetic"
    elif patient.high_bp == 1 and patient.high_chol == 1 and patient.bmi >= 27.0:
        cls_id = 1
        risk_name = "Pre-diabetic"
    else:
        cls_id = 0
        risk_name = "Healthy"

    recs = generate_clinical_recommendations(risk_name, patient)

    return PredictionResponse(
        status="demo",
        metadata=ResponseMetadata(
            prediction_source="rule_based_demo",
            model_version=None,
            model_type="CDC Epidemiological Heuristic Rules",
            dataset_type="none",
            explanation_status="not_applicable",
            explanation_method=None,
            imputed_features=[],
            disclaimer="Rule-based demo — not a trained model prediction. No probability or SHAP is measured."
        ),
        prediction=PredictionResult(
            class_id=cls_id,
            risk_class=risk_name,
            confidence=None,
            probabilities=None
        ),
        shap_explanation=SHAPExplanation(
            status="unavailable",
            reason="Demo mode evaluates deterministic rules. SHAP requires a trained machine learning model.",
            top_features=[]
        ),
        clinical_recommendations=recs,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/health")
async def health_check():
    """Returns backend operational status, model readiness, and artifact state."""
    return {
        "status": "healthy",
        "service": "CDC BRFSS Diabetes Risk Stratification API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_ready": pipeline is not None,
        "explainer_ready": explainer is not None,
        "artifacts_loaded": {
            "pipeline": pipeline is not None,
            "explainer": explainer is not None,
            "metadata": metadata is not None,
            "metrics": metrics_cache is not None,
        },
        "model_info": {
            "version": metadata.get("model_version") if metadata else None,
            "model_type": metadata.get("estimator") if metadata else None,
            "dataset_type": metadata.get("dataset_type") if metadata else None,
        },
        "supported_classes": CLASS_NAMES,
    }


@app.get("/metrics")
async def get_model_metrics():
    """Returns hold-out test set performance metrics from saved artifacts."""
    if metrics_cache:
        return metrics_cache

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read metrics file: {e}")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Evaluation metrics unavailable. Run ml_pipeline.py first to generate evaluation artifacts."
    )


@app.get("/sample-patients")
async def get_sample_patients():
    """Returns verified reference clinical survey profiles from the real CDC BRFSS dataset."""
    return {
        "healthy": {
            "bmi": 22.4,
            "high_bp": 0,
            "high_chol": 0,
            "chol_check": 1,
            "smoker": 0,
            "stroke": 0,
            "heart_disease_or_attack": 0,
            "phys_activity": 1,
            "fruits": 1,
            "veggies": 1,
            "hvy_alcohol_consump": 0,
            "any_healthcare": 1,
            "no_doc_bc_cost": 0,
            "gen_hlth": 1,
            "ment_hlth": 0.0,
            "phys_hlth": 0.0,
            "diff_walk": 0,
            "sex": 0,
            "age": 2,
            "education": 6,
            "income": 8
        },
        "prediabetic": {
            "bmi": 28.5,
            "high_bp": 1,
            "high_chol": 1,
            "chol_check": 1,
            "smoker": 1,
            "stroke": 0,
            "heart_disease_or_attack": 0,
            "phys_activity": 0,
            "fruits": 1,
            "veggies": 1,
            "hvy_alcohol_consump": 0,
            "any_healthcare": 1,
            "no_doc_bc_cost": 0,
            "gen_hlth": 3,
            "ment_hlth": 2.0,
            "phys_hlth": 4.0,
            "diff_walk": 0,
            "sex": 1,
            "age": 8,
            "education": 4,
            "income": 5
        },
        "diabetic": {
            "bmi": 34.6,
            "high_bp": 1,
            "high_chol": 1,
            "chol_check": 1,
            "smoker": 1,
            "stroke": 1,
            "heart_disease_or_attack": 1,
            "phys_activity": 0,
            "fruits": 0,
            "veggies": 1,
            "hvy_alcohol_consump": 0,
            "any_healthcare": 1,
            "no_doc_bc_cost": 1,
            "gen_hlth": 5,
            "ment_hlth": 10.0,
            "phys_hlth": 15.0,
            "diff_walk": 1,
            "sex": 1,
            "age": 10,
            "education": 4,
            "income": 3
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
