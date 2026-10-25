"""
api.py - Clinical Diabetes Risk Machine Learning REST API (Educational Prototype)
==================================================================================
Scope & Ethical Framing:
This API provides multi-class risk classification (Healthy, Pre-diabetic, Diabetic)
for educational demonstration. It is trained on synthetic benchmark data and MUST NOT
be used for clinical diagnostics or patient care.

Key Architectural Guarantees:
1. Strict Pydantic v2 validation adhering to physiological clinical bounds.
2. Blood pressure boundary condition: Systolic must be at least 15 mmHg higher than Diastolic.
3. Transparent inference:
   - When model artifacts are loaded, runs inference through the trained pipeline.
   - If model artifacts are missing, returns HTTP 503 Service Unavailable (NO silent fallback).
4. Truthful SHAP XAI:
   - Base value is read directly from explainer for the explained class.
   - Contributions are labeled relative to the explained class ('Supports X' vs 'Opposes X').
   - Additivity is validated: base_value + sum(shap_values) ≈ model probability output.
   - If SHAP is unavailable, returns status='unavailable' with NO hand-written formulas.
5. Explicit Demo Mode:
   - POST /predict: strictly evaluates trained model pipeline.
   - POST /demo-predict: clearly labeled rule-based demo returning confidence=None, probabilities=None.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import joblib

from clinical_transformers import PhysiologicalOutlierCapper, PHYSIOLOGICAL_BOUNDS

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

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
    title="Clinical Diabetes Risk Intelligence API (Educational Prototype)",
    description=(
        "Educational machine learning decision support service for multi-class diabetes risk "
        "classification with verified SHAP explainability. Trained on synthetic benchmark data."
    ),
    version="1.1.0",
)

# Enable CORS for frontend integration
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
# Pydantic Schemas with Strict Clinical Validation
# -----------------------------------------------------------------------------
class PatientData(BaseModel):
    """
    Patient clinical profile with validated physiological limits.
    Laboratory biomarkers support Optional[float] = None; missing values are
    safely imputed by the trained KNN pipeline at inference.
    """
    # Demographics
    age: float = Field(..., ge=18.0, le=100.0, description="Patient age in years (18 - 100)")
    gender: Literal["Male", "Female", "Other"] = Field(..., description="Biological sex / gender identity")
    ethnicity: Literal["Caucasian", "African American", "Asian", "Hispanic", "Other"] = Field(
        ..., description="Self-reported racial or ethnic background"
    )
    family_history_diabetes: int = Field(
        ..., ge=0, le=1, description="Immediate biological family history of diabetes (1 = Yes, 0 = No)"
    )

    # Anthropometrics & Blood Pressure
    bmi: float = Field(..., ge=12.0, le=70.0, description="Body Mass Index in kg/m² (12.0 - 70.0)")
    systolic_bp: float = Field(..., ge=60.0, le=260.0, description="Systolic Blood Pressure in mmHg (60 - 260)")
    diastolic_bp: float = Field(..., ge=40.0, le=160.0, description="Diastolic Blood Pressure in mmHg (40 - 160)")

    # Laboratory Panel (Optional: if omitted, imputed by pipeline KNNImputer)
    fasting_blood_glucose: Optional[float] = Field(
        default=None, ge=40.0, le=450.0, description="Fasting plasma glucose in mg/dL (40 - 450)"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=3.5, le=18.0, description="Glycated Hemoglobin percentage (3.5 - 18.0%)"
    )
    total_cholesterol: Optional[float] = Field(
        default=None, ge=80.0, le=450.0, description="Serum Total Cholesterol in mg/dL (80 - 450)"
    )
    hdl_cholesterol: Optional[float] = Field(
        default=None, ge=15.0, le=140.0, description="HDL Cholesterol in mg/dL (15 - 140)"
    )
    ldl_cholesterol: Optional[float] = Field(
        default=None, ge=30.0, le=320.0, description="LDL Cholesterol in mg/dL (30 - 320)"
    )
    triglycerides: Optional[float] = Field(
        default=None, ge=30.0, le=750.0, description="Serum Triglycerides in mg/dL (30 - 750)"
    )
    serum_creatinine: Optional[float] = Field(
        default=None, ge=0.2, le=8.0, description="Serum Creatinine in mg/dL (0.2 - 8.0)"
    )

    # Lifestyle & Behavioral
    physical_activity_minutes_per_week: Optional[float] = Field(
        default=None, ge=0.0, le=1200.0, description="Weekly exercise in minutes (0 - 1200)"
    )
    smoking_status: Literal["Never", "Former", "Current"] = Field(
        ..., description="Smoking status: 'Never', 'Former', or 'Current'"
    )
    alcohol_intake: Literal["None", "Moderate", "Heavy"] = Field(
        ..., description="Alcohol consumption frequency: 'None', 'Moderate', or 'Heavy'"
    )
    diet_quality_score: Optional[float] = Field(
        default=None, ge=1.0, le=10.0, description="Nutritional health index (1 = poor, 10 = optimal)"
    )
    sleep_hours: Optional[float] = Field(
        default=None, ge=2.0, le=16.0, description="Nightly sleep duration in hours (2.0 - 16.0)"
    )

    @model_validator(mode="after")
    def validate_blood_pressure(self):
        """Clinical consistency: Systolic BP must be at least 15 mmHg higher than Diastolic BP."""
        if self.systolic_bp < self.diastolic_bp + 15:
            raise ValueError(
                f"Systolic BP ({self.systolic_bp} mmHg) must be at least 15 mmHg higher than Diastolic BP ({self.diastolic_bp} mmHg)."
            )
        return self


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
    "hba1c": "Glycated Hemoglobin (HbA1c)",
    "fasting_blood_glucose": "Fasting Plasma Glucose",
    "bmi": "Body Mass Index (BMI)",
    "age": "Patient Age",
    "family_history_diabetes": "Family History of Diabetes",
    "triglycerides": "Serum Triglycerides",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "hdl_cholesterol": "HDL ('Good') Cholesterol",
    "ldl_cholesterol": "LDL ('Bad') Cholesterol",
    "total_cholesterol": "Total Serum Cholesterol",
    "physical_activity_minutes_per_week": "Physical Activity Level",
    "diet_quality_score": "Diet Quality Score",
    "smoking_status": "Smoking History",
    "alcohol_intake": "Alcohol Consumption",
    "sleep_hours": "Sleep Duration",
    "serum_creatinine": "Serum Creatinine",
    "ethnicity": "Racial / Ethnic Predisposition",
    "gender": "Biological Sex",
}


def generate_feature_clinical_insight(feature: str, value: Any, shap_val: float, explained_class: str) -> str:
    """Explains feature contribution relative to the specific explained class."""
    supports = shap_val > 0
    verb = "supports" if supports else "opposes"

    if feature == "hba1c":
        if explained_class == "Healthy":
            if supports:
                return f"HbA1c of {value}% is optimal (<5.7%), supporting the Healthy classification."
            return f"HbA1c of {value}% is borderline/elevated, opposing the Healthy classification."
        else:
            if supports:
                return f"HbA1c of {value}% is elevated (≥5.7%), supporting the {explained_class} classification."
            return f"HbA1c of {value}% is relatively low, opposing the {explained_class} classification."

    elif feature == "fasting_blood_glucose":
        if explained_class == "Healthy":
            if supports:
                return f"Fasting glucose of {value} mg/dL is within normal limits (70-99 mg/dL), supporting Healthy status."
            return f"Fasting glucose of {value} mg/dL is elevated, opposing Healthy status."
        else:
            if supports:
                return f"Fasting glucose of {value} mg/dL is above normal fasting thresholds, supporting {explained_class}."
            return f"Fasting glucose of {value} mg/dL is within or near baseline, opposing {explained_class}."

    elif feature == "bmi":
        if explained_class == "Healthy":
            if supports:
                return f"BMI of {value} kg/m² reflects normal adiposity, supporting Healthy classification."
            return f"BMI of {value} kg/m² is elevated, opposing Healthy classification."
        else:
            if supports:
                return f"Elevated BMI of {value} kg/m² reflects excess adiposity and insulin resistance, supporting {explained_class}."
            return f"Normal BMI of {value} kg/m² opposes {explained_class} classification."

    elif feature == "physical_activity_minutes_per_week":
        if supports:
            return f"Physical activity ({value} min/wk) aligns with physiological patterns that support {explained_class}."
        return f"Physical activity level ({value} min/wk) opposes {explained_class} classification."

    return f"Feature value of {value} contributes {shap_val:+.4f} in probability space and {verb} {explained_class} prediction."


def generate_clinical_recommendations(risk_tier: str, patient: PatientData) -> List[ClinicalRecommendation]:
    """Generates educational preventive guidelines based on risk tier."""
    recs = []

    if risk_tier == "Diabetic":
        recs.append(ClinicalRecommendation(
            category="Medical Consultation",
            title="Formal Diagnostic Evaluation by Licensed Healthcare Provider",
            description="Consult a physician or endocrinologist for confirmatory lab tests (repeat fasting glucose / HbA1c), renal surveillance, and medical assessment.",
            urgency="Immediate"
        ))
        recs.append(ClinicalRecommendation(
            category="Clinical Monitoring",
            title="Glycemic Surveillance & Follow-Up Protocol",
            description="Discuss structured home glucose monitoring and 90-day repeat HbA1c intervals with a healthcare provider.",
            urgency="Immediate"
        ))
        recs.append(ClinicalRecommendation(
            category="Nutrition & Diet",
            title="Medical Nutrition Therapy (MNT)",
            description="Consult a Registered Dietitian for individualized carbohydrate budgeting and whole-food meal planning.",
            urgency="Moderate"
        ))

    elif risk_tier == "Pre-diabetic":
        recs.append(ClinicalRecommendation(
            category="Physical Activity",
            title="Diabetes Prevention Program (DPP) Exercise Target",
            description="Target 150-300 minutes per week of moderate-intensity aerobic exercise combined with resistance training.",
            urgency="Moderate"
        ))
        recs.append(ClinicalRecommendation(
            category="Nutrition & Diet",
            title="5-7% Weight Reduction & Fiber Optimization",
            description="Target a 5-7% body weight reduction if indicated, emphasizing high dietary fiber (>30g/day) and low-glycemic foods.",
            urgency="Moderate"
        ))
        recs.append(ClinicalRecommendation(
            category="Clinical Monitoring",
            title="6-Month Metabolic Surveillance",
            description="Schedule repeat fasting plasma glucose, HbA1c, and lipid panel in 6 months to track metabolic trajectory.",
            urgency="Moderate"
        ))

    else:  # Healthy
        recs.append(ClinicalRecommendation(
            category="Nutrition & Diet",
            title="Sustain Balanced Nutritional Habits",
            description="Continue adhering to a balanced, nutrient-dense diet rich in fiber, lean proteins, and unsaturated fats.",
            urgency="Routine"
        ))
        recs.append(ClinicalRecommendation(
            category="Physical Activity",
            title="Maintain Aerobic and Strength Routine",
            description="Continue meeting or exceeding WHO guidelines of at least 150 minutes of weekly moderate exercise.",
            urgency="Routine"
        ))
        recs.append(ClinicalRecommendation(
            category="Clinical Monitoring",
            title="Annual Preventive Health Checkup",
            description="Maintain routine annual physical exams and metabolic wellness screenings.",
            urgency="Routine"
        ))

    return recs


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_diabetes_risk(patient: PatientData):
    """
    Evaluates patient data against the trained machine learning pipeline.
    If the model pipeline artifact is missing, returns HTTP 503 (NO silent fallback).
    """
    global pipeline, explainer, metadata

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifacts are not loaded. Please train the model using 'python ml_pipeline.py' first."
        )

    try:
        raw_dict = patient.model_dump()
        imputed_features = [k for k, v in raw_dict.items() if v is None]

        # Convert None to NaN so Scikit-Learn RobustScaler + KNNImputer processes them
        patient_df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in raw_dict.items()}])

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["classifier"]

        # Run pipeline inference
        X_trans = preprocessor.transform(patient_df)
        probs_raw = model.predict_proba(X_trans)[0]
        class_id = int(np.argmax(probs_raw))
        risk_class = CLASS_NAMES[class_id]
        confidence = float(probs_raw[class_id])

        probs_dict = {
            CLASS_NAMES[i]: round(float(probs_raw[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        # Compute SHAP explanation strictly from explainer
        shap_explanation = None
        if explainer is not None and metadata is not None:
            try:
                # Extract actual expected value from explainer for the explained class
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base_val = float(ev[class_id])
                else:
                    base_val = float(ev)

                raw_shap = explainer.shap_values(X_trans)
                if isinstance(raw_shap, list):
                    class_shap = raw_shap[class_id][0]
                elif hasattr(raw_shap, "values"):
                    class_shap = raw_shap.values[0, :, class_id]
                elif raw_shap.ndim == 3:
                    class_shap = raw_shap[0, :, class_id]
                else:
                    class_shap = raw_shap[0]

                feat_names = metadata.get("encoded_feature_names", [])
                categorical_roots = metadata.get("categorical_features", [])

                # Aggregate one-hot dummy features back into parent feature groups
                feature_scores: Dict[str, float] = {}
                for name, score in zip(feat_names, class_shap):
                    matched_cat = next((cat for cat in categorical_roots if name.startswith(cat)), None)
                    root = matched_cat if matched_cat else name
                    feature_scores[root] = feature_scores.get(root, 0.0) + float(score)

                # Validate additivity: base_value + sum(shap_values) ≈ model output
                all_shap_sum = float(sum(feature_scores.values()))
                reconstructed = base_val + all_shap_sum
                additivity_diff = abs(reconstructed - confidence)

                sorted_feats = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                top_contribs = []
                for feat, s_val in sorted_feats[:6]:
                    pval = raw_dict.get(feat, "N/A")
                    if pval is None:
                        pval = "Imputed (KNN)"

                    if s_val > 0:
                        impact_label = f"Supports {risk_class} prediction"
                        direction = "supports"
                    elif s_val < 0:
                        impact_label = f"Opposes {risk_class} prediction"
                        direction = "opposes"
                    else:
                        impact_label = f"Neutral regarding {risk_class}"
                        direction = "neutral"

                    insight = generate_feature_clinical_insight(feat, pval, s_val, risk_class)
                    top_contribs.append(FeatureContribution(
                        feature=feat,
                        display_name=FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
                        patient_value=pval,
                        shap_value=round(float(s_val), 4),
                        impact=impact_label,
                        direction=direction,
                        clinical_insight=insight
                    ))

                top_sum = float(sum(f.shap_value for f in top_contribs))
                omitted_sum = round(all_shap_sum - top_sum, 4)

                shap_explanation = SHAPExplanation(
                    status="available",
                    explained_class=risk_class,
                    explained_class_id=class_id,
                    output_space="probability",
                    base_value=round(base_val, 4),
                    top_features=top_contribs,
                    omitted_features_shap_sum=omitted_sum,
                    additivity_check=AdditivityCheck(
                        base_value=round(base_val, 4),
                        all_features_shap_sum=round(all_shap_sum, 4),
                        reconstructed_output=round(reconstructed, 4),
                        model_output=round(confidence, 4),
                        absolute_difference=round(additivity_diff, 5),
                        is_additive=bool(additivity_diff < 0.05)
                    )
                )
            except Exception as e:
                logger.warning("SHAP attribution computation failed: %s", e)
                shap_explanation = SHAPExplanation(
                    status="unavailable",
                    reason=f"SHAP explanation calculation failed: {str(e)}",
                    top_features=[]
                )
        else:
            shap_explanation = SHAPExplanation(
                status="unavailable",
                reason="SHAP explainer artifact is not loaded.",
                top_features=[]
            )

        recs = generate_clinical_recommendations(risk_class, patient)

        resp_meta = ResponseMetadata(
            prediction_source="trained_model",
            model_version=metadata.get("model_version", "1.1.0-educational") if metadata else "1.1.0-educational",
            model_type=type(model).__name__,
            dataset_type="synthetic",
            explanation_status="available" if shap_explanation.status == "available" else "unavailable",
            explanation_method="shap_tree_explainer" if shap_explanation.status == "available" else None,
            imputed_features=imputed_features,
            disclaimer="Educational prototype evaluated on synthetic benchmark data. Not for clinical diagnosis."
        )

        return PredictionResponse(
            status="success",
            metadata=resp_meta,
            prediction=PredictionResult(
                class_id=class_id,
                risk_class=risk_class,
                confidence=round(confidence, 4),
                probabilities=probs_dict
            ),
            shap_explanation=shap_explanation,
            clinical_recommendations=recs,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error("Inference execution error: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(err)}"
        )


@app.post("/demo-predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def demo_predict_rules(patient: PatientData):
    """
    Explicit rule-based educational demo.
    Returns confidence=None and probabilities=None so it cannot be misrepresented
    as a machine learning prediction.
    """
    fbg = patient.fasting_blood_glucose or 95.0
    hba1c = patient.hba1c or 5.4

    if hba1c >= 6.5 or fbg >= 126.0:
        cls_id = 2
        risk_name = "Diabetic"
    elif hba1c >= 5.7 or fbg >= 100.0:
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
            model_type="ADA Deterministic Threshold Rules",
            dataset_type="none",
            explanation_status="not_applicable",
            explanation_method=None,
            imputed_features=[],
            disclaimer="Rule-based demo — not a trained model prediction. No confidence probability or SHAP is measured."
        ),
        prediction=PredictionResult(
            class_id=cls_id,
            risk_class=risk_name,
            confidence=None,
            probabilities=None
        ),
        shap_explanation=SHAPExplanation(
            status="unavailable",
            reason="Demo mode evaluates heuristic ADA rules. SHAP requires a trained machine learning model.",
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
        "service": "Clinical Diabetes Risk Stratification API (Educational Prototype)",
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
    """
    Returns hold-out test set performance metrics from saved artifacts.
    If artifacts are missing, returns HTTP 503 (NO hardcoded fake numbers).
    """
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
    """Returns verified reference clinical profiles for testing."""
    return {
        "healthy": {
            "age": 34.0,
            "gender": "Female",
            "ethnicity": "Caucasian",
            "family_history_diabetes": 0,
            "bmi": 22.4,
            "systolic_bp": 115.0,
            "diastolic_bp": 72.0,
            "fasting_blood_glucose": 86.0,
            "hba1c": 5.1,
            "total_cholesterol": 172.0,
            "hdl_cholesterol": 64.0,
            "ldl_cholesterol": 92.0,
            "triglycerides": 80.0,
            "serum_creatinine": 0.8,
            "physical_activity_minutes_per_week": 240.0,
            "smoking_status": "Never",
            "alcohol_intake": "None",
            "diet_quality_score": 8.5,
            "sleep_hours": 7.5
        },
        "prediabetic": {
            "age": 52.0,
            "gender": "Male",
            "ethnicity": "Hispanic",
            "family_history_diabetes": 1,
            "bmi": 29.8,
            "systolic_bp": 136.0,
            "diastolic_bp": 86.0,
            "fasting_blood_glucose": 116.0,
            "hba1c": 6.1,
            "total_cholesterol": 218.0,
            "hdl_cholesterol": 41.0,
            "ldl_cholesterol": 138.0,
            "triglycerides": 195.0,
            "serum_creatinine": 1.05,
            "physical_activity_minutes_per_week": 60.0,
            "smoking_status": "Former",
            "alcohol_intake": "Moderate",
            "diet_quality_score": 4.8,
            "sleep_hours": 6.2
        },
        "diabetic": {
            "age": 59.0,
            "gender": "Male",
            "ethnicity": "African American",
            "family_history_diabetes": 1,
            "bmi": 34.2,
            "systolic_bp": 148.0,
            "diastolic_bp": 92.0,
            "fasting_blood_glucose": 168.0,
            "hba1c": 8.4,
            "total_cholesterol": 248.0,
            "hdl_cholesterol": 34.0,
            "ldl_cholesterol": 158.0,
            "triglycerides": 280.0,
            "serum_creatinine": 1.35,
            "physical_activity_minutes_per_week": 20.0,
            "smoking_status": "Current",
            "alcohol_intake": "Heavy",
            "diet_quality_score": 3.0,
            "sleep_hours": 5.5
        }
    }


@app.get("/clinical-ranges")
async def get_clinical_ranges():
    """Returns valid clinical reference ranges and units for form validation."""
    return {
        "ranges": {
            "age": {"min": 18, "max": 100, "unit": "years", "normal": "18 - 65"},
            "bmi": {"min": 12.0, "max": 70.0, "unit": "kg/m²", "normal": "18.5 - 24.9"},
            "systolic_bp": {"min": 60, "max": 260, "unit": "mmHg", "normal": "< 120"},
            "diastolic_bp": {"min": 40, "max": 160, "unit": "mmHg", "normal": "< 80"},
            "fasting_blood_glucose": {"min": 40, "max": 450, "unit": "mg/dL", "normal": "70 - 99"},
            "hba1c": {"min": 3.5, "max": 18.0, "unit": "%", "normal": "< 5.7%"},
            "total_cholesterol": {"min": 80, "max": 450, "unit": "mg/dL", "normal": "< 200"},
            "hdl_cholesterol": {"min": 15, "max": 140, "unit": "mg/dL", "normal": ">= 50"},
            "ldl_cholesterol": {"min": 30, "max": 320, "unit": "mg/dL", "normal": "< 100"},
            "triglycerides": {"min": 30, "max": 750, "unit": "mg/dL", "normal": "< 150"},
            "serum_creatinine": {"min": 0.2, "max": 8.0, "unit": "mg/dL", "normal": "0.6 - 1.2"},
            "physical_activity_minutes_per_week": {"min": 0, "max": 1200, "unit": "min/wk", "normal": ">= 150"},
            "diet_quality_score": {"min": 1.0, "max": 10.0, "unit": "scale (1-10)", "normal": ">= 7.0"},
            "sleep_hours": {"min": 2.0, "max": 16.0, "unit": "hours/day", "normal": "7.0 - 9.0"},
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
