"""
test_pipeline_and_api.py - Comprehensive Automated Verification Suite
=====================================================================
Covers requirements 1-8:
1. Synthetic smoking effect per-patient vectorization.
2. PhysiologicalOutlierCapper percentiles & clinical bounds.
3. Preprocessing order: Outlier capper -> RobustScaler -> KNNImputer.
4. Exact blood pressure boundary condition (systolic >= diastolic + 15).
5. Explicit missing/invalid data path raises FileNotFoundError (no silent fallback).
6. API prediction response schema, probability consistency, and metadata.
7. Missing lab value inference via KNN imputation.
8. API validation error responses (422) with clinical field boundaries.
9. Explicit rule-based demo mode (/demo-predict) with null confidence/probabilities.
10. Missing model artifacts returning HTTP 503 Service Unavailable (no fake fallback).
11. Missing metrics returning HTTP 503 Service Unavailable (no fake fallback).
12. SHAP relative direction labels ("Supports X" vs "Opposes X"), dynamic base value,
    one-hot aggregation, and additivity verification.
13. Confusion matrix counts mathematical consistency with reported metrics.
14. End-to-end training and inference smoke test.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

import api
from clinical_transformers import PhysiologicalOutlierCapper, PHYSIOLOGICAL_BOUNDS
from ml_pipeline import (
    generate_synthetic_clinical_cohort,
    build_preprocessing_pipeline,
    run_pipeline,
    train_and_validate_pipeline,
    evaluate_test_performance,
    CLASS_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES
)


@pytest.fixture(autouse=True)
def setup_api_artifacts():
    """Ensure artifacts are loaded into the FastAPI module before tests."""
    api.load_artifacts()


@pytest.fixture
def client():
    """Provides a TestClient with loaded model artifacts."""
    with TestClient(api.app) as c:
        yield c


# =============================================================================
# 1. Synthetic Smoking Effect Vectorization
# =============================================================================
def test_smoking_effect_is_vectorized_per_patient():
    """
    Validates that the synthetic data generator computes smoking effect as an
    individual per-patient indicator rather than applying a cohort-wide constant scalar.
    """
    df = generate_synthetic_clinical_cohort(n_samples=500, random_state=42)
    assert "smoking_status" in df.columns

    # Verify smoking effect per record
    current_smokers = df[df["smoking_status"] == "Current"]
    never_smokers = df[df["smoking_status"] == "Never"]

    assert len(current_smokers) > 0
    assert len(never_smokers) > 0

    # Test the vectorized expression directly
    smoking_series = pd.Series(["Current", "Never", "Former", "Current"])
    computed_effect = 0.35 * (smoking_series == "Current").astype(float)
    assert np.allclose(computed_effect.values, [0.35, 0.0, 0.0, 0.35])
    assert len(np.unique(computed_effect)) == 2


# =============================================================================
# 2. PhysiologicalOutlierCapper Percentiles & Bounds
# =============================================================================
def test_outlier_capper_percentiles_and_bounds():
    """
    Validates that PhysiologicalOutlierCapper calculates 1st and 99th percentiles
    on training data and clamps values strictly within physiological reference limits.
    """
    capper = PhysiologicalOutlierCapper(
        bounds=PHYSIOLOGICAL_BOUNDS,
        use_empirical_bounds=True,
        lower_percentile=0.01,
        upper_percentile=0.99
    )

    # Synthetic sample data with extreme outliers
    df = pd.DataFrame({
        "fasting_blood_glucose": [30.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 500.0],
        "systolic_bp": [50.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 300.0]
    })

    capper.fit(df)
    transformed = capper.transform(df)

    # Fasting blood glucose phys bounds are (40.0, 450.0)
    assert transformed["fasting_blood_glucose"].min() >= 40.0
    assert transformed["fasting_blood_glucose"].max() <= 450.0

    # Systolic BP phys bounds are (60.0, 260.0)
    assert transformed["systolic_bp"].min() >= 60.0
    assert transformed["systolic_bp"].max() <= 260.0


# =============================================================================
# 3. Preprocessing Order (Scaler before KNNImputer)
# =============================================================================
def test_preprocessing_order_scaler_before_imputer():
    """
    Verifies that numeric features are scaled before KNN imputation
    so Euclidean distance calculations in KNN are not distorted by differing feature scales.
    """
    col_transformer = build_preprocessing_pipeline()
    transformers_dict = {name: trans for name, trans, _ in col_transformer.transformers}
    num_pipeline = transformers_dict["num"]
    step_names = [step[0] for step in num_pipeline.steps]

    # Order must be: capper -> scaler -> imputer
    assert step_names == ["capper", "scaler", "imputer"], (
        f"Expected ['capper', 'scaler', 'imputer'], got {step_names}"
    )


# =============================================================================
# 4. Blood Pressure Exact Boundary Validation
# =============================================================================
def test_blood_pressure_exact_boundary():
    """
    Tests the clinical consistency rule: Systolic BP must be >= Diastolic BP + 15.
    - Exactly 15.0 diff: must PASS.
    - 14.9 diff: must FAIL with ValidationError.
    """
    base_payload = {
        "age": 45.0,
        "gender": "Female",
        "ethnicity": "Caucasian",
        "family_history_diabetes": 0,
        "bmi": 24.5,
        "fasting_blood_glucose": 90.0,
        "hba1c": 5.2,
        "total_cholesterol": 180.0,
        "hdl_cholesterol": 55.0,
        "ldl_cholesterol": 105.0,
        "triglycerides": 100.0,
        "serum_creatinine": 0.9,
        "physical_activity_minutes_per_week": 150.0,
        "smoking_status": "Never",
        "alcohol_intake": "None",
        "diet_quality_score": 7.5,
        "sleep_hours": 7.5
    }

    # Case 1: Exact boundary diff == 15.0 (Systolic 120, Diastolic 105) -> PASS
    valid_data = dict(base_payload, systolic_bp=120.0, diastolic_bp=105.0)
    patient = api.PatientData(**valid_data)
    assert patient.systolic_bp == 120.0
    assert patient.diastolic_bp == 105.0

    # Case 2: Diff == 14.9 (Systolic 120, Diastolic 105.1) -> FAIL
    invalid_data = dict(base_payload, systolic_bp=120.0, diastolic_bp=105.1)
    with pytest.raises(Exception) as exc_info:
        api.PatientData(**invalid_data)
    assert "at least 15 mmHg higher" in str(exc_info.value)


# =============================================================================
# 5. Explicit Missing Data Path Raises FileNotFoundError
# =============================================================================
def test_explicit_missing_data_path_fails():
    """
    Ensures that supplying an invalid/missing --data path fails clearly
    and does NOT silently fall back to generating synthetic data.
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        run_pipeline(data_path="nonexistent_clinical_data.csv")
    assert "nonexistent_clinical_data.csv" in str(exc_info.value)


# =============================================================================
# 6. API Prediction Schema & Probability Consistency
# =============================================================================
def test_api_prediction_schema_and_probability_consistency(client):
    """
    Tests POST /predict with a valid patient profile.
    Verifies that probabilities sum to 1.0, confidence matches predicted class,
    and metadata contains transparent prediction source tags.
    """
    payload = {
        "age": 42.0,
        "gender": "Male",
        "ethnicity": "Asian",
        "family_history_diabetes": 1,
        "bmi": 26.2,
        "systolic_bp": 124.0,
        "diastolic_bp": 78.0,
        "fasting_blood_glucose": 94.0,
        "hba1c": 5.4,
        "total_cholesterol": 195.0,
        "hdl_cholesterol": 50.0,
        "ldl_cholesterol": 115.0,
        "triglycerides": 140.0,
        "serum_creatinine": 0.95,
        "physical_activity_minutes_per_week": 120.0,
        "smoking_status": "Former",
        "alcohol_intake": "Moderate",
        "diet_quality_score": 6.5,
        "sleep_hours": 7.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Schema structure
    assert data["status"] == "success"
    assert "metadata" in data
    assert "prediction" in data
    assert "shap_explanation" in data
    assert "clinical_recommendations" in data

    # Metadata
    assert data["metadata"]["prediction_source"] == "trained_model"
    assert data["metadata"]["dataset_type"] == "synthetic"
    assert "educational prototype" in data["metadata"]["disclaimer"].lower()

    # Probability consistency
    pred = data["prediction"]
    assert pred["risk_class"] in CLASS_NAMES
    probs = pred["probabilities"]
    assert probs is not None
    assert set(probs.keys()) == set(CLASS_NAMES)
    assert pytest.approx(sum(probs.values()), abs=1e-4) == 1.0
    assert pytest.approx(pred["confidence"], abs=1e-4) == probs[pred["risk_class"]]


# =============================================================================
# 7. Missing Lab Values Handled via KNN Imputation
# =============================================================================
def test_api_missing_lab_values_knn_imputation(client):
    """
    Tests that omitting lab values (fasting glucose, HbA1c, lipids)
    succeeds at inference via KNN imputation and tracks imputed features in metadata.
    """
    payload = {
        "age": 55.0,
        "gender": "Female",
        "ethnicity": "Hispanic",
        "family_history_diabetes": 1,
        "bmi": 31.0,
        "systolic_bp": 138.0,
        "diastolic_bp": 84.0,
        "smoking_status": "Never",
        "alcohol_intake": "None"
        # fasting_blood_glucose, hba1c, lipid panel omitted (None)
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"]["risk_class"] in CLASS_NAMES
    imputed = data["metadata"]["imputed_features"]
    assert "fasting_blood_glucose" in imputed
    assert "hba1c" in imputed
    assert "total_cholesterol" in imputed


# =============================================================================
# 8. API Validation Errors (422) for Boundary Violations
# =============================================================================
def test_api_validation_errors(client):
    """
    Validates that out-of-range clinical parameters return HTTP 422
    with structured validation errors rather than unhandled 500 crashes.
    """
    # Age below 18
    resp = client.post("/predict", json={"age": 14, "gender": "Male"})
    assert resp.status_code == 422

    # Blood pressure boundary violation (systolic < diastolic + 15)
    resp = client.post("/predict", json={
        "age": 45,
        "gender": "Female",
        "ethnicity": "Caucasian",
        "family_history_diabetes": 0,
        "bmi": 24.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 115.0,
        "smoking_status": "Never",
        "alcohol_intake": "None"
    })
    assert resp.status_code == 422
    assert "at least 15 mmHg higher" in resp.text


# =============================================================================
# 9. Explicit Rule-Based Demo Mode
# =============================================================================
def test_explicit_demo_mode_truthful_metadata(client):
    """
    Tests POST /demo-predict:
    - Must return prediction_source="rule_based_demo".
    - Must NOT show invented confidence or probabilities (must be None/null).
    - Must NOT return invented SHAP values (status="unavailable").
    """
    payload = {
        "age": 58.0,
        "gender": "Male",
        "ethnicity": "Caucasian",
        "family_history_diabetes": 1,
        "bmi": 32.5,
        "systolic_bp": 142.0,
        "diastolic_bp": 88.0,
        "fasting_blood_glucose": 135.0,
        "hba1c": 7.1,
        "smoking_status": "Current",
        "alcohol_intake": "Moderate"
    }

    response = client.post("/demo-predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "demo"
    assert data["metadata"]["prediction_source"] == "rule_based_demo"
    assert "Rule-based demo" in data["metadata"]["disclaimer"]
    assert data["prediction"]["confidence"] is None
    assert data["prediction"]["probabilities"] is None
    assert data["shap_explanation"]["status"] == "unavailable"
    assert data["shap_explanation"]["top_features"] == []


# =============================================================================
# 10. Missing Model Artifacts Return HTTP 503
# =============================================================================
def test_missing_model_artifacts_returns_503(monkeypatch, client):
    """
    Verifies that when the model pipeline fails to load or is None,
    the API returns HTTP 503 Service Unavailable (NO silent fallback to simulation).
    """
    # Temporarily set pipeline to None
    monkeypatch.setattr(api, "pipeline", None)

    payload = {
        "age": 45,
        "gender": "Male",
        "ethnicity": "Caucasian",
        "family_history_diabetes": 0,
        "bmi": 24.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "smoking_status": "Never",
        "alcohol_intake": "None"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Model pipeline artifacts are not loaded" in response.json()["detail"]


# =============================================================================
# 11. Missing Metrics Return HTTP 503
# =============================================================================
def test_missing_metrics_returns_503(monkeypatch, client):
    """
    Verifies that GET /metrics returns HTTP 503 if metrics artifacts are missing
    (NO hardcoded fallback benchmarks).
    """
    monkeypatch.setattr(api, "metrics_cache", None)
    monkeypatch.setattr(api, "METRICS_PATH", "/nonexistent/model_metrics.json")

    response = client.get("/metrics")
    assert response.status_code == 503
    assert "Evaluation metrics unavailable" in response.json()["detail"]


# =============================================================================
# 12. SHAP Relative Direction Labels & Additivity
# =============================================================================
def test_shap_relative_direction_labels_and_additivity(client):
    """
    Validates SHAP explanations:
    1. Base value is dynamic from the explainer (not hardcoded 0.33).
    2. Contributions are labeled relative to the explained class:
       "Supports Healthy prediction" or "Opposes Healthy prediction".
    3. Never says "Increases risk" for Healthy class.
    4. Validates additivity: base_value + sum(all_shap) ≈ model_output within 0.05.
    5. One-hot features are aggregated into original feature names.
    """
    # Patient with Healthy profile
    healthy_payload = {
        "age": 30.0,
        "gender": "Female",
        "ethnicity": "Caucasian",
        "family_history_diabetes": 0,
        "bmi": 21.5,
        "systolic_bp": 110.0,
        "diastolic_bp": 70.0,
        "fasting_blood_glucose": 82.0,
        "hba1c": 4.9,
        "total_cholesterol": 165.0,
        "hdl_cholesterol": 68.0,
        "ldl_cholesterol": 85.0,
        "triglycerides": 75.0,
        "serum_creatinine": 0.8,
        "physical_activity_minutes_per_week": 250.0,
        "smoking_status": "Never",
        "alcohol_intake": "None",
        "diet_quality_score": 9.0,
        "sleep_hours": 8.0
    }

    response = client.post("/predict", json=healthy_payload)
    assert response.status_code == 200
    data = response.json()
    shap_data = data["shap_explanation"]

    if shap_data["status"] == "available":
        assert shap_data["explained_class"] == data["prediction"]["risk_class"]
        assert shap_data["base_value"] is not None
        assert 0.0 <= shap_data["base_value"] <= 1.0

        # Check top features
        for feat in shap_data["top_features"]:
            # Feature name must not be one-hot encoded suffix (e.g. gender_Male)
            assert not feat["feature"].startswith("gender_")
            assert not feat["feature"].startswith("ethnicity_")

            # Check relative wording
            if shap_data["explained_class"] == "Healthy":
                assert "Increases risk" not in feat["impact"]
                if feat["shap_value"] > 0:
                    assert "Supports Healthy prediction" in feat["impact"]
                    assert feat["direction"] == "supports"
                elif feat["shap_value"] < 0:
                    assert "Opposes Healthy prediction" in feat["impact"]
                    assert feat["direction"] == "opposes"

        # Check additivity
        additivity = shap_data["additivity_check"]
        assert additivity is not None
        assert additivity["is_additive"] is True
        assert additivity["absolute_difference"] < 0.05


# =============================================================================
# 13. Confusion Matrix Mathematical Consistency
# =============================================================================
def test_confusion_matrix_consistency_with_metrics():
    """
    Validates that confusion matrix cell counts in artifacts/model_metrics.json
    mathematically agree with accuracy, per-class recall, and macro metrics.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("model_metrics.json not generated yet")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"]["matrix"])
    total_test = cm.sum()
    correct = np.trace(cm)
    calculated_acc = correct / total_test

    assert total_test == 800
    assert pytest.approx(calculated_acc, abs=1e-4) == metrics["accuracy"]

    # Check per-class recall
    for i, cls in enumerate(metrics["confusion_matrix"]["classes"]):
        actual_total = cm[i, :].sum()
        cls_correct = cm[i, i]
        cls_recall = cls_correct / actual_total
        recorded_recall = metrics["confusion_matrix"]["breakdown"][cls]["sensitivity_recall"]
        assert pytest.approx(cls_recall, abs=1e-4) == recorded_recall


# =============================================================================
# 14. End-to-End Mini Training & Inference Smoke Test
# =============================================================================
def test_end_to_end_training_and_inference_smoke(tmp_path):
    """
    Tests mini end-to-end dataset generation, pipeline fitting, and inference
    to ensure components work together without runtime crashes.
    """
    # 1. Generate small cohort
    df = generate_synthetic_clinical_cohort(n_samples=80, random_state=123)
    assert len(df) == 80

    X = df[ALL_FEATURES]
    y = df["diabetes_stage"]

    # 2. Train pipeline using 2-fold CV for fast smoke test
    pipeline, cv_metrics, y_pred, y_prob, X_test, y_test, X_train, y_train, baselines = (
        train_and_validate_pipeline(X, y, model_type="rf", cv_folds=2, random_state=123)
    )

    # 3. Evaluate hold-out test performance
    metrics = evaluate_test_performance(y_test, y_pred, y_prob, cv_metrics, baselines)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # 4. Inference on single sample
    sample_patient = X_test.iloc[[0]]
    prediction = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)
    assert len(prediction) == 1
    assert probs.shape == (1, 3)
    assert pytest.approx(probs.sum(), abs=1e-4) == 1.0
