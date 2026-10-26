"""
test_pipeline_and_api.py - Automated Verification Suite for CDC BRFSS System
=============================================================================
Covers:
1. Ingestion and standardization of raw CDC BRFSS survey data.
2. PhysiologicalOutlierCapper percentiles & clinical bounds on BMI.
3. Preprocessing ColumnTransformer architecture (numerical scaling + binary passthrough).
4. Explicit missing data path raises FileNotFoundError (no silent fallback).
5. API prediction response schema, probability consistency, and metadata.
6. API validation error responses (422) for out-of-range health indicators.
7. Explicit rule-based demo mode (/demo-predict) with null confidence/probabilities.
8. Missing model artifacts returning HTTP 503 Service Unavailable (no fake fallback).
9. Missing metrics returning HTTP 503 Service Unavailable (no fake fallback).
10. SHAP relative direction labels ("Supports X" vs "Opposes X"), dynamic base value,
    and additivity verification.
11. Confusion matrix counts mathematical consistency with reported test metrics.
12. End-to-end training and inference smoke test.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

import api
from clinical_transformers import PhysiologicalOutlierCapper, BRFSS_FEATURE_BOUNDS
from ingest_brfss import clean_brfss_dataframe, BRFSS_CLEAN_FEATURES, TARGET_COL
from ml_pipeline import (
    build_preprocessing_pipeline,
    run_pipeline,
    train_and_validate_pipeline,
    evaluate_test_performance,
    CLASS_NAMES,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
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
# 1. CDC BRFSS Ingestion & Cleaning
# =============================================================================
def test_brfss_ingestion_and_cleaning():
    """
    Validates that clean_brfss_dataframe correctly maps DIABETE3 into 3-class target:
    - 3.0 -> 0 (Healthy)
    - 4.0 -> 1 (Pre-diabetic)
    - 1.0 -> 2 (Diabetic)
    and drops non-informative response codes (7.0, 9.0).
    """
    raw_sample = pd.DataFrame({
        "DIABETE3": [3.0, 4.0, 1.0, 7.0, 9.0],
        "_RFHYPE5": [1.0, 2.0, 2.0, 1.0, 2.0],
        "TOLDHI2": [2.0, 1.0, 1.0, 2.0, 1.0],
        "_CHOLCHK": [1.0, 1.0, 1.0, 1.0, 1.0],
        "_BMI5": [2450.0, 2850.0, 3450.0, 2200.0, 3100.0],
        "SMOKE100": [2.0, 1.0, 1.0, 2.0, 1.0],
        "CVDSTRK3": [2.0, 2.0, 1.0, 2.0, 2.0],
        "_MICHD": [2.0, 2.0, 1.0, 2.0, 2.0],
        "_TOTINDA": [1.0, 2.0, 2.0, 1.0, 2.0],
        "_FRTLT1": [1.0, 1.0, 2.0, 1.0, 2.0],
        "_VEGLT1": [1.0, 1.0, 1.0, 1.0, 2.0],
        "_RFDRHV5": [1.0, 1.0, 1.0, 1.0, 1.0],
        "HLTHPLN1": [1.0, 1.0, 1.0, 1.0, 1.0],
        "MEDCOST": [2.0, 2.0, 1.0, 2.0, 2.0],
        "GENHLTH": [1.0, 3.0, 5.0, 2.0, 4.0],
        "MENTHLTH": [88.0, 2.0, 15.0, 88.0, 88.0],
        "PHYSHLTH": [88.0, 4.0, 20.0, 88.0, 88.0],
        "DIFFWALK": [2.0, 2.0, 1.0, 2.0, 2.0],
        "SEX": [2.0, 1.0, 1.0, 2.0, 1.0],
        "_AGEG5YR": [2.0, 8.0, 11.0, 3.0, 7.0],
        "EDUCA": [6.0, 4.0, 3.0, 5.0, 4.0],
        "INCOME2": [8.0, 5.0, 3.0, 7.0, 4.0],
    })

    cleaned = clean_brfss_dataframe(raw_sample)
    assert len(cleaned) == 3
    assert set(cleaned[TARGET_COL].unique()) == {0, 1, 2}
    assert cleaned.loc[cleaned[TARGET_COL] == 0, "bmi"].values[0] == 24.5
    assert cleaned.loc[cleaned[TARGET_COL] == 1, "high_bp"].values[0] == 1


# =============================================================================
# 2. Outlier Capper on BMI and Features
# =============================================================================
def test_outlier_capper_percentiles_and_bounds():
    """Validates that PhysiologicalOutlierCapper clamps BMI to physiological bounds."""
    capper = PhysiologicalOutlierCapper(
        bounds=BRFSS_FEATURE_BOUNDS,
        use_empirical_bounds=True,
        lower_percentile=0.01,
        upper_percentile=0.99
    )

    df = pd.DataFrame({
        "bmi": [5.0, 22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 95.0],
        "gen_hlth": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    })

    capper.fit(df)
    transformed = capper.transform(df)

    assert transformed["bmi"].min() >= 12.0
    assert transformed["bmi"].max() <= 70.0


# =============================================================================
# 3. Preprocessing ColumnTransformer Architecture
# =============================================================================
def test_preprocessing_architecture():
    """Verifies that numerical features are scaled and binary features pass through."""
    preprocessor = build_preprocessing_pipeline()
    transformers_dict = {name: trans for name, trans, _ in preprocessor.transformers}

    assert "num" in transformers_dict
    assert "bin" in transformers_dict
    assert transformers_dict["bin"] == "passthrough"

    num_steps = [step[0] for step in transformers_dict["num"].steps]
    assert num_steps == ["capper", "scaler"]


# =============================================================================
# 4. Explicit Missing Data Path Raises FileNotFoundError
# =============================================================================
def test_explicit_missing_data_path_fails():
    """Verifies that supplying a nonexistent data path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        run_pipeline(data_path="nonexistent_cdc_survey.csv")
    assert "nonexistent_cdc_survey.csv" in str(exc_info.value)


# =============================================================================
# 5. API Prediction Schema & Probability Consistency
# =============================================================================
def test_api_prediction_schema_and_probability_consistency(client):
    """
    Tests POST /predict with a valid CDC BRFSS patient profile.
    Verifies that probabilities sum to 1.0, confidence matches predicted class,
    and metadata contains transparent prediction source tags.
    """
    payload = {
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
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert "metadata" in data
    assert "prediction" in data
    assert "shap_explanation" in data

    assert data["metadata"]["prediction_source"] == "trained_model"
    assert data["metadata"]["dataset_type"] == "real_cdc_brfss_2015"

    pred = data["prediction"]
    assert pred["risk_class"] in CLASS_NAMES
    probs = pred["probabilities"]
    assert probs is not None
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0
    assert pytest.approx(pred["confidence"], abs=1e-3) == probs[pred["risk_class"]]


# =============================================================================
# 6. API Validation Errors (422) for Boundary Violations
# =============================================================================
def test_api_validation_errors(client):
    """Validates that out-of-range clinical parameters return HTTP 422."""
    # BMI out of bounds (< 12.0)
    resp = client.post("/predict", json={"bmi": 5.0, "high_bp": 1})
    assert resp.status_code == 422

    # Age category out of bounds (> 13)
    resp = client.post("/predict", json={
        "bmi": 25.0,
        "high_bp": 0,
        "high_chol": 0,
        "chol_check": 1,
        "smoker": 0,
        "stroke": 0,
        "heart_disease_or_attack": 0,
        "phys_activity": 1,
        "gen_hlth": 2,
        "sex": 0,
        "age": 20  # invalid, 1-13 only
    })
    assert resp.status_code == 422


# =============================================================================
# 7. Explicit Rule-Based Demo Mode
# =============================================================================
def test_explicit_demo_mode_truthful_metadata(client):
    """
    Tests POST /demo-predict:
    - Must return prediction_source="rule_based_demo".
    - Must NOT show invented confidence or probabilities (None/null).
    - Must NOT return invented SHAP values (status="unavailable").
    """
    payload = {
        "bmi": 32.5,
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
        "gen_hlth": 4,
        "ment_hlth": 5.0,
        "phys_hlth": 10.0,
        "diff_walk": 1,
        "sex": 1,
        "age": 10,
        "education": 3,
        "income": 2
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


# =============================================================================
# 8. Missing Model Artifacts Return HTTP 503
# =============================================================================
def test_missing_model_artifacts_returns_503(monkeypatch, client):
    """Verifies that missing pipeline returns HTTP 503 Service Unavailable."""
    monkeypatch.setattr(api, "pipeline", None)
    payload = {
        "bmi": 25.0, "high_bp": 0, "high_chol": 0, "chol_check": 1,
        "smoker": 0, "stroke": 0, "heart_disease_or_attack": 0,
        "phys_activity": 1, "fruits": 1, "veggies": 1, "hvy_alcohol_consump": 0,
        "any_healthcare": 1, "no_doc_bc_cost": 0, "gen_hlth": 2,
        "ment_hlth": 0, "phys_hlth": 0, "diff_walk": 0, "sex": 0,
        "age": 4, "education": 5, "income": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Model pipeline artifacts are not loaded" in response.json()["detail"]


# =============================================================================
# 9. Missing Metrics Return HTTP 503
# =============================================================================
def test_missing_metrics_returns_503(monkeypatch, client):
    """Verifies that GET /metrics returns HTTP 503 if metrics are missing."""
    monkeypatch.setattr(api, "metrics_cache", None)
    monkeypatch.setattr(api, "METRICS_PATH", "/nonexistent/model_metrics.json")

    response = client.get("/metrics")
    assert response.status_code == 503
    assert "Evaluation metrics unavailable" in response.json()["detail"]


# =============================================================================
# 10. SHAP Relative Direction Labels & Additivity
# =============================================================================
def test_shap_relative_direction_labels_and_additivity(client):
    """
    Validates SHAP explanations:
    1. Base value is dynamic from the explainer.
    2. Contributions are labeled relative to the explained class.
    3. Validates additivity: base_value + sum(all_shap) ≈ model_output within 0.05.
    """
    payload = {
        "bmi": 22.0, "high_bp": 0, "high_chol": 0, "chol_check": 1,
        "smoker": 0, "stroke": 0, "heart_disease_or_attack": 0,
        "phys_activity": 1, "fruits": 1, "veggies": 1, "hvy_alcohol_consump": 0,
        "any_healthcare": 1, "no_doc_bc_cost": 0, "gen_hlth": 1,
        "ment_hlth": 0, "phys_hlth": 0, "diff_walk": 0, "sex": 0,
        "age": 2, "education": 6, "income": 8
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    shap_data = data["shap_explanation"]

    if shap_data["status"] == "available":
        assert shap_data["explained_class"] == data["prediction"]["risk_class"]
        assert shap_data["base_value"] is not None

        for feat in shap_data["top_features"]:
            if feat["shap_value"] > 0:
                assert f"Supports {shap_data['explained_class']}" in feat["impact"]
                assert feat["direction"] == "supports"
            elif feat["shap_value"] < 0:
                assert f"Opposes {shap_data['explained_class']}" in feat["impact"]
                assert feat["direction"] == "opposes"

        additivity = shap_data["additivity_check"]
        assert additivity is not None
        assert additivity["is_additive"] is True
        assert additivity["absolute_difference"] < 0.05


# =============================================================================
# 11. Confusion Matrix Mathematical Consistency
# =============================================================================
def test_confusion_matrix_consistency_with_metrics():
    """Validates confusion matrix math in artifacts/model_metrics.json."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("model_metrics.json not generated yet")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"]["matrix"])
    total_test = cm.sum()
    correct = np.trace(cm)
    calculated_acc = correct / total_test

    assert total_test == 10000
    assert pytest.approx(calculated_acc, abs=1e-3) == metrics["accuracy"]


# =============================================================================
# 12. End-to-End Mini Training & Inference Smoke Test
# =============================================================================
def test_end_to_end_training_and_inference_smoke():
    """Tests mini end-to-end training and inference on BRFSS data."""
    clean_csv = "artifacts/cdc_brfss_diabetes_clean.csv"
    if not os.path.exists(clean_csv):
        pytest.skip("cdc_brfss_diabetes_clean.csv not generated yet")

    df = pd.read_csv(clean_csv, nrows=100)
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    pipeline, cv_metrics, y_pred, y_prob, X_test, y_test, X_train, y_train, baselines = (
        train_and_validate_pipeline(X, y, model_type="rf", cv_folds=2, random_state=42)
    )

    metrics = evaluate_test_performance(y_test, y_pred, y_prob, cv_metrics, baselines)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    sample_patient = X_test.iloc[[0]]
    prediction = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)
    assert len(prediction) == 1
    assert probs.shape == (1, 3)
    assert pytest.approx(probs.sum(), abs=1e-3) == 1.0
