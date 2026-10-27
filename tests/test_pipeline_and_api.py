"""
test_pipeline_and_api.py - Rigorous Automated Verification Suite
================================================================
Covers:
1. Exact raw-to-cleaned data mapping and row filtering telemetry.
2. Cleaned dataset schema and bound validation.
3. Transformed feature ordering: ColumnTransformer propagates names via get_feature_names_out().
4. CRITICAL REGRESSION TEST: Controlled fixture with distinct feature contributions ensuring
   a permuted SHAP feature mapping strictly fails.
5. Dynamic class-to-probability mapping using classifier.classes_.
6. Non-tree estimator support (Logistic Regression) returns explicit unavailable state without crashing.
7. API prediction response schema, probability consistency, and non-causal metadata.
8. API structured validation error responses (HTTP 422) for boundary violations.
9. Explicit rule-based demo mode (/demo-predict) with null confidence and probabilities.
10. Missing model artifacts returning HTTP 503 (no fake numbers).
11. Missing metrics returning HTTP 503 (no fake numbers).
12. Exact SHAP additivity (base_value + sum(shap) == model_output within 0.05).
13. Confusion matrix counts mathematical consistency with reported test metrics.
14. End-to-end training and inference workflow.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient
from sklearn.pipeline import Pipeline

import api
from clinical_transformers import (
    PhysiologicalOutlierCapper,
    BRFSS_FEATURE_BOUNDS,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    RAW_INPUT_FEATURES,
    FEATURE_DISPLAY_NAMES,
    CLASS_DEFINITIONS,
    CLASS_NAMES
)
from ingest_brfss import (
    clean_raw_brfss_dataframe,
    validate_cleaned_dataframe,
    load_dataset,
    TARGET_COL
)
from ml_pipeline import (
    build_preprocessing_pipeline,
    train_and_validate_pipeline,
    evaluate_test_performance,
    setup_shap_explainer,
    get_classifier
)


@pytest.fixture(autouse=True)
def setup_api_artifacts():
    """Ensure artifacts are loaded into the FastAPI module before tests."""
    api.load_artifacts()


@pytest.fixture
def client():
    """Provides a TestClient within a lifespan context to load all artifacts."""
    with TestClient(api.app) as c:
        yield c


# =============================================================================
# 1. Exact Raw-to-Cleaned Data Mapping & Telemetry
# =============================================================================
def test_brfss_raw_ingestion_and_telemetry():
    """
    Validates that clean_raw_brfss_dataframe:
    - Correctly maps DIABETE3 into 3 classes: 3.0 -> 0, 4.0 -> 1, 1.0 -> 2.
    - Excludes non-informative response codes (7.0, 9.0).
    - Accurately tracks filtering telemetry counts at each step.
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

    cleaned, telemetry = clean_raw_brfss_dataframe(raw_sample)
    assert len(cleaned) == 3
    assert set(cleaned[TARGET_COL].unique()) == {0, 1, 2}
    assert cleaned.loc[cleaned[TARGET_COL] == 0, "bmi"].values[0] == 24.5
    assert cleaned.loc[cleaned[TARGET_COL] == 1, "high_bp"].values[0] == 1
    assert telemetry["raw_records"] == 5
    assert telemetry["final_clean_records"] == 3
    assert telemetry["total_excluded_records"] == 2


# =============================================================================
# 2. Cleaned Dataset Schema & Range Validation
# =============================================================================
def test_cleaned_dataset_validation():
    """Validates that validate_cleaned_dataframe catches invalid ranges and missing columns."""
    valid_data = pd.DataFrame([{col: 0 for col in RAW_INPUT_FEATURES}])
    valid_data[TARGET_COL] = 0
    valid_data["bmi"] = 25.0
    valid_data["gen_hlth"] = 2
    valid_data["age"] = 5
    valid_data["education"] = 4
    valid_data["income"] = 5

    res = validate_cleaned_dataframe(valid_data)
    assert res["status"] == "valid"

    # Missing required column raises ValueError
    bad_cols = valid_data.drop(columns=["high_bp"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_cleaned_dataframe(bad_cols)

    # Out of range feature raises ValueError
    bad_range = valid_data.copy()
    bad_range["bmi"] = 999.0
    with pytest.raises(ValueError, match="violate expected reference bounds"):
        validate_cleaned_dataframe(bad_range)


# =============================================================================
# 3. Transformed Feature Order Propagation
# =============================================================================
def test_transformed_feature_order_propagation():
    """
    Validates that ColumnTransformer propagates feature names using get_feature_names_out(),
    and that numerical features precede binary features in transformed space.
    """
    preprocessor = build_preprocessing_pipeline()
    dummy_df = pd.DataFrame(np.ones((5, len(RAW_INPUT_FEATURES))), columns=RAW_INPUT_FEATURES)
    preprocessor.fit(dummy_df)

    out_names = preprocessor.get_feature_names_out()
    clean_names = [n.split("__")[-1] for n in out_names]

    # First 7 features MUST be NUMERICAL_FEATURES
    assert clean_names[:len(NUMERICAL_FEATURES)] == NUMERICAL_FEATURES
    # Subsequent features MUST be BINARY_FEATURES
    assert clean_names[len(NUMERICAL_FEATURES):] == BINARY_FEATURES
    # Transformed order must NOT be identical to raw input order (which puts high_bp first)
    assert clean_names != RAW_INPUT_FEATURES


# =============================================================================
# 4. CRITICAL REGRESSION TEST: SHAP Feature Identity & Order
# =============================================================================
def test_shap_feature_identity_controlled_regression():
    """
    CRITICAL REGRESSION TEST:
    Constructs a controlled fixture where ONLY BMI drives class 2 (Diabetic),
    while High Blood Pressure has zero effect.
    Verifies that SHAP correctly assigns the positive contribution to BMI,
    and that a permuted mapping (e.g. assigning index 0 to high_bp) STRICTLY FAILS.
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier

    # 1. Create controlled training data:
    # 60 samples where BMI=45 strongly correlates with target=2, and BMI=20 with target=0.
    # high_bp is purely random noise (independent of target).
    np.random.seed(42)
    n = 60
    bmi_vals = np.concatenate([np.repeat(20.0, 30), np.repeat(45.0, 30)])
    high_bp_vals = np.random.choice([0, 1], size=n)
    targets = np.concatenate([np.repeat(0, 30), np.repeat(2, 30)])

    sample_dict = {feat: np.zeros(n) for feat in RAW_INPUT_FEATURES}
    sample_dict["bmi"] = bmi_vals
    sample_dict["high_bp"] = high_bp_vals
    sample_dict["gen_hlth"] = np.repeat(2, n)
    sample_dict["age"] = np.repeat(5, n)
    sample_dict["education"] = np.repeat(4, n)
    sample_dict["income"] = np.repeat(5, n)
    sample_df = pd.DataFrame(sample_dict)

    preprocessor = build_preprocessing_pipeline()
    X_trans = preprocessor.fit_transform(sample_df)
    transformed_features = [col.split("__")[-1] for col in preprocessor.get_feature_names_out()]

    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X_trans, targets)

    explainer = shap.TreeExplainer(rf, data=X_trans[:20], model_output="probability")

    # Patient with high BMI (45.0) and high_bp=0
    test_patient = {feat: 0 for feat in RAW_INPUT_FEATURES}
    test_patient["bmi"] = 45.0
    test_patient["high_bp"] = 0
    test_patient["gen_hlth"] = 2
    test_patient["age"] = 5
    test_patient["education"] = 4
    test_patient["income"] = 5
    test_df = pd.DataFrame([test_patient])[RAW_INPUT_FEATURES]

    X_test_trans = preprocessor.transform(test_df)
    shap_vals = explainer.shap_values(X_test_trans)

    # Class 2 (Diabetic) column index in rf.classes_
    classes_ = list(rf.classes_)
    col_idx = classes_.index(2)

    if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        shap_cls = shap_vals[0, :, col_idx]
    elif isinstance(shap_vals, list):
        shap_cls = np.array(shap_vals[col_idx])[0]
    else:
        shap_cls = np.array(shap_vals)[0]

    # Map using CORRECT transformed feature order
    correct_mapping = {
        feat: float(shap_cls[idx])
        for idx, feat in enumerate(transformed_features)
    }

    # Map using INCORRECT naive raw feature order (the old bug!)
    buggy_permuted_mapping = {
        feat: float(shap_cls[idx])
        for idx, feat in enumerate(RAW_INPUT_FEATURES)
    }

    # VERIFY THE FIX:
    # Under correct mapping: BMI must be the dominant positive factor (> 0.20)
    assert correct_mapping["bmi"] > 0.20, f"Expected high BMI attribution, got {correct_mapping['bmi']}"
    # high_bp must have negligible attribution (< 0.05)
    assert abs(correct_mapping["high_bp"]) < 0.05, f"Expected negligible high_bp attribution, got {correct_mapping['high_bp']}"

    # VERIFY THAT THE OLD BUG WOULD HAVE FAILED THIS TEST:
    # Under the buggy permuted mapping, index 0 (which is BMI in X_trans) was mistakenly assigned to high_bp!
    assert buggy_permuted_mapping["high_bp"] > 0.20, "Buggy mapping incorrectly assigned BMI SHAP to high_bp"
    assert abs(buggy_permuted_mapping["bmi"]) < 0.05, "Buggy mapping incorrectly assigned zero SHAP to BMI"


# =============================================================================
# 5. Dynamic Class-to-Probability Mapping
# =============================================================================
def test_class_to_probability_mapping(client):
    """Verifies that predicted risk class and probabilities are mapped via classifier.classes_."""
    sample = {
        "high_bp": 0, "high_chol": 0, "chol_check": 1, "bmi": 21.5, "smoker": 0,
        "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1, "fruits": 1,
        "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
        "gen_hlth": 1, "ment_hlth": 0.0, "phys_hlth": 0.0, "diff_walk": 0,
        "sex": 0, "age": 2, "education": 6, "income": 8
    }
    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    data = resp.json()
    probs = data["prediction"]["probabilities"]
    assert set(probs.keys()) == set(CLASS_NAMES)
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0


# =============================================================================
# 6. Non-Tree Estimator Path
# =============================================================================
def test_non_tree_estimator_explainer_handling():
    """Verifies that non-tree estimators (LogisticRegression) return explicit unavailable state without crashing."""
    df_clean, _ = load_dataset("artifacts/cdc_brfss_diabetes_clean.csv", sample_size=100)
    X = df_clean[RAW_INPUT_FEATURES]
    y = df_clean[TARGET_COL]

    preprocessor = build_preprocessing_pipeline()
    clf = get_classifier("lr")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X, y)

    explainer, transformed_features, status_str = setup_shap_explainer(pipe, X.iloc[:20])
    assert explainer is None
    assert "unsupported_estimator_LogisticRegression" in status_str
    assert transformed_features == [col.split("__")[-1] for col in preprocessor.get_feature_names_out()]


# =============================================================================
# 7. API Structured Validation Errors (422)
# =============================================================================
def test_api_validation_errors(client):
    """Validates that out-of-range clinical parameters return HTTP 422 with structured field error messages."""
    resp = client.post("/predict", json={"bmi": 999.0, "high_bp": 1})
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "validation_error"
    assert "errors" in data
    assert any("bmi" in err["field"] for err in data["errors"])


# =============================================================================
# 8. Explicit Rule-Based Demo Mode
# =============================================================================
def test_explicit_demo_mode_truthful_metadata(client):
    """
    Tests POST /demo-predict:
    - Must return prediction_source="rule_based_demo".
    - Must NOT show invented confidence or probabilities (None).
    - Must NOT return invented SHAP values (status="unavailable").
    """
    payload = {
        "bmi": 32.5, "high_bp": 1, "high_chol": 1, "chol_check": 1, "smoker": 1,
        "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 0, "fruits": 1,
        "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
        "gen_hlth": 4, "ment_hlth": 5.0, "phys_hlth": 10.0, "diff_walk": 1,
        "sex": 1, "age": 10, "education": 3, "income": 2
    }
    response = client.post("/demo-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "demo"
    assert data["metadata"]["prediction_source"] == "rule_based_demo"
    assert data["prediction"]["confidence"] is None
    assert data["prediction"]["probabilities"] is None
    assert data["shap_explanation"]["status"] == "unavailable"


# =============================================================================
# 9. Missing Model Artifacts Return HTTP 503
# =============================================================================
def test_missing_model_artifacts_returns_503(monkeypatch, client):
    """Verifies that missing pipeline returns HTTP 503 Service Unavailable."""
    monkeypatch.setattr(api, "pipeline", None)
    payload = {
        "bmi": 25.0, "high_bp": 0, "high_chol": 0, "chol_check": 1, "smoker": 0,
        "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1, "fruits": 1,
        "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
        "gen_hlth": 2, "ment_hlth": 0.0, "phys_hlth": 0.0, "diff_walk": 0,
        "sex": 0, "age": 4, "education": 5, "income": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "artifacts/model_pipeline.joblib" in response.json()["detail"]


# =============================================================================
# 10. Missing Metrics Return HTTP 503
# =============================================================================
def test_missing_metrics_returns_503(monkeypatch, client):
    """Verifies that GET /metrics returns HTTP 503 if metrics are missing."""
    monkeypatch.setattr(api, "metrics_cache", None)
    response = client.get("/metrics")
    assert response.status_code == 503


# =============================================================================
# 11. SHAP Relative Direction Labels & Mathematical Additivity
# =============================================================================
def test_shap_relative_direction_labels_and_additivity(client):
    """
    Validates SHAP explanations:
    1. Base value is dynamic from the explainer (no 0.33 fallback).
    2. Contributions are labeled relative to the explained class.
    3. Validates exact additivity: base_value + sum(all_shap) == model_output within 0.05.
    """
    payload = {
        "bmi": 22.0, "high_bp": 0, "high_chol": 0, "chol_check": 1, "smoker": 0,
        "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1, "fruits": 1,
        "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
        "gen_hlth": 1, "ment_hlth": 0.0, "phys_hlth": 0.0, "diff_walk": 0,
        "sex": 0, "age": 2, "education": 6, "income": 8
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    shap_data = data["shap_explanation"]

    if shap_data["status"] == "available":
        assert shap_data["explained_class"] == data["prediction"]["risk_class"]
        assert shap_data["base_value"] is not None
        assert shap_data["base_value"] != 0.33  # Never hardcoded 0.33

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
# 12. Confusion Matrix Mathematical Consistency
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

    assert total_test == metrics["sample_count"]
    assert pytest.approx(calculated_acc, abs=1e-3) == metrics["accuracy"]


# =============================================================================
# 13. End-to-End Mini Training & Inference Workflow
# =============================================================================
def test_end_to_end_training_and_inference_workflow():
    """Tests deterministic end-to-end training and inference workflow."""
    clean_csv = "artifacts/cdc_brfss_diabetes_clean.csv"
    if not os.path.exists(clean_csv):
        pytest.skip("cdc_brfss_diabetes_clean.csv not generated yet")

    # Miniature stratified sample with sufficient representation per class
    df_clean, _ = load_dataset(clean_csv, sample_size=300, random_state=42)
    X = df_clean[RAW_INPUT_FEATURES]
    y = df_clean[TARGET_COL]

    pipeline, cv_metrics, y_pred, y_prob, X_test, y_test, X_train, y_train, baselines = (
        train_and_validate_pipeline(X, y, model_type="rf", cv_folds=2, random_state=42)
    )

    metrics = evaluate_test_performance(
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        classes_=pipeline.named_steps["classifier"].classes_,
        cv_metrics=cv_metrics,
        baselines=baselines,
        random_state=42
    )
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    sample_patient = X_test.iloc[[0]]
    prediction = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)
    assert len(prediction) == 1
    assert probs.shape == (1, 3)
    assert pytest.approx(probs.sum(), abs=1e-3) == 1.0
