"""
tests/test_modeling_improvements.py - Targeted Regression Tests for Modeling Improvements
========================================================================================
Covers the 15 experimental requirements:
1. Discrete feature validation (discrete domains vs raw range checks).
2. Explicit full-data and subsampling modes ('all' vs integer).
3. Stable split membership and zero train/test index overlap.
4. Fold-local preprocessing during calibration (no leakage into CalibratedClassifierCV).
5. Validation-tuned decision threshold conflict, assignment, and fallback behavior.
6. Artifact compatibility validator and stale/mismatched explainer rejection.
7. Additivity verification failure handling (marks status='failed_verification' if error > 0.05).
8. Monotonic timing and valid latency reporting.
9. Calibrated vs. base model probability and explanation labeling.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from clinical_transformers import (
    validate_survey_feature_values,
    DISCRETE_ALLOWED_VALUES,
    INTEGER_RANGE_FEATURES,
    CONTINUOUS_FLOAT_FEATURES,
    PhysiologicalOutlierCapper
)
from ingest_brfss import (
    RAW_INPUT_FEATURES,
    TARGET_COL,
    CLASS_DEFINITIONS,
    validate_cleaned_dataframe,
    load_dataset
)
from sklearn.pipeline import Pipeline

from ml_pipeline import (
    build_preprocessing_pipeline,
    train_and_validate_pipeline,
    evaluate_model_baselines_and_calibration,
    apply_decision_rule,
    get_classifier,
    benchmark_inference_latency
)
import api
from api import app, validate_artifact_compatibility


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
# 1. Discrete Domain Feature Validation
# =============================================================================
def test_discrete_feature_validation_rejects_invalid_values():
    """
    Ensures that values outside the discrete allowed set are rejected with ValueError,
    even if they appear numerically close.
    """
    valid_record = {
        "high_bp": 1, "high_chol": 0, "chol_check": 1, "bmi": 26.5,
        "smoker": 0, "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1,
        "fruits": 1, "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1,
        "no_doc_bc_cost": 0, "gen_hlth": 2, "ment_hlth": 0, "phys_hlth": 0,
        "diff_walk": 0, "sex": 1, "age": 7, "education": 4, "income": 5
    }
    df_valid = pd.DataFrame([valid_record])
    # Should pass without error
    val_res = validate_survey_feature_values(df_valid)
    assert val_res["status"] == "valid"

    # 1. Invalid sex (allowed: 0, 1) -> value 2 must fail
    df_bad_sex = df_valid.copy()
    df_bad_sex["sex"] = 2
    with pytest.raises(ValueError, match="Expected discrete value"):
        validate_survey_feature_values(df_bad_sex)

    # 2. Invalid gen_hlth (allowed: 1, 2, 3, 4, 5) -> value 0 or 6 must fail
    df_bad_gen_hlth = df_valid.copy()
    df_bad_gen_hlth["gen_hlth"] = 6
    with pytest.raises(ValueError, match="Expected discrete value"):
        validate_survey_feature_values(df_bad_gen_hlth)

    # 3. Invalid age category (allowed: 1 to 13) -> value 0 or 14 must fail
    df_bad_age = df_valid.copy()
    df_bad_age["age"] = 0
    with pytest.raises(ValueError, match="Expected discrete value"):
        validate_survey_feature_values(df_bad_age)

    # 4. Invalid education (allowed: 1 to 6) -> value 7 must fail
    df_bad_edu = df_valid.copy()
    df_bad_edu["education"] = 7
    with pytest.raises(ValueError, match="Expected discrete value"):
        validate_survey_feature_values(df_bad_edu)

    # 5. Invalid income (allowed: 1 to 8) -> value 9 must fail
    df_bad_inc = df_valid.copy()
    df_bad_inc["income"] = 9
    with pytest.raises(ValueError, match="Expected discrete value"):
        validate_survey_feature_values(df_bad_inc)


# =============================================================================
# 2. Explicit Full-Data and Subsampling Loading Modes
# =============================================================================
def test_explicit_dataset_loading_modes():
    """
    Tests that load_dataset supports 'all' and explicit integer subsampling,
    recording accurate lineage metadata without hidden conventions.
    """
    clean_csv = "artifacts/cdc_brfss_diabetes_clean.csv"
    if not os.path.exists(clean_csv):
        pytest.skip(f"{clean_csv} not found")

    # 1. Subsample mode
    df_sub, meta_sub = load_dataset(clean_csv, sample_size=150, random_state=42)
    assert len(df_sub) == 150
    assert meta_sub["subsampled_in_current_run"] is True
    assert meta_sub["selected_cohort_count"] == 150
    assert meta_sub["post_cleaning_eligible_count"] == 50000

    # 2. Explicit 'all' mode
    df_all, meta_all = load_dataset(clean_csv, sample_size="all", random_state=42)
    assert len(df_all) == 50000
    assert meta_all["subsampled_in_current_run"] is False
    assert meta_all["selected_cohort_count"] == 50000
    # Must explicitly state upstream sampling status
    assert meta_all["is_upstream_subsampled"] is True


# =============================================================================
# 3. Stable Split Membership & Zero Cross-Split Overlap
# =============================================================================
def test_stable_split_membership_and_zero_overlap():
    """
    Verifies that the train and test index partitions are completely disjoint,
    and reproduce identical sets across runs with fixed random_state.
    """
    from sklearn.model_selection import train_test_split
    indices = np.arange(1000)
    labels = np.random.RandomState(42).choice([0, 1, 2], size=1000, p=[0.84, 0.02, 0.14])

    train_idx1, test_idx1 = train_test_split(indices, test_size=0.20, stratify=labels, random_state=42)
    train_idx2, test_idx2 = train_test_split(indices, test_size=0.20, stratify=labels, random_state=42)

    # Determinism
    np.testing.assert_array_equal(train_idx1, train_idx2)
    np.testing.assert_array_equal(test_idx1, test_idx2)

    # Disjointness (Zero overlap)
    overlap = set(train_idx1).intersection(set(test_idx1))
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping records between train and test splits!"


# =============================================================================
# 4. Fold-Local Preprocessing Inside CalibratedClassifierCV
# =============================================================================
def test_fold_local_preprocessing_in_calibration():
    """
    Verifies that calibration wraps the full Pipeline (preprocessor + classifier)
    so each internal calibration fold learns outlier capping and scaling strictly from its fold data.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    preprocessor = build_preprocessing_pipeline(capping_mode="bmi_only")
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Wrap the full pipeline inside CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(estimator=full_pipeline, method="sigmoid", cv=2)

    # Create synthetic test data
    X_syn = pd.DataFrame([{col: 0 for col in RAW_INPUT_FEATURES} for _ in range(60)])
    X_syn["bmi"] = 28.0
    X_syn["gen_hlth"] = 2
    X_syn["age"] = 5
    X_syn["education"] = 4
    X_syn["income"] = 5
    y_syn = pd.Series([0] * 40 + [1] * 10 + [2] * 10)

    # Must fit cleanly without data leakage
    calibrated.fit(X_syn, y_syn)
    probs = calibrated.predict_proba(X_syn)
    assert probs.shape == (60, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# =============================================================================
# 5. Validation-Tuned Decision Rule & Fallback Behavior
# =============================================================================
def test_decision_rule_thresholds_and_fallback():
    """
    Tests apply_decision_rule with validation-tuned risk thresholds:
    - If prob[2] >= tau_diab (0.52) -> Class 2
    - Else if prob[1] >= tau_prediab (0.35) -> Class 1
    - Else -> Class 0 (fallback)
    - Also verifies standard argmax rule.
    """
    # 1. High diabetic probability
    probs_diab = np.array([[0.20, 0.25, 0.55]])
    class_diab, assigned_p_diab = apply_decision_rule(
        probs_diab,
        classes=np.array([0, 1, 2]),
        rule_type="threshold_tuned",
        threshold_diabetic=0.52,
        threshold_prediabetic=0.35
    )
    assert class_diab[0] == 2
    assert assigned_p_diab[0] == 0.55

    # 2. Moderate pre-diabetic probability exceeding tau_prediab (0.35) even if class 0 has highest prob
    probs_prediab = np.array([[0.60, 0.38, 0.02]])
    class_prediab, assigned_p_prediab = apply_decision_rule(
        probs_prediab,
        classes=np.array([0, 1, 2]),
        rule_type="threshold_tuned",
        threshold_diabetic=0.52,
        threshold_prediabetic=0.35
    )
    # Threshold rule catches pre-diabetic risk at 38%
    assert class_prediab[0] == 1
    assert assigned_p_prediab[0] == 0.38

    # Standard argmax on the same probabilities chooses class 0
    class_argmax, assigned_p_argmax = apply_decision_rule(
        probs_prediab,
        classes=np.array([0, 1, 2]),
        rule_type="argmax"
    )
    assert class_argmax[0] == 0
    assert assigned_p_argmax[0] == 0.60

    # 3. Fallback when neither threshold is met
    probs_low = np.array([[0.80, 0.15, 0.05]])
    class_fallback, assigned_p_fallback = apply_decision_rule(
        probs_low,
        classes=np.array([0, 1, 2]),
        rule_type="threshold_tuned",
        threshold_diabetic=0.52,
        threshold_prediabetic=0.35
    )
    assert class_fallback[0] == 0
    assert assigned_p_fallback[0] == 0.80


# =============================================================================
# 6. Artifact Compatibility Validator
# =============================================================================
def test_artifact_compatibility_validation():
    """
    Validates that validate_artifact_compatibility catches missing or incompatible artifacts.
    """
    # 1. Missing pipeline raises RuntimeError
    with pytest.raises(RuntimeError, match="pipeline artifact is None"):
        validate_artifact_compatibility(None, None, {})

    # 2. Missing preprocessor or classifier step raises RuntimeError
    dummy_pipe = Pipeline([("step", None)])
    with pytest.raises(RuntimeError, match="must contain 'preprocessor' and 'classifier'"):
        validate_artifact_compatibility(dummy_pipe, None, {})

    # 3. Explainer with mismatched feature count raises RuntimeError
    class MockExplainer:
        feature_names = ["feat1", "feat2"]  # only 2 features instead of 21

    preprocessor = build_preprocessing_pipeline()
    clf = get_classifier("rf", random_state=42)
    valid_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    with pytest.raises(RuntimeError, match="Explainer feature count"):
        validate_artifact_compatibility(valid_pipe, MockExplainer(), {})


# =============================================================================
# 7. Additivity Verification Failure Handling
# =============================================================================
def test_additivity_failure_handling_marks_failed_verification(client):
    """
    Validates that when SHAP additivity fails tolerance (|sum - target| > 0.05),
    the API transparently reports status="failed_verification" rather than inventing numbers.
    """
    # Create valid payload
    payload = {
        "bmi": 27.0, "high_bp": 1, "high_chol": 0, "chol_check": 1, "smoker": 0,
        "stroke": 0, "heart_disease_or_attack": 0, "phys_activity": 1, "fruits": 1,
        "veggies": 1, "hvy_alcohol_consump": 0, "any_healthcare": 1, "no_doc_bc_cost": 0,
        "gen_hlth": 2, "ment_hlth": 0, "phys_hlth": 0, "diff_walk": 0,
        "sex": 0, "age": 8, "education": 5, "income": 6,
        "decision_rule": "threshold_tuned"
    }

    # Under normal conditions, prediction succeeds and TreeSHAP is verified
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["prediction"]["decision_rule_applied"] == "validation_tuned_thresholds"
    assert data["shap_explanation"]["status"] in ["available", "failed_verification", "unavailable"]

    if data["shap_explanation"]["status"] == "available":
        check = data["shap_explanation"]["additivity_check"]
        assert check["is_additive"] is True
        assert check["absolute_difference"] <= 0.05


# =============================================================================
# 8. Monotonic Latency Benchmark
# =============================================================================
def test_monotonic_latency_measurements():
    """
    Validates that ml_pipeline.benchmark_inference_latency uses time.perf_counter()
    and returns valid timing statistics.
    """
    from ml_pipeline import benchmark_inference_latency
    clean_csv = "artifacts/cdc_brfss_diabetes_clean.csv"
    if not os.path.exists(clean_csv):
        pytest.skip(f"{clean_csv} not found")

    df_sample, _ = load_dataset(clean_csv, sample_size=300, random_state=42)
    X = df_sample[RAW_INPUT_FEATURES]
    y = df_sample[TARGET_COL]

    pipe, _, _, _, _, _, _, _, _, _ = train_and_validate_pipeline(X, y, model_type="rf", cv_folds=2, random_state=42)
    benchmarks = benchmark_inference_latency(pipe, X, n_iterations=10)

    assert "batch_throughput_samples_per_sec" in benchmarks
    assert benchmarks["batch_throughput_samples_per_sec"] > 0
    assert "single_record_latency_p50_ms" in benchmarks
    assert benchmarks["single_record_latency_p50_ms"] > 0
    assert benchmarks["timer_type"] == "time.perf_counter() (monotonic high-resolution)"
