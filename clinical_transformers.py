"""
clinical_transformers.py - Clinical Feature Schemas & Dedicated Transformers
=============================================================================
Centralized feature schemas, discrete allowed value constraints, and standalone
transformers to guarantee clean, portable serialization across training, APIs,
and evaluation suites.
"""

from typing import Dict, Tuple, Optional, Set, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Default clinical physiological limits (aligned with API & frontend validators)
PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (1.0, 13.0),
    "bmi": (12.0, 70.0),
    "gen_hlth": (1.0, 5.0),
    "ment_hlth": (0.0, 30.0),
    "phys_hlth": (0.0, 30.0),
    "education": (1.0, 6.0),
    "income": (1.0, 8.0),
}

# CDC BRFSS Health Indicators reference bounds
BRFSS_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bmi": (12.0, 70.0),
    "gen_hlth": (1.0, 5.0),
    "ment_hlth": (0.0, 30.0),
    "phys_hlth": (0.0, 30.0),
    "age": (1.0, 13.0),
    "education": (1.0, 6.0),
    "income": (1.0, 8.0),
    "high_bp": (0.0, 1.0),
    "high_chol": (0.0, 1.0),
    "chol_check": (0.0, 1.0),
    "smoker": (0.0, 1.0),
    "stroke": (0.0, 1.0),
    "heart_disease_or_attack": (0.0, 1.0),
    "phys_activity": (0.0, 1.0),
    "fruits": (0.0, 1.0),
    "veggies": (0.0, 1.0),
    "hvy_alcohol_consump": (0.0, 1.0),
    "any_healthcare": (0.0, 1.0),
    "no_doc_bc_cost": (0.0, 1.0),
    "diff_walk": (0.0, 1.0),
    "sex": (0.0, 1.0),
}

UNIFIED_BOUNDS: Dict[str, Tuple[float, float]] = {**PHYSIOLOGICAL_BOUNDS, **BRFSS_FEATURE_BOUNDS}

# Discrete allowed sets for categorical, binary, and ordinal features
DISCRETE_ALLOWED_VALUES: Dict[str, Set[int]] = {
    "high_bp": {0, 1},
    "high_chol": {0, 1},
    "chol_check": {0, 1},
    "smoker": {0, 1},
    "stroke": {0, 1},
    "heart_disease_or_attack": {0, 1},
    "phys_activity": {0, 1},
    "fruits": {0, 1},
    "veggies": {0, 1},
    "hvy_alcohol_consump": {0, 1},
    "any_healthcare": {0, 1},
    "no_doc_bc_cost": {0, 1},
    "diff_walk": {0, 1},
    "sex": {0, 1},
    "gen_hlth": {1, 2, 3, 4, 5},
    "age": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    "education": {1, 2, 3, 4, 5, 6},
    "income": {1, 2, 3, 4, 5, 6, 7, 8},
}

INTEGER_RANGE_FEATURES: Dict[str, Tuple[int, int]] = {
    "ment_hlth": (0, 30),
    "phys_hlth": (0, 30),
}

CONTINUOUS_FLOAT_FEATURES: Dict[str, Tuple[float, float]] = {
    "bmi": (12.0, 70.0),
}

# Centralized feature schemas
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

RAW_INPUT_FEATURES = [
    "high_bp", "high_chol", "chol_check", "bmi", "smoker",
    "stroke", "heart_disease_or_attack", "phys_activity", "fruits",
    "veggies", "hvy_alcohol_consump", "any_healthcare", "no_doc_bc_cost",
    "gen_hlth", "ment_hlth", "phys_hlth", "diff_walk", "sex", "age",
    "education", "income"
]

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

CLASS_DEFINITIONS: Dict[int, str] = {
    0: "No Reported Diabetes",
    1: "Pre-diabetic",
    2: "Diabetic"
}

CLASS_NAMES = ["No Reported Diabetes", "Pre-diabetic", "Diabetic"]


def validate_survey_feature_values(row_or_df: Any) -> Dict[str, Any]:
    """
    Validates features as discrete allowed values, integer day ranges, or bounded floats.
    Catches invalid values, non-discrete entries, and out-of-range inputs.
    """
    if isinstance(row_or_df, dict):
        records = [row_or_df]
    elif isinstance(row_or_df, pd.DataFrame):
        records = row_or_df.to_dict(orient="records")
    else:
        raise TypeError("Input must be a dict or pd.DataFrame")

    violations = []
    for i, rec in enumerate(records):
        for feat, allowed_set in DISCRETE_ALLOWED_VALUES.items():
            if feat in rec and rec[feat] is not None:
                val = rec[feat]
                try:
                    int_val = int(val)
                    if int_val != val or int_val not in allowed_set:
                        violations.append({
                            "record_index": i, "feature": feat, "value": val,
                            "reason": f"Expected discrete value in {sorted(list(allowed_set))}, got {val}"
                        })
                except (ValueError, TypeError):
                    violations.append({
                        "record_index": i, "feature": feat, "value": val,
                        "reason": f"Non-numeric or non-discrete value: {val}"
                    })

        for feat, (low, high) in INTEGER_RANGE_FEATURES.items():
            if feat in rec and rec[feat] is not None:
                val = rec[feat]
                try:
                    int_val = int(val)
                    if int_val < low or int_val > high:
                        violations.append({
                            "record_index": i, "feature": feat, "value": val,
                            "reason": f"Expected integer in [{low}, {high}], got {val}"
                        })
                except (ValueError, TypeError):
                    violations.append({
                        "record_index": i, "feature": feat, "value": val,
                        "reason": f"Non-numeric integer-range value: {val}"
                    })

        for feat, (low, high) in CONTINUOUS_FLOAT_FEATURES.items():
            if feat in rec and rec[feat] is not None:
                val = rec[feat]
                try:
                    fval = float(val)
                    if fval < low or fval > high:
                        violations.append({
                            "record_index": i, "feature": feat, "value": val,
                            "reason": f"Expected continuous float in [{low}, {high}], got {val}"
                        })
                except (ValueError, TypeError):
                    violations.append({
                        "record_index": i, "feature": feat, "value": val,
                        "reason": f"Non-numeric float value: {val}"
                    })

    if violations:
        sample_violations = violations[:5]
        raise ValueError(
            f"Feature validation failed on {len(violations)} checks. Sample violations: {sample_violations}"
        )

    return {"status": "valid", "records_validated": len(records)}


class PhysiologicalOutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps survey and clinical values using strict physiological bounds and
    optional data-driven empirical percentiles fitted strictly on training data.

    Supports three ablation modes:
    - 'none': No empirical quantile clipping (only physical reference limits).
    - 'bmi_only': Empirical percentile clipping strictly on continuous BMI (leaving discrete/ordinal categories intact).
    - 'all_numerical': Legacy behavior clipping all numerical features.
    """

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        use_empirical_bounds: bool = True,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        capping_mode: str = "all_numerical"
    ):
        self.bounds = bounds or UNIFIED_BOUNDS
        self.use_empirical_bounds = use_empirical_bounds
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.capping_mode = capping_mode
        self.empirical_limits_: Dict[str, Tuple[float, float]] = {}
        self.feature_names_in_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.empirical_limits_ = {}

        for col in X.columns:
            col_str = str(col)
            phys_min, phys_max = self.bounds.get(col_str, (-np.inf, np.inf))

            apply_empirical = False
            if self.use_empirical_bounds:
                if self.capping_mode == "all_numerical":
                    apply_empirical = True
                elif self.capping_mode == "bmi_only" and col_str == "bmi":
                    apply_empirical = True
                elif self.capping_mode == "none":
                    apply_empirical = False

            if apply_empirical:
                q_low = X[col].quantile(self.lower_percentile)
                q_high = X[col].quantile(self.upper_percentile)
                col_min = max(phys_min, q_low) if not pd.isna(q_low) else phys_min
                col_max = min(phys_max, q_high) if not pd.isna(q_high) else phys_max
            else:
                col_min, col_max = phys_min, phys_max

            self.empirical_limits_[col_str] = (float(col_min), float(col_max))

        return self

    def transform(self, X):
        X_out = X.copy()
        is_df = isinstance(X_out, pd.DataFrame)
        cols = X_out.columns if is_df else [str(i) for i in range(X_out.shape[1])]

        for idx, col in enumerate(cols):
            col_str = str(col)
            if col_str in self.empirical_limits_:
                low, high = self.empirical_limits_[col_str]
                if is_df:
                    X_out[col] = X_out[col].clip(lower=low, upper=high)
                else:
                    X_out[:, idx] = np.clip(X_out[:, idx], low, high)

        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        raise ValueError("Estimator capper is not fitted yet.")
