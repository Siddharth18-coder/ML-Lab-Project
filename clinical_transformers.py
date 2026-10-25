"""
clinical_transformers.py - Standalone Clinical Feature Transformers
====================================================================
Isolated transformer definitions to guarantee clean, portable serialization
and unpickling across training scripts, background workers, and REST APIs.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Default clinical physiological bounds
PHYSIOLOGICAL_BOUNDS = {
    "age": (18.0, 95.0),
    "bmi": (14.0, 65.0),
    "fasting_blood_glucose": (50.0, 400.0),
    "hba1c": (3.8, 16.0),
    "total_cholesterol": (90.0, 420.0),
    "hdl_cholesterol": (15.0, 120.0),
    "ldl_cholesterol": (35.0, 300.0),
    "triglycerides": (35.0, 600.0),
    "systolic_bp": (70.0, 240.0),
    "diastolic_bp": (40.0, 140.0),
    "serum_creatinine": (0.3, 6.0),
    "physical_activity_minutes_per_week": (0.0, 800.0),
    "diet_quality_score": (1.0, 10.0),
    "sleep_hours": (3.0, 13.0),
}


class PhysiologicalOutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical laboratory and anthropometric values using strict clinical
    physiological reference limits combined with data-driven empirical percentiles
    (1st and 99th percentiles calculated on training split only).
    """

    def __init__(self, bounds: Dict[str, Tuple[float, float]] = None, use_empirical_bounds: bool = True):
        self.bounds = bounds or PHYSIOLOGICAL_BOUNDS
        self.use_empirical_bounds = use_empirical_bounds
        self.empirical_limits_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.empirical_limits_ = {}
        for col in X.columns:
            col_str = str(col)
            phys_min, phys_max = self.bounds.get(col_str, (-np.inf, np.inf))
            if self.use_empirical_bounds:
                q01 = X[col].quantile(0.005)
                q99 = X[col].quantile(0.995)
                col_min = max(phys_min, q01) if not pd.isna(q01) else phys_min
                col_max = min(phys_max, q99) if not pd.isna(q99) else phys_max
            else:
                col_min, col_max = phys_min, phys_max
            self.empirical_limits_[col_str] = (col_min, col_max)
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
