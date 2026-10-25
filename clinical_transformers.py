"""
clinical_transformers.py - Standalone Clinical Feature Transformers
====================================================================
Isolated transformer definitions to guarantee clean, portable serialization
and unpickling across training scripts, background workers, and REST APIs.

Design Decisions:
1. Physiological bounds mirror clinical limits in api.py and frontend forms.
2. Empirical percentiles are computed strictly on the training partition
   (1st and 99th percentiles) to prevent test-set data leakage.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Default clinical physiological limits (aligned with API & frontend validators)
PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 100.0),
    "bmi": (12.0, 70.0),
    "fasting_blood_glucose": (40.0, 450.0),
    "hba1c": (3.5, 18.0),
    "total_cholesterol": (80.0, 450.0),
    "hdl_cholesterol": (15.0, 140.0),
    "ldl_cholesterol": (30.0, 320.0),
    "triglycerides": (30.0, 750.0),
    "systolic_bp": (60.0, 260.0),
    "diastolic_bp": (40.0, 160.0),
    "serum_creatinine": (0.2, 8.0),
    "physical_activity_minutes_per_week": (0.0, 1200.0),
    "diet_quality_score": (1.0, 10.0),
    "sleep_hours": (2.0, 16.0),
}


class PhysiologicalOutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical laboratory and anthropometric values using strict clinical
    physiological reference limits combined with data-driven empirical percentiles
    (1st and 99th percentiles calculated strictly on training split).
    """

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        use_empirical_bounds: bool = True,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ):
        self.bounds = bounds or PHYSIOLOGICAL_BOUNDS
        self.use_empirical_bounds = use_empirical_bounds
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.empirical_limits_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.empirical_limits_ = {}
        for col in X.columns:
            col_str = str(col)
            phys_min, phys_max = self.bounds.get(col_str, (-np.inf, np.inf))
            if self.use_empirical_bounds:
                q_low = X[col].quantile(self.lower_percentile)
                q_high = X[col].quantile(self.upper_percentile)
                # Keep capping within physiological bounds
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
