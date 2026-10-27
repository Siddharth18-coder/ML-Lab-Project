"""
ingest_brfss.py - CDC BRFSS Real-World Dataset Ingestion and Standardization
=============================================================================
Data Provenance & Scientific Context:
- Source: U.S. Centers for Disease Control and Prevention (CDC)
- Survey: Behavioral Risk Factor Surveillance System (BRFSS) 2015 Annual Survey
- Official CDC Survey Documentation: https://www.cdc.gov/brfss/annual_data/annual_2015.html
- Scope: Cross-sectional telephone survey of non-institutionalized U.S. adults.
- Sample Note: The complete-case cleaned dataset represents a convenience cohort of
  survey respondents with complete answers across 21 specific health indicators;
  it is not weighted for population-representative census estimates without sampling weights.

Target Variable Definition (DIABETE3):
- 0 = No Reported Diabetes (No diabetes, DIABETE3 == 3.0)
- 1 = Pre-diabetic (Told pre-diabetes or borderline diabetes, DIABETE3 == 4.0)
- 2 = Diabetic (Told diabetes, DIABETE3 == 1.0)
- Excluded: Gestational diabetes (2.0), Don't know / Not sure (7.0), Refused (9.0), Missing (NaN).

21 Standardized Features:
- Binary: high_bp, high_chol, chol_check, smoker, stroke, heart_disease_or_attack,
  phys_activity, fruits, veggies, hvy_alcohol_consump, any_healthcare, no_doc_bc_cost,
  diff_walk, sex
- Numerical / Ordinal: bmi, gen_hlth, ment_hlth, phys_hlth, age, education, income
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from clinical_transformers import (
    RAW_INPUT_FEATURES,
    BRFSS_FEATURE_BOUNDS,
    CLASS_DEFINITIONS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("BRFSS_Ingestion")

PROJECT_ROOT = Path(__file__).resolve().parent

BRFSS_RAW_COLS = [
    "DIABETE3", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5",
    "SMOKE100", "CVDSTRK3", "_MICHD", "_TOTINDA", "_FRTLT1",
    "_VEGLT1", "_RFDRHV5", "HLTHPLN1", "MEDCOST", "GENHLTH",
    "MENTHLTH", "PHYSHLTH", "DIFFWALK", "SEX", "_AGEG5YR",
    "EDUCA", "INCOME2"
]

TARGET_COL = "diabetes_stage"


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_cleaned_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Strict validation of cleaned cohort format.
    Checks required columns, allowed values, target labels, missingness, and feature types.
    """
    expected_cols = [TARGET_COL] + RAW_INPUT_FEATURES
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Cleaned dataset validation failed: Missing required columns: {missing_cols}. "
            f"Ensure data was produced by ingest_brfss.py."
        )

    # 1. Missingness check
    null_counts = df[expected_cols].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Cleaned dataset contains unexpected null values: {cols_with_nulls}")

    # 2. Target validation
    unique_targets = set(df[TARGET_COL].unique())
    allowed_targets = {0, 1, 2}
    if not unique_targets.issubset(allowed_targets):
        raise ValueError(
            f"Invalid target values in {TARGET_COL}: {unique_targets}. "
            f"Allowed values are 0 (No Reported Diabetes), 1 (Pre-diabetic), 2 (Diabetic)."
        )

    # 3. Feature range validation
    range_violations = {}
    for feat in RAW_INPUT_FEATURES:
        low, high = BRFSS_FEATURE_BOUNDS[feat]
        min_val = float(df[feat].min())
        max_val = float(df[feat].max())
        if min_val < low or max_val > high:
            range_violations[feat] = {"min": min_val, "max": max_val, "expected": (low, high)}

    if range_violations:
        raise ValueError(f"Cleaned dataset features violate expected reference bounds: {range_violations}")

    class_counts = df[TARGET_COL].value_counts().to_dict()
    class_dist = {CLASS_DEFINITIONS[int(k)]: int(v) for k, v in class_counts.items()}

    return {
        "status": "valid",
        "sample_count": len(df),
        "feature_count": len(RAW_INPUT_FEATURES),
        "target_distribution": class_dist,
        "features": RAW_INPUT_FEATURES
    }


def clean_raw_brfss_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw CDC BRFSS dataframe into 21 standardized predictors and 3-class target.
    Tracks exact row removal telemetry at each filtering stage.
    """
    initial_count = len(raw_df)
    telemetry: Dict[str, Any] = {"raw_records": initial_count, "filtering_stages": {}}
    prev_count = initial_count

    # 1. Target DIABETE3: 3.0 -> 0 (No diabetes), 4.0 -> 1 (Pre-diabetic), 1.0 -> 2 (Diabetic)
    df = raw_df[raw_df["DIABETE3"].isin([1.0, 3.0, 4.0])].copy()
    target_map = {3.0: 0, 4.0: 1, 1.0: 2}
    df[TARGET_COL] = df["DIABETE3"].map(target_map)
    telemetry["filtering_stages"]["target_validity_diabete3"] = prev_count - len(df)
    prev_count = len(df)

    # 2. High Blood Pressure: _RFHYPE5 (1=No, 2=Yes)
    df = df[df["_RFHYPE5"].isin([1.0, 2.0])]
    df["high_bp"] = (df["_RFHYPE5"] == 2.0).astype(int)
    telemetry["filtering_stages"]["high_bp"] = prev_count - len(df)
    prev_count = len(df)

    # 3. High Cholesterol: TOLDHI2 (1=Yes, 2=No)
    df = df[df["TOLDHI2"].isin([1.0, 2.0])]
    df["high_chol"] = (df["TOLDHI2"] == 1.0).astype(int)
    telemetry["filtering_stages"]["high_chol"] = prev_count - len(df)
    prev_count = len(df)

    # 4. Cholesterol Check: _CHOLCHK (1=Yes in past 5 yrs, 2/3=No)
    df = df[df["_CHOLCHK"].isin([1.0, 2.0, 3.0])]
    df["chol_check"] = (df["_CHOLCHK"] == 1.0).astype(int)
    telemetry["filtering_stages"]["chol_check"] = prev_count - len(df)
    prev_count = len(df)

    # 5. BMI: _BMI5 (implied 2 decimal places, divide by 100.0)
    df = df[df["_BMI5"].notna() & (df["_BMI5"] >= 1200) & (df["_BMI5"] <= 7000)]
    df["bmi"] = np.round(df["_BMI5"] / 100.0, 2)
    telemetry["filtering_stages"]["bmi_valid_range"] = prev_count - len(df)
    prev_count = len(df)

    # 6. Smoker: SMOKE100 (1=Yes, 2=No)
    df = df[df["SMOKE100"].isin([1.0, 2.0])]
    df["smoker"] = (df["SMOKE100"] == 1.0).astype(int)

    # 7. Stroke: CVDSTRK3 (1=Yes, 2=No)
    df = df[df["CVDSTRK3"].isin([1.0, 2.0])]
    df["stroke"] = (df["CVDSTRK3"] == 1.0).astype(int)

    # 8. Heart Disease or Myocardial Infarction: _MICHD (1=Yes, 2=No)
    df = df[df["_MICHD"].isin([1.0, 2.0])]
    df["heart_disease_or_attack"] = (df["_MICHD"] == 1.0).astype(int)

    # 9. Physical Activity: _TOTINDA (1=Yes, 2=No)
    df = df[df["_TOTINDA"].isin([1.0, 2.0])]
    df["phys_activity"] = (df["_TOTINDA"] == 1.0).astype(int)

    # 10. Fruits: _FRTLT1 (1=Yes, 2=No)
    df = df[df["_FRTLT1"].isin([1.0, 2.0])]
    df["fruits"] = (df["_FRTLT1"] == 1.0).astype(int)

    # 11. Veggies: _VEGLT1 (1=Yes, 2=No)
    df = df[df["_VEGLT1"].isin([1.0, 2.0])]
    df["veggies"] = (df["_VEGLT1"] == 1.0).astype(int)

    # 12. Heavy Alcohol: _RFDRHV5 (1=No, 2=Yes)
    df = df[df["_RFDRHV5"].isin([1.0, 2.0])]
    df["hvy_alcohol_consump"] = (df["_RFDRHV5"] == 2.0).astype(int)

    # 13. Health Insurance: HLTHPLN1 (1=Yes, 2=No)
    df = df[df["HLTHPLN1"].isin([1.0, 2.0])]
    df["any_healthcare"] = (df["HLTHPLN1"] == 1.0).astype(int)

    # 14. Doctor Cost Barrier: MEDCOST (1=Yes, 2=No)
    df = df[df["MEDCOST"].isin([1.0, 2.0])]
    df["no_doc_bc_cost"] = (df["MEDCOST"] == 1.0).astype(int)

    # 15. General Health: GENHLTH (1=Excellent ... 5=Poor)
    df = df[df["GENHLTH"].isin([1.0, 2.0, 3.0, 4.0, 5.0])]
    df["gen_hlth"] = df["GENHLTH"].astype(int)

    # 16. Mental Health: MENTHLTH (88=0, 1-30)
    df = df[df["MENTHLTH"].isin(list(range(1, 31)) + [88.0])]
    df["ment_hlth"] = df["MENTHLTH"].replace(88.0, 0.0).astype(int)

    # 17. Physical Health: PHYSHLTH (88=0, 1-30)
    df = df[df["PHYSHLTH"].isin(list(range(1, 31)) + [88.0])]
    df["phys_hlth"] = df["PHYSHLTH"].replace(88.0, 0.0).astype(int)

    # 18. Difficulty Walking: DIFFWALK (1=Yes, 2=No)
    df = df[df["DIFFWALK"].isin([1.0, 2.0])]
    df["diff_walk"] = (df["DIFFWALK"] == 1.0).astype(int)

    # 19. Sex: SEX (1=Male, 2=Female -> 1=Male, 0=Female)
    df = df[df["SEX"].isin([1.0, 2.0])]
    df["sex"] = (df["SEX"] == 1.0).astype(int)

    # 20. Age: _AGEG5YR (1=18-24 ... 13=80+)
    df = df[df["_AGEG5YR"].isin(list(range(1, 14)))]
    df["age"] = df["_AGEG5YR"].astype(int)

    # 21. Education: EDUCA (1-6)
    df = df[df["EDUCA"].isin(list(range(1, 7)))]
    df["education"] = df["EDUCA"].astype(int)

    # 22. Income: INCOME2 (1-8)
    df = df[df["INCOME2"].isin(list(range(1, 9)))]
    df["income"] = df["INCOME2"].astype(int)

    telemetry["filtering_stages"]["remaining_lifestyle_demographic_filters"] = prev_count - len(df)
    final_count = len(df)
    telemetry["final_clean_records"] = final_count
    telemetry["total_excluded_records"] = initial_count - final_count

    final_cols = [TARGET_COL] + RAW_INPUT_FEATURES
    df_clean = df[final_cols].copy()
    logger.info("Successfully cleaned %d records from %d raw records (excluded %d).",
                final_count, initial_count, initial_count - final_count)
    return df_clean, telemetry


def load_dataset(
    data_path: str,
    is_raw: Optional[bool] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified dataset loader supporting both:
    Path A: Bundled cleaned cohort (e.g. artifacts/cdc_brfss_diabetes_clean.csv)
    Path B: Raw CDC BRFSS dataset (e.g. archive/2015.csv)
    """
    resolved_path = Path(data_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{resolved_path}'. "
            f"If training from the bundled clean cohort, use 'artifacts/cdc_brfss_diabetes_clean.csv'. "
            f"If training from raw CDC survey data, place '2015.csv' in archive/ and specify --is-raw."
        )

    file_hash = compute_file_sha256(str(resolved_path))

    # Peek headers to auto-detect format if not explicitly specified
    sample_header = pd.read_csv(resolved_path, nrows=2)
    has_raw_target = "DIABETE3" in sample_header.columns
    has_clean_target = TARGET_COL in sample_header.columns

    if is_raw is True and not has_raw_target:
        raise ValueError(
            f"Dataset '{resolved_path}' was specified as raw (--is-raw), but does not contain "
            f"the raw CDC 'DIABETE3' column. Did you pass a cleaned dataset by mistake?"
        )
    if is_raw is False and not has_clean_target:
        raise ValueError(
            f"Dataset '{resolved_path}' was specified as cleaned, but does not contain "
            f"the required '{TARGET_COL}' column. Did you pass a raw CDC file without --is-raw?"
        )

    detected_as_raw = is_raw if is_raw is not None else has_raw_target

    metadata: Dict[str, Any] = {
        "source_path": str(resolved_path),
        "source_filename": resolved_path.name,
        "sha256_checksum": file_hash,
        "is_raw_source": detected_as_raw,
    }

    if detected_as_raw:
        logger.info("Reading raw CDC BRFSS survey file: %s", resolved_path)
        raw_df = pd.read_csv(resolved_path, usecols=BRFSS_RAW_COLS)
        df_clean, telemetry = clean_raw_brfss_dataframe(raw_df)
        metadata["telemetry"] = telemetry
    else:
        logger.info("Reading pre-cleaned BRFSS cohort: %s", resolved_path)
        df_clean = pd.read_csv(resolved_path)
        val_info = validate_cleaned_dataframe(df_clean)
        metadata["validation"] = val_info

    if sample_size and sample_size < len(df_clean):
        logger.info("Extracting stratified subsample of %d records (seed=%d)...", sample_size, random_state)
        from sklearn.model_selection import train_test_split
        df_clean, _ = train_test_split(
            df_clean,
            train_size=sample_size,
            stratify=df_clean[TARGET_COL],
            random_state=random_state
        )
        metadata["subsampled"] = True
        metadata["sample_size"] = sample_size
    else:
        metadata["subsampled"] = False
        metadata["sample_size"] = len(df_clean)

    df_clean = df_clean.reset_index(drop=True)
    metadata["final_record_count"] = len(df_clean)
    metadata["class_distribution"] = {
        CLASS_DEFINITIONS[int(k)]: int(v)
        for k, v in df_clean[TARGET_COL].value_counts().items()
    }

    return df_clean, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest and clean CDC BRFSS survey dataset.")
    parser.add_argument("--raw-path", type=str, default="archive/2015.csv", help="Path to raw CDC CSV.")
    parser.add_argument("--out-path", type=str, default="artifacts/cdc_brfss_diabetes_clean.csv", help="Output clean CSV path.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Stratified sample size to save.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    clean_df, meta = load_dataset(
        data_path=args.raw_path,
        is_raw=True,
        sample_size=args.sample_size,
        random_state=args.seed
    )

    out_file = PROJECT_ROOT / args.out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_file, index=False)
    print(f"Saved {len(clean_df)} validated records to {out_file}")
    print(f"Dataset SHA-256: {compute_file_sha256(str(out_file))}")
    print(f"Class breakdown: {meta['class_distribution']}")
