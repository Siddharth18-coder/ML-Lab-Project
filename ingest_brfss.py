"""
ingest_brfss.py - CDC BRFSS Real-World Dataset Ingestion and Standardization
=============================================================================
Data Provenance:
- Source: U.S. Centers for Disease Control and Prevention (CDC)
- Survey: Behavioral Risk Factor Surveillance System (BRFSS) 2015
- License: Public Domain (U.S. Federal Government work)
- Scope: 441,456 adult survey respondents across all 50 states

Target Variable Definition (DIABETE3):
- 0 = Healthy (No diabetes, DIABETE3 == 3.0)
- 1 = Pre-diabetic (Told pre-diabetes or borderline diabetes, DIABETE3 == 4.0)
- 2 = Diabetic (Told diabetes, DIABETE3 == 1.0)
- Excluded: Gestational diabetes (2.0), Don't know / Not sure (7.0), Refused (9.0), Missing (NaN).

21 Standardized Features:
1.  high_bp: High blood pressure diagnosis (1=Yes, 0=No)
2.  high_chol: High cholesterol diagnosis (1=Yes, 0=No)
3.  chol_check: Cholesterol check in past 5 years (1=Yes, 0=No)
4.  bmi: Body Mass Index in kg/m² (_BMI5 / 100.0)
5.  smoker: Smoked at least 100 cigarettes (1=Yes, 0=No)
6.  stroke: History of stroke (1=Yes, 0=No)
7.  heart_disease_or_attack: Coronary heart disease or myocardial infarction (1=Yes, 0=No)
8.  phys_activity: Any physical activity in past 30 days (1=Yes, 0=No)
9.  fruits: Consumes fruit >= 1 time per day (1=Yes, 0=No)
10. veggies: Consumes vegetables >= 1 time per day (1=Yes, 0=No)
11. hvy_alcohol_consump: Heavy alcohol consumption (1=Yes, 0=No)
12. any_healthcare: Any health insurance / coverage (1=Yes, 0=No)
13. no_doc_bc_cost: Could not see doctor due to cost in past 12 mos (1=Yes, 0=No)
14. gen_hlth: General health status (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)
15. ment_hlth: Days of poor mental health in past 30 days (0 - 30)
16. phys_hlth: Days of poor physical health in past 30 days (0 - 30)
17. diff_walk: Serious difficulty walking or climbing stairs (1=Yes, 0=No)
18. sex: Biological sex (1=Male, 0=Female)
19. age: 13-level age category (1=18-24 ... 13=80+)
20. education: Education level (1=Never/Kindergarten ... 6=College grad)
21. income: Income bracket (1=<10k ... 8=>=75k)
"""

import os
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("BRFSS_Ingestion")

BRFSS_RAW_COLS = [
    "DIABETE3", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5",
    "SMOKE100", "CVDSTRK3", "_MICHD", "_TOTINDA", "_FRTLT1",
    "_VEGLT1", "_RFDRHV5", "HLTHPLN1", "MEDCOST", "GENHLTH",
    "MENTHLTH", "PHYSHLTH", "DIFFWALK", "SEX", "_AGEG5YR",
    "EDUCA", "INCOME2"
]

BRFSS_CLEAN_FEATURES = [
    "high_bp", "high_chol", "chol_check", "bmi", "smoker",
    "stroke", "heart_disease_or_attack", "phys_activity", "fruits",
    "veggies", "hvy_alcohol_consump", "any_healthcare", "no_doc_bc_cost",
    "gen_hlth", "ment_hlth", "phys_hlth", "diff_walk", "sex", "age",
    "education", "income"
]

TARGET_COL = "diabetes_stage"


def clean_brfss_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw CDC BRFSS dataframe into 21 standardized predictors and 3-class target."""
    logger.info("Raw records received: %d", len(raw_df))

    # 1. Target DIABETE3: 3.0 -> 0 (Healthy), 4.0 -> 1 (Pre-diabetic), 1.0 -> 2 (Diabetic)
    df = raw_df[raw_df["DIABETE3"].isin([1.0, 3.0, 4.0])].copy()
    target_map = {3.0: 0, 4.0: 1, 1.0: 2}
    df[TARGET_COL] = df["DIABETE3"].map(target_map)

    # 2. High Blood Pressure: _RFHYPE5 (1=No, 2=Yes)
    df = df[df["_RFHYPE5"].isin([1.0, 2.0])]
    df["high_bp"] = (df["_RFHYPE5"] == 2.0).astype(int)

    # 3. High Cholesterol: TOLDHI2 (1=Yes, 2=No)
    df = df[df["TOLDHI2"].isin([1.0, 2.0])]
    df["high_chol"] = (df["TOLDHI2"] == 1.0).astype(int)

    # 4. Cholesterol Check: _CHOLCHK (1=Yes in past 5 yrs, 2/3=No)
    df = df[df["_CHOLCHK"].isin([1.0, 2.0, 3.0])]
    df["chol_check"] = (df["_CHOLCHK"] == 1.0).astype(int)

    # 5. BMI: _BMI5 (implied 2 decimal places, divide by 100.0)
    df = df[df["_BMI5"].notna() & (df["_BMI5"] >= 1200) & (df["_BMI5"] <= 7000)]
    df["bmi"] = np.round(df["_BMI5"] / 100.0, 2)

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

    final_cols = [TARGET_COL] + BRFSS_CLEAN_FEATURES
    df_clean = df[final_cols].copy()
    logger.info("Successfully cleaned %d records across %d features.", len(df_clean), len(BRFSS_CLEAN_FEATURES))
    return df_clean


def load_brfss_cohort(
    raw_csv_path: str = "archive/2015.csv",
    sample_size: Optional[int] = 50000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Ingests and cleans CDC BRFSS dataset.
    If sample_size is provided, returns a stratified sample across the 3 classes.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"CDC BRFSS raw file '{raw_csv_path}' not found.")

    logger.info("Reading raw CDC BRFSS survey file: %s", raw_csv_path)
    raw_df = pd.read_csv(raw_csv_path, usecols=BRFSS_RAW_COLS)
    clean_df = clean_brfss_dataframe(raw_df)

    if sample_size and sample_size < len(clean_df):
        logger.info("Performing stratified sample of %d records...", sample_size)
        from sklearn.model_selection import train_test_split
        sampled_df, _ = train_test_split(
            clean_df,
            train_size=sample_size,
            stratify=clean_df[TARGET_COL],
            random_state=random_state
        )
        return sampled_df.reset_index(drop=True)

    return clean_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_brfss_cohort(sample_size=50000)
    out_path = "artifacts/cdc_brfss_diabetes_clean.csv"
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} cleaned records to {out_path}")
    print("Class distribution:")
    print(df[TARGET_COL].value_counts(normalize=True))
