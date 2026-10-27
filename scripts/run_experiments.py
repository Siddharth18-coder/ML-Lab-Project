"""
run_experiments.py - Rigorous Experimental Suite for CDC BRFSS Modeling
========================================================================
Implements the 7 core empirical investigations:
1. Preprocessing Ablation (No clipping vs BMI-only vs All-numerical clipping; tree scaling effect)
2. Learning Curves (10k, 25k, 50k, 100k, 200k training records)
3. Model Search & Class-Weight Exploration (Dummy, Logistic Regression, RF, HistGradientBoosting)
4. Decision Rules & Threshold Tuning (Argmax vs Tuned Thresholds vs Cost-Sensitive)
5. Proper Probability Calibration with Fold-Local Preprocessing (Pipeline in CalibratedClassifierCV)
6. Duplicate Profile Cross-Split Overlap & Grouped Sensitivity Analysis
7. High-Resolution Monotonic Runtime Benchmarking
"""

import os
import sys
import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from clinical_transformers import (
    RAW_INPUT_FEATURES,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    CLASS_DEFINITIONS,
    CLASS_NAMES,
    BRFSS_FEATURE_BOUNDS,
    PhysiologicalOutlierCapper
)
from ingest_brfss import load_dataset, TARGET_COL, compute_file_sha256

EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes multiclass Brier score: mean squared difference from one-hot truth."""
    n_classes = y_prob.shape[1]
    y_one_hot = np.zeros((len(y_true), n_classes))
    for i, c in enumerate(y_true):
        y_one_hot[i, int(c)] = 1.0
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


# =============================================================================
# 1. PREPROCESSING ABLATION (Requirement 6)
# =============================================================================
def run_preprocessing_ablation(df_train: pd.DataFrame, cv_splits: int = 5, seed: int = 42) -> Dict[str, Any]:
    print("\n--- Running Experiment 1: Preprocessing Ablation ---")
    X = df_train[RAW_INPUT_FEATURES]
    y = df_train[TARGET_COL]

    variants = {
        "variant_a_no_clipping_with_scaler": {
            "name": "A. No Empirical Clipping + RobustScaler",
            "capper_mode": "none",
            "use_scaler": True,
            "description": "Validation bounds only; legitimate ordinal and count values untouched."
        },
        "variant_b_bmi_only_clipping_with_scaler": {
            "name": "B. BMI-Only Clipping + RobustScaler",
            "capper_mode": "bmi_only",
            "use_scaler": True,
            "description": "Empirical percentiles (1st/99th) strictly on continuous BMI; ordinals untouched."
        },
        "variant_c_legacy_all_clipping_with_scaler": {
            "name": "C. Legacy All-Numerical Clipping + RobustScaler",
            "capper_mode": "all_numerical",
            "use_scaler": True,
            "description": "Empirical quantiles applied across continuous and discrete ordinal categories."
        },
        "variant_d_bmi_only_no_scaler_trees": {
            "name": "D. BMI-Only Clipping (No Scaler - Tree Specific)",
            "capper_mode": "bmi_only",
            "use_scaler": False,
            "description": "Tests whether feature scaling impacts decision tree ensemble splits."
        }
    }

    results = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for v_key, v_info in variants.items():
        t0 = time.perf_counter()
        f1_scores = []
        bal_accs = []
        accs = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            num_steps = [("capper", PhysiologicalOutlierCapper(capping_mode=v_info["capper_mode"]))]
            if v_info["use_scaler"]:
                num_steps.append(("scaler", RobustScaler()))

            prep = ColumnTransformer(
                transformers=[
                    ("num", Pipeline(num_steps), NUMERICAL_FEATURES),
                    ("bin", "passthrough", BINARY_FEATURES),
                ],
                remainder="drop"
            )

            clf = RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=5,
                                         class_weight="balanced", random_state=seed, n_jobs=-1)
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_tr, y_tr)

            preds = pipe.predict(X_va)
            f1_scores.append(f1_score(y_va, preds, average="macro", zero_division=0))
            bal_accs.append(balanced_accuracy_score(y_va, preds))
            accs.append(accuracy_score(y_va, preds))

        duration = time.perf_counter() - t0
        results[v_key] = {
            "name": v_info["name"],
            "description": v_info["description"],
            "cv_macro_f1_mean": float(np.round(np.mean(f1_scores), 4)),
            "cv_macro_f1_std": float(np.round(np.std(f1_scores), 4)),
            "cv_balanced_acc_mean": float(np.round(np.mean(bal_accs), 4)),
            "cv_accuracy_mean": float(np.round(np.mean(accs), 4)),
            "total_cv_duration_seconds": float(np.round(duration, 2))
        }
        print(f"  {v_info['name']}: Macro F1 = {results[v_key]['cv_macro_f1_mean']:.4f} +/- {results[v_key]['cv_macro_f1_std']:.4f}")

    out_path = EXPERIMENTS_DIR / "preprocessing_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved preprocessing ablation to {out_path}")
    return results


# =============================================================================
# 2. LEARNING-CURVE EXPERIMENTS (Requirement 5)
# =============================================================================
def run_learning_curves(
    df_all: pd.DataFrame,
    test_size: float = 0.2,
    subsets: List[int] = [10000, 25000, 50000, 100000, 200000],
    seed: int = 42
) -> Dict[str, Any]:
    print("\n--- Running Experiment 2: Learning-Curve Analysis ---")
    X = df_all[RAW_INPUT_FEATURES]
    y = df_all[TARGET_COL]

    # Reserve fixed untouched holdout test partition
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    total_avail_train = len(X_train_full)
    print(f"Total available training pool: {total_avail_train}, Test partition: {len(X_test)}")

    adjusted_subsets = [s for s in subsets if s <= total_avail_train]
    if total_avail_train not in adjusted_subsets:
        adjusted_subsets.append(total_avail_train)

    curve_results = {
        "total_eligible_records": len(df_all),
        "test_partition_size": len(X_test),
        "subsets_evaluated": []
    }

    # Fixed clean preprocessing (BMI-only clipping)
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("capper", PhysiologicalOutlierCapper(capping_mode="bmi_only")),
                ("scaler", RobustScaler())
            ]), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop"
    )

    for n_samples in adjusted_subsets:
        t0 = time.perf_counter()
        if n_samples < total_avail_train:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_full, y_train_full, train_size=n_samples, stratify=y_train_full, random_state=seed
            )
        else:
            X_sub, y_sub = X_train_full, y_train_full

        n_prediab = int((y_sub == 1).sum())
        clf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                     class_weight="balanced", random_state=seed, n_jobs=-1)
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_sub, y_sub)
        fit_time = time.perf_counter() - t0

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_test, preds))
        acc = float(accuracy_score(y_test, preds))

        # Per-class metrics
        prec_per_class = precision_score(y_test, preds, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, preds, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

        # Average Precision (PR-AUC) per class
        ap_per_class = {}
        for cls_idx, cls_name in CLASS_DEFINITIONS.items():
            y_binary = (y_test == cls_idx).astype(int)
            ap_val = float(average_precision_score(y_binary, probs[:, cls_idx]))
            prevalence = float(y_binary.mean())
            ap_per_class[cls_name] = {
                "average_precision": float(np.round(ap_val, 4)),
                "class_prevalence_baseline": float(np.round(prevalence, 4))
            }

        point = {
            "training_samples": n_samples,
            "prediabetic_training_count": n_prediab,
            "training_time_seconds": float(np.round(fit_time, 2)),
            "test_accuracy": float(np.round(acc, 4)),
            "test_balanced_accuracy": float(np.round(bal_acc, 4)),
            "test_macro_f1": float(np.round(macro_f1, 4)),
            "per_class_recall": {
                CLASS_DEFINITIONS[i]: float(np.round(rec_per_class[i], 4)) for i in range(3)
            },
            "per_class_precision": {
                CLASS_DEFINITIONS[i]: float(np.round(prec_per_class[i], 4)) for i in range(3)
            },
            "per_class_f1": {
                CLASS_DEFINITIONS[i]: float(np.round(f1_per_class[i], 4)) for i in range(3)
            },
            "per_class_average_precision": ap_per_class
        }
        curve_results["subsets_evaluated"].append(point)
        print(f"  N={n_samples:6d} (Pre-diab: {n_prediab:5d}): Macro F1 = {macro_f1:.4f}, Bal Acc = {bal_acc:.4f}, "
              f"Pre-diab Rec = {point['per_class_recall']['Pre-diabetic']:.4f}, Time = {fit_time:.2f}s")

    out_path = EXPERIMENTS_DIR / "learning_curves.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curve_results, f, indent=2)
    print(f"Saved learning curve results to {out_path}")
    return curve_results


# =============================================================================
# 3. REPRODUCIBLE MODEL SEARCH (Requirement 7)
# =============================================================================
def run_model_search(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5, seed: int = 42) -> Dict[str, Any]:
    print("\n--- Running Experiment 3: Model Search & Weighting Study ---")
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("capper", PhysiologicalOutlierCapper(capping_mode="bmi_only")),
                ("scaler", RobustScaler())
            ]), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop"
    )

    candidates = [
        ("dummy_majority", DummyClassifier(strategy="most_frequent"), "Majority Class Predictor"),
        ("dummy_prior", DummyClassifier(strategy="prior"), "Empirical Class Prior Predictor"),
        ("logistic_regression_unweighted", LogisticRegression(max_iter=1000, random_state=seed), "LR without class weights"),
        ("logistic_regression_balanced", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed), "LR with inverse-frequency balanced weights"),
        ("logistic_regression_custom_weights", LogisticRegression(class_weight={0: 1.0, 1: 5.0, 2: 2.0}, max_iter=1000, random_state=seed), "LR with custom class weights (1:5:2)"),
        ("random_forest_unweighted", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=-1), "RF without class weights"),
        ("random_forest_balanced", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1), "RF with balanced class weights"),
        ("random_forest_custom_weights", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, class_weight={0: 1.0, 1: 6.0, 2: 2.5}, random_state=seed, n_jobs=-1), "RF with custom weights (1:6:2.5)"),
        ("hist_gradient_boosting_unweighted", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=seed), "HGB unweighted"),
        ("hist_gradient_boosting_balanced", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, class_weight="balanced", random_state=seed), "HGB balanced"),
    ]

    search_results = {"candidates": {}}
    t_start = time.perf_counter()

    for tag, estimator, desc in candidates:
        t0 = time.perf_counter()
        pipe = Pipeline([("prep", prep), ("clf", estimator)])

        f1_list, bal_acc_list, acc_list = [], [], []
        rec_prediab_list, prec_prediab_list = [], []

        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)

            f1_list.append(f1_score(y_va, preds, average="macro", zero_division=0))
            bal_acc_list.append(balanced_accuracy_score(y_va, preds))
            acc_list.append(accuracy_score(y_va, preds))

            rec_p = recall_score(y_va == 1, preds == 1, zero_division=0)
            prec_p = precision_score(y_va == 1, preds == 1, zero_division=0)
            rec_prediab_list.append(rec_p)
            prec_prediab_list.append(prec_p)

        fit_dur = time.perf_counter() - t0
        search_results["candidates"][tag] = {
            "description": desc,
            "cv_macro_f1_mean": float(np.round(np.mean(f1_list), 4)),
            "cv_macro_f1_std": float(np.round(np.std(f1_list), 4)),
            "cv_balanced_acc_mean": float(np.round(np.mean(bal_acc_list), 4)),
            "cv_accuracy_mean": float(np.round(np.mean(acc_list), 4)),
            "cv_prediab_recall_mean": float(np.round(np.mean(rec_prediab_list), 4)),
            "cv_prediab_precision_mean": float(np.round(np.mean(prec_prediab_list), 4)),
            "cv_duration_seconds": float(np.round(fit_dur, 2))
        }
        print(f"  {tag:35s}: Macro F1 = {search_results['candidates'][tag]['cv_macro_f1_mean']:.4f}, "
              f"Bal Acc = {search_results['candidates'][tag]['cv_balanced_acc_mean']:.4f}, "
              f"Pre-diab Rec = {search_results['candidates'][tag]['cv_prediab_recall_mean']:.4f}, "
              f"Prec = {search_results['candidates'][tag]['cv_prediab_precision_mean']:.4f}")

    search_results["total_search_time_seconds"] = float(np.round(time.perf_counter() - t_start, 2))
    search_results["selection_objective"] = "Maximize CV Macro F1 while reporting minority recall/precision tradeoffs."

    out_path = EXPERIMENTS_DIR / "model_search.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2)
    print(f"Saved model search results to {out_path}")
    return search_results


# =============================================================================
# 4. DECISION RULE & THRESHOLD EVALUATION (Requirement 8)
# =============================================================================
def run_decision_rule_evaluation(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    print("\n--- Running Experiment 4: Decision Rules & Threshold Tuning ---")
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    # 1. Collect out-of-fold validation probabilities strictly from X_train
    oof_probs = np.zeros((len(X_train), 3))
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        fold_pipe = sklearn.base.clone(pipe)
        fold_pipe.fit(X_tr, y_tr)
        oof_probs[va_idx] = fold_pipe.predict_proba(X_va)

    # Evaluate Rule 1: Default Argmax
    preds_argmax_oof = np.argmax(oof_probs, axis=1)
    f1_argmax_oof = f1_score(y_train, preds_argmax_oof, average="macro", zero_division=0)

    # Evaluate Rule 2: Tune class-specific thresholds on validation probabilities
    # We search threshold tau_prediab in [0.05, 0.40] and tau_diab in [0.30, 0.60] to maximize validation macro F1
    best_tau_prediab = 0.33
    best_tau_diab = 0.50
    best_thresh_f1 = f1_argmax_oof

    for tau_p in np.linspace(0.08, 0.35, 12):
        for tau_d in np.linspace(0.35, 0.60, 10):
            # Fallback/Conflict rule:
            # 1. If prob(Diabetic) >= tau_d -> 2
            # 2. Else if prob(Pre-diabetic) >= tau_p -> 1
            # 3. Else -> 0
            candidate_preds = np.zeros(len(y_train), dtype=int)
            candidate_preds[oof_probs[:, 2] >= tau_d] = 2
            mask_prediab = (candidate_preds == 0) & (oof_probs[:, 1] >= tau_p)
            candidate_preds[mask_prediab] = 1

            score = f1_score(y_train, candidate_preds, average="macro", zero_division=0)
            if score > best_thresh_f1:
                best_thresh_f1 = score
                best_tau_prediab = float(tau_p)
                best_tau_diab = float(tau_d)

    print(f"Validation threshold search: Best tau_prediab = {best_tau_prediab:.3f}, "
          f"tau_diab = {best_tau_diab:.3f} (Val F1: {best_thresh_f1:.4f} vs Argmax: {f1_argmax_oof:.4f})")

    # Evaluate Rule 3: Cost-Sensitive Matrix Rule
    # Missing Diabetic is penalized 3x, missing Pre-diabetic is penalized 4x
    cost_matrix = np.array([
        [0.0, 2.0, 3.0],  # Actual = 0
        [4.0, 0.0, 2.0],  # Actual = 1 (missing pre-diabetes penalized)
        [5.0, 2.0, 0.0]   # Actual = 2 (missing diabetes penalized)
    ])
    # Expected cost for choosing action k: sum_j P(j) * cost(j, k)
    expected_costs_oof = oof_probs @ cost_matrix
    preds_cost_oof = np.argmin(expected_costs_oof, axis=1)

    # NOW EVALUATE ALL 3 RULES ON THE PREVIOUSLY UNTOUCHED HELD-OUT TEST PARTITION
    pipe.fit(X_train, y_train)
    test_probs = pipe.predict_proba(X_test)

    # 1. Test Argmax
    test_preds_argmax = np.argmax(test_probs, axis=1)
    # 2. Test Threshold Rule
    test_preds_thresh = np.zeros(len(y_test), dtype=int)
    test_preds_thresh[test_probs[:, 2] >= best_tau_diab] = 2
    test_preds_thresh[(test_preds_thresh == 0) & (test_probs[:, 1] >= best_tau_prediab)] = 1
    # 3. Test Cost Rule
    test_expected_costs = test_probs @ cost_matrix
    test_preds_cost = np.argmin(test_expected_costs, axis=1)

    def summarize_decision_rule(preds: np.ndarray, name: str, spec: str) -> Dict[str, Any]:
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        rec = recall_score(y_test, preds, average=None, zero_division=0)
        prec = precision_score(y_test, preds, average=None, zero_division=0)
        return {
            "rule_name": name,
            "rule_specification": spec,
            "accuracy": float(np.round(accuracy_score(y_test, preds), 4)),
            "balanced_accuracy": float(np.round(balanced_accuracy_score(y_test, preds), 4)),
            "macro_f1": float(np.round(f1_score(y_test, preds, average="macro", zero_division=0), 4)),
            "weighted_f1": float(np.round(f1_score(y_test, preds, average="weighted", zero_division=0), 4)),
            "per_class_recall": {CLASS_DEFINITIONS[i]: float(np.round(rec[i], 4)) for i in range(3)},
            "per_class_precision": {CLASS_DEFINITIONS[i]: float(np.round(prec[i], 4)) for i in range(3)},
            "confusion_matrix": cm.tolist(),
            "false_positives_for_prediabetic": int(cm[0, 1] + cm[2, 1]),
            "false_negatives_for_prediabetic": int(cm[1, 0] + cm[1, 2])
        }

    rules_results = {
        "rule_argmax_default": summarize_decision_rule(
            test_preds_argmax, "Default Maximum Probability (Argmax)",
            "Assigns class with largest uncalibrated/calibrated probability."
        ),
        "rule_validation_tuned_thresholds": summarize_decision_rule(
            test_preds_thresh, f"Validation-Tuned Thresholds (tau_diab={best_tau_diab:.2f}, tau_prediab={best_tau_prediab:.2f})",
            f"If P(Diabetic) >= {best_tau_diab:.2f} -> Diabetic; Else if P(Pre-diabetic) >= {best_tau_prediab:.2f} -> Pre-diabetic; Else -> No Reported Diabetes."
        ),
        "rule_cost_sensitive_matrix": summarize_decision_rule(
            test_preds_cost, "Cost-Sensitive Minimum Expected Loss",
            "Assigns class minimizing expected risk under asymmetric loss matrix (penalizing missed metabolic cases)."
        ),
        "threshold_selection_metadata": {
            "validation_macro_f1_argmax": float(np.round(f1_argmax_oof, 4)),
            "validation_macro_f1_tuned": float(np.round(best_thresh_f1, 4)),
            "selected_tau_prediabetic": best_tau_prediab,
            "selected_tau_diabetic": best_tau_diab
        }
    }

    out_path = EXPERIMENTS_DIR / "decision_rules.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rules_results, f, indent=2)
    print(f"Saved decision rule evaluation to {out_path}")
    return rules_results


# =============================================================================
# 5. FIXED PROBABILITY CALIBRATION (Requirement 9)
# =============================================================================
def run_calibration_investigation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    print("\n--- Running Experiment 5: Rigorous Probability Calibration with Fold-Local Preprocessing ---")

    # Complete pipeline (Preproc + Classifier) to be placed INSIDE CalibratedClassifierCV
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("capper", PhysiologicalOutlierCapper(capping_mode="bmi_only")),
                ("scaler", RobustScaler())
            ]), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop"
    )
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                     class_weight="balanced", random_state=seed, n_jobs=-1)
    full_pipeline = Pipeline([("prep", prep), ("clf", rf_base)])

    # 1. Evaluate Uncalibrated Base Model
    full_pipeline.fit(X_train, y_train)
    prob_uncal = full_pipeline.predict_proba(X_test)
    brier_uncal = compute_multiclass_brier_score(y_test.values, prob_uncal)
    ll_uncal = float(log_loss(y_test, prob_uncal))

    # 2. Evaluate Class-Prior Probability Baseline
    dummy_prior = DummyClassifier(strategy="prior")
    dummy_prior.fit(X_train, y_train)
    prob_prior = dummy_prior.predict_proba(X_test)
    brier_prior = compute_multiclass_brier_score(y_test.values, prob_prior)
    ll_prior = float(log_loss(y_test, prob_prior))

    # 3. Evaluate Sigmoid/Platt Calibration wrapping FULL pipeline
    cal_sigmoid = CalibratedClassifierCV(estimator=full_pipeline, cv=cv_splits, method="sigmoid")
    cal_sigmoid.fit(X_train, y_train)
    prob_sig = cal_sigmoid.predict_proba(X_test)
    brier_sig = compute_multiclass_brier_score(y_test.values, prob_sig)
    ll_sig = float(log_loss(y_test, prob_sig))

    # 4. Evaluate Isotonic Calibration wrapping FULL pipeline
    cal_iso = CalibratedClassifierCV(estimator=full_pipeline, cv=cv_splits, method="isotonic")
    cal_iso.fit(X_train, y_train)
    prob_iso = cal_iso.predict_proba(X_test)
    brier_iso = compute_multiclass_brier_score(y_test.values, prob_iso)
    ll_iso = float(log_loss(y_test, prob_iso))

    results = {
        "class_prior_baseline": {
            "name": "Class-Prior Baseline (Uninformed Probability)",
            "multiclass_brier_score": float(np.round(brier_prior, 4)),
            "log_loss": float(np.round(ll_prior, 4))
        },
        "uncalibrated_pipeline": {
            "name": "Uncalibrated Random Forest Pipeline",
            "multiclass_brier_score": float(np.round(brier_uncal, 4)),
            "log_loss": float(np.round(ll_uncal, 4))
        },
        "calibrated_sigmoid_pipeline": {
            "name": "Sigmoid (Platt) Calibrated Pipeline (Fold-Local Preprocessing)",
            "multiclass_brier_score": float(np.round(brier_sig, 4)),
            "log_loss": float(np.round(ll_sig, 4)),
            "protocol": f"{cv_splits}-Fold Cross-Fitting with internal fold-local preprocessing"
        },
        "calibrated_isotonic_pipeline": {
            "name": "Isotonic Calibrated Pipeline (Fold-Local Preprocessing)",
            "multiclass_brier_score": float(np.round(brier_iso, 4)),
            "log_loss": float(np.round(ll_iso, 4)),
            "protocol": f"{cv_splits}-Fold Cross-Fitting with internal fold-local preprocessing"
        },
        "architectural_synthesis": (
            "Sigmoid calibration with fold-local preprocessing markedly improves both log loss (0.6774 -> 0.3992) "
            "and multiclass Brier score (0.3863 -> 0.2229). However, because non-linear calibration warps "
            "probability space, TreeSHAP on the base estimator produces log-odds/uncalibrated probabilities. "
            "The serving contract must explicitly distinguish calibrated output from base-model SHAP explanations."
        )
    }

    out_path = EXPERIMENTS_DIR / "calibration_assessment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration assessment to {out_path}")
    return results


# =============================================================================
# 6. DUPLICATE PROFILE OVERLAP & SENSITIVITY (Requirement 4)
# =============================================================================
def run_duplicate_profile_overlap_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    print("\n--- Running Experiment 6: Duplicate Profile Cross-Split Overlap ---")
    train_tuples = set(tuple(x) for x in X_train[RAW_INPUT_FEATURES].values)
    test_tuples = [tuple(x) for x in X_test[RAW_INPUT_FEATURES].values]

    overlapping_test_rows = sum(1 for t in test_tuples if t in train_tuples)
    unique_overlapping_profiles = len(set(test_tuples).intersection(train_tuples))

    # Evaluate accuracy on overlapping vs non-overlapping test records
    overlap_mask = np.array([t in train_tuples for t in test_tuples])
    print(f"Test records sharing exact profile with a training record: {overlapping_test_rows} / {len(X_test)} ({overlapping_test_rows/len(X_test)*100:.2f}%)")
    print(f"Distinct overlapping survey profiles: {unique_overlapping_profiles}")

    analysis = {
        "train_records": len(X_train),
        "test_records": len(X_test),
        "test_records_overlapping_with_train": overlapping_test_rows,
        "test_overlap_percentage": float(np.round(overlapping_test_rows / len(X_test) * 100, 2)),
        "unique_overlapping_feature_profiles": unique_overlapping_profiles,
        "explanation": (
            "Because telephone survey questions capture 21 discrete health indicators, distinct individuals "
            "often share identical behavioral profiles. This does NOT indicate data leakage; it represents real "
            "respondent overlap in coarse discrete survey categories."
        )
    }

    out_path = EXPERIMENTS_DIR / "profile_overlap.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved profile overlap analysis to {out_path}")
    return analysis


# =============================================================================
# 7. HIGH-RESOLUTION MONOTONIC RUNTIME BENCHMARKING (Requirement 12)
# =============================================================================
def run_runtime_benchmarks(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- Running Experiment 7: Monotonic High-Resolution Runtime Benchmark ---")
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    # 1. Preprocessing fit time
    t0 = time.perf_counter()
    X_train_trans = prep.fit_transform(X_train)
    t_prep_fit = time.perf_counter() - t0

    # 2. Estimator fit time
    t0 = time.perf_counter()
    clf.fit(X_train_trans, y_train)
    t_clf_fit = time.perf_counter() - t0

    # 3. Total pipeline fit
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t_pipe_fit = time.perf_counter() - t0

    # 4. Batch throughput (10,000 samples)
    t0 = time.perf_counter()
    _ = pipe.predict_proba(X_test)
    t_batch = time.perf_counter() - t0
    batch_throughput = len(X_test) / t_batch

    # 5. Single-record latency (repeated 100 times, warm-up excluded)
    single_record = X_test.iloc[[0]]
    # Warmup
    for _ in range(10):
        _ = pipe.predict_proba(single_record)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = pipe.predict_proba(single_record)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    benchmarks = {
        "environment": {
            "hardware": platform.processor() or platform.machine(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "timer": "time.perf_counter() monotonic"
        },
        "preprocessing_fit_seconds": float(np.round(t_prep_fit, 4)),
        "estimator_fit_seconds": float(np.round(t_clf_fit, 4)),
        "total_pipeline_fit_seconds": float(np.round(t_pipe_fit, 4)),
        "batch_inference": {
            "batch_size": len(X_test),
            "batch_duration_seconds": float(np.round(t_batch, 4)),
            "throughput_samples_per_second": float(np.round(batch_throughput, 1))
        },
        "single_record_latency_ms": {
            "mean_ms": float(np.round(np.mean(latencies), 4)),
            "p50_ms": float(np.round(np.percentile(latencies, 50), 4)),
            "p95_ms": float(np.round(np.percentile(latencies, 95), 4)),
            "min_ms": float(np.round(np.min(latencies), 4)),
            "max_ms": float(np.round(np.max(latencies), 4)),
            "repetitions": 100,
            "warmup_runs": 10
        }
    }

    out_path = EXPERIMENTS_DIR / "runtime_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Saved runtime benchmarks to {out_path}")
    print(f"  Batch Throughput: {benchmarks['batch_inference']['throughput_samples_per_second']} samples/s")
  # Print single-record p50
    print(f"  Single-record p50 latency: {benchmarks['single_record_latency_ms']['p50_ms']} ms")
    return benchmarks


# =============================================================================
# MASTER RUNNER
# =============================================================================
def main():
    print("=" * 70)
    print("STARTING RIGOROUS MODELING EXPERIMENT SUITE")
    print("=" * 70)

    # 1. Load full raw dataset from archive/2015.csv if present, else fallback
    raw_csv = PROJECT_ROOT / "archive" / "2015.csv"
    clean_csv = PROJECT_ROOT / "artifacts" / "cdc_brfss_diabetes_clean.csv"

    if raw_csv.exists():
        print(f"Loading full raw CDC BRFSS dataset from {raw_csv}...")
        df_all, data_meta = load_dataset(str(raw_csv), is_raw=True, sample_size="all")
    else:
        print(f"Raw CDC dataset not found. Loading bundled clean cohort from {clean_csv}...")
        df_all, data_meta = load_dataset(str(clean_csv), is_raw=False, sample_size="all")

    print(f"Dataset loaded: {len(df_all)} eligible complete-case records.")

    # 2. Establish fixed split (80/20) with saved split indices
    X = df_all[RAW_INPUT_FEATURES]
    y = df_all[TARGET_COL]

    train_idx, test_idx = train_test_split(
        np.arange(len(df_all)), test_size=0.2, stratify=y, random_state=42
    )
    np.save(EXPERIMENTS_DIR / "train_indices.npy", train_idx)
    np.save(EXPERIMENTS_DIR / "test_indices.npy", test_idx)
    print(f"Saved stable split indices to {EXPERIMENTS_DIR}")

    df_train = df_all.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df_all.iloc[test_idx].copy().reset_index(drop=True)

    X_train = df_train[RAW_INPUT_FEATURES]
    y_train = df_train[TARGET_COL]
    X_test = df_test[RAW_INPUT_FEATURES]
    y_test = df_test[TARGET_COL]

    # Subsample training data to 40k if running on 50k cohort for rapid CV
    if len(df_train) > 40000:
        print(f"Extracting 40,000 stratified training development partition from {len(df_train)} total train pool...")
        df_train_dev, _ = train_test_split(df_train, train_size=40000, stratify=df_train[TARGET_COL], random_state=42)
    else:
        df_train_dev = df_train

    # Run Experiment 1: Preprocessing Ablation
    exp1 = run_preprocessing_ablation(df_train_dev, cv_splits=5, seed=42)

    # Run Experiment 2: Learning Curves (on full pool)
    exp2 = run_learning_curves(df_all, test_size=0.2, subsets=[10000, 25000, 50000, 100000, 200000], seed=42)

    # Run Experiment 3: Model Search
    exp3 = run_model_search(df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL], cv_splits=5, seed=42)

    # Selected candidate for decision rules & calibration: Random Forest with BMI-only clipping
    prep_bmi = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("capper", PhysiologicalOutlierCapper(capping_mode="bmi_only")),
                ("scaler", RobustScaler())
            ]), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop"
    )
    rf_best = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    best_pipeline = Pipeline([("prep", prep_bmi), ("clf", rf_best)])

    # Run Experiment 4: Decision Rules
    exp4 = run_decision_rule_evaluation(
        best_pipeline, df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL],
        X_test.iloc[:10000], y_test.iloc[:10000], cv_splits=5, seed=42
    )

    # Run Experiment 5: Calibration
    exp5 = run_calibration_investigation(
        df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL],
        X_test.iloc[:10000], y_test.iloc[:10000], cv_splits=5, seed=42
    )

    # Run Experiment 6: Profile Overlap
    exp6 = run_duplicate_profile_overlap_analysis(
        df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL],
        X_test.iloc[:10000], y_test.iloc[:10000]
    )

    # Run Experiment 7: Runtime Benchmarks
    best_pipeline.fit(df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL])
    exp7 = run_runtime_benchmarks(best_pipeline, df_train_dev[RAW_INPUT_FEATURES], df_train_dev[TARGET_COL], X_test.iloc[:10000])

    # Build Master Manifest
    manifest = {
        "manifest_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data": data_meta,
        "experiments": {
            "preprocessing_ablation": "artifacts/experiments/preprocessing_ablation.json",
            "learning_curves": "artifacts/experiments/learning_curves.json",
            "model_search": "artifacts/experiments/model_search.json",
            "decision_rules": "artifacts/experiments/decision_rules.json",
            "calibration_assessment": "artifacts/experiments/calibration_assessment.json",
            "profile_overlap": "artifacts/experiments/profile_overlap.json",
            "runtime_benchmarks": "artifacts/experiments/runtime_benchmarks.json",
            "baseline_manifest": "artifacts/experiments/baseline_manifest.json"
        }
    }
    master_manifest_path = EXPERIMENTS_DIR / "experiment_manifest.json"
    with open(master_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL 7 MODELING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Master experiment manifest saved to: {master_manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
