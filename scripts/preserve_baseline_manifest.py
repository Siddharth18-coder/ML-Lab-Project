"""
preserve_baseline_manifest.py - Preserve Baseline Metadata and Checksums
========================================================================
Locks in the historical 50,000-cohort baseline configuration, metrics, dataset
hashes, and environment versions before running new modeling experiments.
"""

import os
import json
import hashlib
import platform
import sys
import sklearn
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def generate_baseline_manifest():
    metadata_path = os.path.join(PROJECT_ROOT, "artifacts", "metadata.json")
    metrics_path = os.path.join(PROJECT_ROOT, "artifacts", "model_metrics.json")
    clean_csv_path = os.path.join(PROJECT_ROOT, "artifacts", "cdc_brfss_diabetes_clean.csv")
    model_pipeline_path = os.path.join(PROJECT_ROOT, "artifacts", "model_pipeline.joblib")
    explainer_path = os.path.join(PROJECT_ROOT, "artifacts", "explainer.joblib")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    manifest = {
        "experiment_id": "EXP-00-BASELINE-50K",
        "name": "Historical 50k Development Baseline",
        "description": "Historical development benchmark trained on the bundled 50,000-respondent stratified subsample.",
        "commit_hash": "d319972",
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__
        },
        "dataset_lineage": {
            "file_name": "cdc_brfss_diabetes_clean.csv",
            "file_sha256": compute_sha256(clean_csv_path),
            "record_count": 50000,
            "is_upstream_subsampled": True,
            "upstream_source": "CDC BRFSS 2015 (441,456 raw interviews, 251,117 post-cleaning eligible records)",
            "upstream_sampling_note": "This bundled cohort was pre-sampled to 50,000 complete-case records; it is not the full eligible dataset."
        },
        "artifacts_checksums": {
            "model_pipeline_sha256": compute_sha256(model_pipeline_path),
            "explainer_sha256": compute_sha256(explainer_path),
            "metadata_sha256": compute_sha256(metadata_path),
            "metrics_sha256": compute_sha256(metrics_path)
        },
        "hyperparameters": {
            "estimator": metadata.get("estimator", "RandomForestClassifier"),
            "params": metadata.get("hyperparameters", {}),
            "preprocessing": "PhysiologicalOutlierCapper(legacy_all_numerical) + RobustScaler"
        },
        "split_specification": {
            "split_type": "Stratified train_test_split (80/20)",
            "random_state": 42,
            "n_train": 40000,
            "n_test": 10000,
            "status": "historical_inspected_benchmark"
        },
        "benchmark_metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "multiclass_brier_score": metrics["multiclass_brier_score"],
            "calibration_log_loss": metrics["calibration_log_loss"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "per_class_recall": {
                cls: metrics["confusion_matrix"]["breakdown"][cls]["sensitivity_recall"]
                for cls in metrics["confusion_matrix"]["classes"]
            },
            "per_class_precision": {
                cls: metrics["confusion_matrix"]["breakdown"][cls]["precision"]
                for cls in metrics["confusion_matrix"]["classes"]
            }
        }
    }

    manifest_path = os.path.join(PROJECT_ROOT, "artifacts", "experiments", "baseline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully preserved baseline manifest to {manifest_path}")


if __name__ == "__main__":
    generate_baseline_manifest()
