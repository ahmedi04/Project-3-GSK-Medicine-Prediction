#!/usr/bin/env python3
"""Scenario 3 baseline models using the prepared validation input and saved splits."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from scenario3_preprocessing import build_scenario3_preprocessor


BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")
METRICS_FILE = Path("outputs/scenario3_baseline_metrics.csv")
REPORT_FILE = Path("outputs/scenario3_baseline_report.txt")
CM_FILE = Path("outputs/scenario3_validation_confusion_matrices.png")
PRED_FILE = Path("outputs/scenario3_validation_predictions.parquet")

DT_MODEL_FILE = Path("models/scenario3_decision_tree_pipeline.joblib")
RF_MODEL_FILE = Path("models/scenario3_random_forest_pipeline.joblib")
XGB_MODEL_FILE = Path("models/scenario3_xgboost_pipeline.joblib")

TARGET = "treatment_outcome"
EXCLUDED_COLUMNS = ["patient_id", TARGET, "admission_date", "adverse_event", "readmission_30d"]


def split_summary(name, y):
    total = len(y)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    pos_pct = (positive / total) * 100 if total else 0.0
    neg_pct = (negative / total) * 100 if total else 0.0
    return f"{name}: rows={total:,}, negative(0)={negative:,} ({neg_pct:.3f}%), positive(1)={positive:,} ({pos_pct:.3f}%)"


def evaluate_metrics(y_true, probabilities, threshold=0.50):
    preds = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]),
    }


def make_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", build_scenario3_preprocessor()),
            ("model", model),
        ]
    )


def validate_inputs(base_df, split_df):
    """Validate the prepared dataset and split assignments before any model training."""
    print("Progress: Validating Scenario 3 inputs and split assignments...")
    if base_df.shape != (1_050_000, 31):
        raise SystemExit(f"Scenario 3 base shape must be (1050000, 31); found {base_df.shape}.")

    labelled_count = int(base_df[TARGET].notna().sum())
    missing_count = int(base_df[TARGET].isna().sum())
    if labelled_count != 985_876 or missing_count != 64_124:
        raise SystemExit(
            f"Scenario 3 target counts do not match expectations: labelled={labelled_count:,}, missing={missing_count:,}."
        )

    if base_df["patient_id"].isna().any() or not base_df["patient_id"].is_unique:
        raise SystemExit("Scenario 3 base patient_id values must be non-missing and unique.")

    if split_df["patient_id"].isna().any() or not split_df["patient_id"].is_unique:
        raise SystemExit("Scenario 3 split assignments patient_id values must be non-missing and unique.")

    if not set(split_df["split"]).issubset({"train", "validation", "test"}):
        raise SystemExit("Scenario 3 split assignments contain invalid split labels.")

    expected_counts = {"train": 591_525, "validation": 197_175, "test": 197_176}
    actual_counts = split_df["split"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise SystemExit(f"Scenario 3 split counts mismatch. Expected {expected_counts}, found {actual_counts}.")


def join_assignments(base_df, split_df):
    """Join split assignments to the base dataset by patient_id, preserving assignment order."""
    ordered_split = split_df.reset_index(drop=True).copy()
    ordered_split["assignment_order"] = np.arange(len(ordered_split), dtype=np.int64)

    merged = ordered_split.merge(
        base_df,
        on="patient_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.sort_values("assignment_order", kind="stable").reset_index(drop=True)

    # Cover only labelled rows for modelling.
    labelled = merged[merged[TARGET].notna()].copy()
    if len(labelled) != 985_876:
        raise SystemExit(
            f"Labelled merged rows mismatch: expected 985,876, found {len(labelled):,}."
        )

    # Verify the group coverage and overlap properties.
    train_ids = set(labelled.loc[labelled["split"] == "train", "patient_id"])
    val_ids = set(labelled.loc[labelled["split"] == "validation", "patient_id"])
    test_ids = set(labelled.loc[labelled["split"] == "test", "patient_id"])
    overlaps = {
        "train_validation": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(val_ids & test_ids),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Scenario 3 split overlaps detected: {overlaps}")

    if len(train_ids) != 591_525 or len(val_ids) != 197_175 or len(test_ids) != 197_176:
        raise SystemExit(
            "Scenario 3 split membership counts do not match the saved Step 09 recipe."
        )

    labelled_patient_ids = set(labelled["patient_id"])
    assigned_patient_ids = set(split_df["patient_id"])
    if labelled_patient_ids != assigned_patient_ids:
        raise SystemExit("Scenario 3 split assignments do not cover the labelled patients exactly once.")

    # Validate binary targets.
    y_numeric = pd.to_numeric(labelled[TARGET], errors="coerce")
    if y_numeric.isna().any() or not set(y_numeric.unique().tolist()).issubset({0, 1}):
        raise SystemExit("Scenario 3 labelled targets contain invalid binary values.")
    labelled[TARGET] = y_numeric.astype(int)

    return labelled


def build_report_lines(base_df, labelled_df, train_df, val_df, test_df, metrics_rows, reference_row, feature_notes):
    lines = []
    lines.append("Scenario 3 Baseline Report")
    lines.append("==========================")
    lines.append(f"Source file: {BASE_FILE}")
    lines.append(f"Split file: {SPLIT_FILE}")
    lines.append(f"Base shape: {base_df.shape}")
    lines.append(f"Labelled rows: {len(labelled_df):,}")
    lines.append(f"Missing-target rows: {int(base_df[TARGET].isna().sum()):,}")
    lines.append("")
    lines.append("Split counts and class distributions:")
    lines.append(f"- {split_summary('Train', train_df[TARGET])}")
    lines.append(f"- {split_summary('Validation', val_df[TARGET])}")
    lines.append(f"- {split_summary('Test', test_df[TARGET])}")
    lines.append("- Test membership matches the earlier Step 09 recipe and is retained for later evaluation.")
    lines.append("")
    lines.append("Predictors and preprocessing:")
    lines.append("- Excluded columns: patient_id, treatment_outcome, admission_date, adverse_event, readmission_30d")
    lines.append("- Base predictors before feature engineering: 26")
    lines.append("- Predictors after feature engineering: 33")
    lines.append(f"- Encoded feature count after one-hot encoding: {feature_notes['encoded_feature_count']}")
    lines.append("- Categorical missing values were imputed with training modes; numeric missing values with training medians.")
    lines.append("- Step 02 IQR clipping was fit on training data only for age, bmi, dosage_mg, hemoglobin, creatinine, alt_enzyme, and ast_enzyme.")
    lines.append("- Step 03 features were recreated exactly: kidney_stage, bmi_category, age_group, liver_risk, polypharmacy, elderly_high_dose, de_ritis_ratio.")
    lines.append(f"- Training-set dosage_mg median used for elderly_high_dose: {feature_notes['dosage_median']:.6f}")
    lines.append(f"- Training-set de_ritis_ratio median before capping: {feature_notes['ratio_median']:.6f}")
    lines.append(f"- Training-set de_ritis_ratio IQR lower/upper limits: {feature_notes['ratio_lower']:.6f} / {feature_notes['ratio_upper']:.6f}")
    lines.append("")
    lines.append("Model settings:")
    for model_name, settings in feature_notes["model_settings"].items():
        lines.append(f"- {model_name}: {settings}")
    lines.append("- Cutoff used for this baseline: 0.50")
    lines.append("")
    lines.append("Always-predict-0 reference baseline (validation):")
    lines.append(
        f"- accuracy={reference_row['validation_accuracy']:.4f}, precision={reference_row['validation_precision']:.4f}, "
        f"recall={reference_row['validation_recall']:.4f}, f1={reference_row['validation_f1']:.4f}, "
        f"roc_auc={reference_row['validation_roc_auc']:.4f}"
    )
    lines.append("")
    lines.append("Validation metrics by model:")
    for row in metrics_rows:
        lines.append(f"- {row['model']}")
        lines.append(
            f"  Train: accuracy={row['train_accuracy']:.4f}, precision={row['train_precision']:.4f}, "
            f"recall={row['train_recall']:.4f}, f1={row['train_f1']:.4f}, roc_auc={row['train_roc_auc']:.4f}"
        )
        lines.append(
            f"  Validation: accuracy={row['validation_accuracy']:.4f}, precision={row['validation_precision']:.4f}, "
            f"recall={row['validation_recall']:.4f}, f1={row['validation_f1']:.4f}, roc_auc={row['validation_roc_auc']:.4f}"
        )
        lines.append(
            f"  Train minus validation: f1={row['train_minus_validation_f1']:.4f}, roc_auc={row['train_minus_validation_auc']:.4f}"
        )
        lines.append(
            f"  Validation confusion matrix [tn, fp, fn, tp]: {row['validation_confusion_matrix']}"
        )
    lines.append("")
    lines.append("Validation confusion matrices are saved separately in the validation chart.")
    lines.append(
        "This stage uses only training and validation predictions; test evaluation is reserved for later experiments."
    )
    return lines


def main():
    print("Progress: Loading Scenario 3 prepared inputs...")
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {BASE_FILE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split assignment file: {SPLIT_FILE}")

    base_df = pd.read_parquet(BASE_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    validate_inputs(base_df, split_df)

    labelled_df = join_assignments(base_df, split_df)

    train_df = labelled_df[labelled_df["split"] == "train"].copy()
    val_df = labelled_df[labelled_df["split"] == "validation"].copy()
    test_df = labelled_df[labelled_df["split"] == "test"].copy()

    predictor_columns = [c for c in base_df.columns if c not in EXCLUDED_COLUMNS]
    if len(predictor_columns) != 26:
        raise SystemExit(f"Expected 26 base predictors, found {len(predictor_columns)}.")

    X_train = train_df[predictor_columns].copy()
    y_train = train_df[TARGET].astype(int).copy()
    X_val = val_df[predictor_columns].copy()
    y_val = val_df[TARGET].astype(int).copy()

    print("Progress: Building reusable Scenario 3 pipelines...")
    model_specs = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.3,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}
    metrics_rows = []
    validation_probs = {}
    validation_preds = pd.DataFrame({
        "patient_id": val_df["patient_id"].values,
        "actual_target": y_val.values,
    })

    for model_name, model in model_specs.items():
        print(f"Progress: Fitting {model_name} on training data only...")
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        pipelines[model_name] = pipeline

        train_prob = pipeline.predict_proba(X_train)[:, 1]
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        validation_probs[model_name] = val_prob
        validation_preds[f"{model_name.lower()}_probability"] = val_prob

        train_metrics = evaluate_metrics(y_train, train_prob, threshold=0.50)
        val_metrics = evaluate_metrics(y_val, val_prob, threshold=0.50)

        metrics_rows.append({
            "model": model_name,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "train_tn": int(train_metrics["confusion_matrix"][0, 0]),
            "train_fp": int(train_metrics["confusion_matrix"][0, 1]),
            "train_fn": int(train_metrics["confusion_matrix"][1, 0]),
            "train_tp": int(train_metrics["confusion_matrix"][1, 1]),
            "validation_tn": int(val_metrics["confusion_matrix"][0, 0]),
            "validation_fp": int(val_metrics["confusion_matrix"][0, 1]),
            "validation_fn": int(val_metrics["confusion_matrix"][1, 0]),
            "validation_tp": int(val_metrics["confusion_matrix"][1, 1]),
            "train_minus_validation_f1": train_metrics["f1"] - val_metrics["f1"],
            "train_minus_validation_auc": train_metrics["roc_auc"] - val_metrics["roc_auc"],
            "validation_confusion_matrix": val_metrics["confusion_matrix"].ravel().tolist(),
        })

    print("Progress: Creating always-predict-0 reference baseline...")
    zero_train_prob = np.zeros(len(y_train), dtype="float64")
    zero_val_prob = np.zeros(len(y_val), dtype="float64")
    zero_train_metrics = evaluate_metrics(y_train, zero_train_prob, threshold=0.50)
    zero_val_metrics = evaluate_metrics(y_val, zero_val_prob, threshold=0.50)
    reference_row = {
        "validation_accuracy": zero_val_metrics["accuracy"],
        "validation_precision": zero_val_metrics["precision"],
        "validation_recall": zero_val_metrics["recall"],
        "validation_f1": zero_val_metrics["f1"],
        "validation_roc_auc": zero_val_metrics["roc_auc"],
    }

    baseline_row = {
        "model": "AlwaysZero",
        "train_accuracy": zero_train_metrics["accuracy"],
        "train_precision": zero_train_metrics["precision"],
        "train_recall": zero_train_metrics["recall"],
        "train_f1": zero_train_metrics["f1"],
        "train_roc_auc": zero_train_metrics["roc_auc"],
        "validation_accuracy": zero_val_metrics["accuracy"],
        "validation_precision": zero_val_metrics["precision"],
        "validation_recall": zero_val_metrics["recall"],
        "validation_f1": zero_val_metrics["f1"],
        "validation_roc_auc": zero_val_metrics["roc_auc"],
        "train_tn": int(zero_train_metrics["confusion_matrix"][0, 0]),
        "train_fp": int(zero_train_metrics["confusion_matrix"][0, 1]),
        "train_fn": int(zero_train_metrics["confusion_matrix"][1, 0]),
        "train_tp": int(zero_train_metrics["confusion_matrix"][1, 1]),
        "validation_tn": int(zero_val_metrics["confusion_matrix"][0, 0]),
        "validation_fp": int(zero_val_metrics["confusion_matrix"][0, 1]),
        "validation_fn": int(zero_val_metrics["confusion_matrix"][1, 0]),
        "validation_tp": int(zero_val_metrics["confusion_matrix"][1, 1]),
        "train_minus_validation_f1": zero_train_metrics["f1"] - zero_val_metrics["f1"],
        "train_minus_validation_auc": zero_train_metrics["roc_auc"] - zero_val_metrics["roc_auc"],
        "validation_confusion_matrix": zero_val_metrics["confusion_matrix"].ravel().tolist(),
    }

    print("Progress: Saving validation predictions...")
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    validation_preds.to_parquet(PRED_FILE, index=False)

    print("Progress: Saving validation confusion matrices figure...")
    cm_values = [row["validation_confusion_matrix"] for row in metrics_rows]
    max_count = max(max(cm) for cm in cm_values)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = ["Ineffective (0)", "Effective (1)"]
    for ax, row in zip(axes, metrics_rows):
        cm = np.array(row["validation_confusion_matrix"]).reshape(2, 2)
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            square=True,
            vmin=0,
            vmax=max_count,
            ax=ax,
        )
        ax.set_title(row["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)
    CM_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FILE, dpi=150)
    plt.close(fig)

    print("Progress: Saving fitted pipelines...")
    DT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines["DecisionTree"], DT_MODEL_FILE)
    joblib.dump(pipelines["RandomForest"], RF_MODEL_FILE)
    joblib.dump(pipelines["XGBoost"], XGB_MODEL_FILE)

    print("Progress: Saving metrics CSV...")
    metrics_df = pd.DataFrame([baseline_row, *metrics_rows])
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_FILE, index=False)

    feature_notes = {
        "encoded_feature_count": pipelines["XGBoost"].named_steps["preprocessor"].encoded_feature_count_,
        "dosage_median": pipelines["XGBoost"].named_steps["preprocessor"].dosage_median_,
        "ratio_median": pipelines["XGBoost"].named_steps["preprocessor"].ratio_median_,
        "ratio_lower": pipelines["XGBoost"].named_steps["preprocessor"].ratio_lower_limit_,
        "ratio_upper": pipelines["XGBoost"].named_steps["preprocessor"].ratio_upper_limit_,
        "model_settings": {
            "DecisionTree": "max_depth=8, min_samples_split=20, min_samples_leaf=10, class_weight=balanced, random_state=42",
            "RandomForest": "n_estimators=200, max_depth=12, min_samples_split=15, min_samples_leaf=5, max_features=sqrt, class_weight=balanced, n_jobs=-1, random_state=42",
            "XGBoost": "n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.3, objective=binary:logistic, eval_metric=auc, tree_method=hist, random_state=42, n_jobs=-1",
        },
    }

    report_lines = build_report_lines(base_df, labelled_df, train_df, val_df, test_df, metrics_rows, reference_row, feature_notes)
    report_lines.insert(0, "Scenario 3 baseline training completed using training-only preprocessing and validation-only evaluation.")
    report_lines.append("")
    report_lines.append(f"Saved metrics CSV: {METRICS_FILE}")
    report_lines.append(f"Saved validation confusion matrices: {CM_FILE}")
    report_lines.append(f"Saved validation predictions: {PRED_FILE}")
    report_lines.append(f"Saved pipelines: {DT_MODEL_FILE}, {RF_MODEL_FILE}, {XGB_MODEL_FILE}")

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Progress: Report written to {REPORT_FILE}")
    print("Done. Review the scripts before running them.")


if __name__ == "__main__":
    main()
