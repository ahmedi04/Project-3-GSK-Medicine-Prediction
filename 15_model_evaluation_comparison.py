#!/usr/bin/env python3
from __future__ import annotations

"""
Scenario 3 Model Evaluation and Comparison
Section 16, Step 9

Loads all five saved models (Decision Tree, Random Forest, XGBoost, Neural Network, LSTM),
applies them to validation and test splits without retraining or retuning,
calculates metrics for each, selects the best among the first four using validation ROC-AUC,
and generates comprehensive comparison plots and final report.

Key principles:
- No model retraining or retuning
- Use validation ROC-AUC to select best of (DecisionTree, RandomForest, XGBoost, NeuralNetwork)
- Include LSTM as educational comparison with explicit disclaimers
- Cutoff 0.50 throughout
- Record selection before revealing test results
"""
import argparse
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

try:
    import scipy.sparse as sp
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required for this script. Install with: pip install tensorflow")


# ============================================================================
# Configuration
# ============================================================================

BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")

# Saved model files
DT_MODEL_FILE = Path("models/scenario3_decision_tree_pipeline.joblib")
RF_MODEL_FILE = Path("models/scenario3_random_forest_pipeline.joblib")
XGB_MODEL_FILE = Path("models/scenario3_xgboost_pipeline.joblib")
NN_MODEL_FILE = Path("models/scenario3_neural_network.keras")
NN_PREPROCESSING_FILE = Path("models/scenario3_nn_preprocessing.joblib")
LSTM_MODEL_FILE = Path("models/scenario3_lstm_demo.keras")
LSTM_PREPROCESSING_FILE = Path("models/scenario3_lstm_demo_preprocessing.joblib")

# Existing validation predictions
BASELINE_VAL_PRED_FILE = Path("outputs/scenario3_validation_predictions.parquet")
NN_VAL_PRED_FILE = Path("outputs/scenario3_nn_validation_predictions.parquet")
LSTM_VAL_PRED_FILE = Path("outputs/scenario3_lstm_demo_validation_predictions.parquet")

# Output files
FINAL_COMPARISON_FILE = Path("outputs/scenario3_final_model_comparison.csv")
FINAL_REPORT_FILE = Path("outputs/scenario3_final_model_comparison_report.txt")
FINAL_TEST_PRED_FILE = Path("outputs/scenario3_final_test_predictions.parquet")
ROC_CURVES_FILE = Path("outputs/scenario3_final_roc_curves.png")
PR_CURVES_FILE = Path("outputs/scenario3_final_precision_recall_curves.png")
CM_FILE = Path("outputs/scenario3_final_confusion_matrices.png")
METRICS_BAR_FILE = Path("outputs/scenario3_final_metrics_bar_chart.png")

TARGET = "treatment_outcome"
EXCLUDED_COLUMNS = ["patient_id", TARGET, "admission_date", "adverse_event", "readmission_30d"]
CUTOFF = 0.50
EXPECTED_SPLIT_COUNTS = {"train": 591_525, "validation": 197_175, "test": 197_176}
EXPECTED_LABELLED_ROWS = 985_876
EXPECTED_BASE_SHAPE = (1_050_000, 31)
EXPECTED_ENCODED_FEATURE_COUNT = 121
EXPECTED_TIMESTEPS = 1


def ensure_scenario3_preprocessing_module() -> None:
    """Make saved sklearn pipelines importable even if the original module was renamed."""
    import importlib

    fixed_module = importlib.import_module("scenario3_preprocessing_fixed")
    sys.modules["scenario3_preprocessing"] = fixed_module


# ============================================================================
# Data Loading and Validation
# ============================================================================

def load_and_validate_data():
    """Load and validate Scenario 3 inputs and split assignments."""
    print("Progress: Loading and validating Scenario 3 inputs...")
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {BASE_FILE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split assignment file: {SPLIT_FILE}")

    base_df = pd.read_parquet(BASE_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    # Validate base shape and target coverage
    if base_df.shape != EXPECTED_BASE_SHAPE:
        raise ValueError(f"Expected base shape {EXPECTED_BASE_SHAPE}, found {base_df.shape}")

    labelled_count = int(base_df[TARGET].notna().sum())
    if labelled_count != EXPECTED_LABELLED_ROWS:
        raise ValueError(f"Expected {EXPECTED_LABELLED_ROWS:,} labelled rows, found {labelled_count:,}")

    # Validate splits
    actual_split_counts = split_df["split"].value_counts().to_dict()
    if actual_split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Split count mismatch. Expected {EXPECTED_SPLIT_COUNTS}, found {actual_split_counts}")

    # Join splits to base data
    merged = split_df.merge(base_df, on="patient_id", how="inner", validate="one_to_one", sort=False)
    labelled = merged[merged[TARGET].notna()].copy()

    if len(labelled) != EXPECTED_LABELLED_ROWS:
        raise ValueError(f"Expected {EXPECTED_LABELLED_ROWS:,} labelled rows after merge, found {len(labelled):,}")

    # Verify no split overlap
    train_ids = set(labelled.loc[labelled["split"] == "train", "patient_id"])
    val_ids = set(labelled.loc[labelled["split"] == "validation", "patient_id"])
    test_ids = set(labelled.loc[labelled["split"] == "test", "patient_id"])

    if len(train_ids & val_ids) > 0 or len(train_ids & test_ids) > 0 or len(val_ids & test_ids) > 0:
        raise ValueError("Split overlap detected")

    if len(train_ids) != EXPECTED_SPLIT_COUNTS["train"] or len(val_ids) != EXPECTED_SPLIT_COUNTS["validation"] or len(test_ids) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError("Split size mismatch")

    # Validate binary targets
    y_numeric = pd.to_numeric(labelled[TARGET], errors="coerce")
    if y_numeric.isna().any() or not set(y_numeric.unique().tolist()).issubset({0, 1}):
        raise ValueError("Target contains invalid binary values")
    labelled[TARGET] = y_numeric.astype(int)

    # Extract splits
    train_df = labelled[labelled["split"] == "train"].copy()
    val_df = labelled[labelled["split"] == "validation"].copy()
    test_df = labelled[labelled["split"] == "test"].copy()

    predictor_columns = [c for c in base_df.columns if c not in EXCLUDED_COLUMNS]
    if len(predictor_columns) != 26:
        raise ValueError(f"Expected 26 base predictors, found {len(predictor_columns)}")

    print(f"  Train: {len(train_df):,} rows, {(train_df[TARGET] == 1).sum():,} positive ({100 * (train_df[TARGET] == 1).sum() / len(train_df):.2f}%)")
    print(f"  Validation: {len(val_df):,} rows, {(val_df[TARGET] == 1).sum():,} positive ({100 * (val_df[TARGET] == 1).sum() / len(val_df):.2f}%)")
    print(f"  Test: {len(test_df):,} rows, {(test_df[TARGET] == 1).sum():,} positive ({100 * (test_df[TARGET] == 1).sum() / len(test_df):.2f}%)")

    return base_df, labelled, train_df, val_df, test_df, predictor_columns


# ============================================================================
# Model Loading
# ============================================================================

def load_sklearn_pipeline(filepath, model_name):
    """Load a fitted sklearn pipeline."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing {model_name} model file: {filepath}")
    print(f"Progress: Loading {model_name} from {filepath}...")
    ensure_scenario3_preprocessing_module()
    return joblib.load(filepath)


def load_keras_model(filepath, model_name):
    """Load a Keras model."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing {model_name} model file: {filepath}")
    print(f"Progress: Loading {model_name} from {filepath}...")
    return tf.keras.models.load_model(filepath)


def load_preprocessing_bundle(filepath, bundle_name):
    """Load a preprocessing bundle."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing {bundle_name} file: {filepath}")
    print(f"Progress: Loading {bundle_name} from {filepath}...")
    bundle = joblib.load(filepath)
    required_keys = {
        "preprocessor",
        "scaler",
        "numeric_positions",
        "encoded_feature_names",
        "numeric_feature_names",
        "encoded_feature_count",
    }
    missing = sorted(required_keys - set(bundle))
    if missing:
        raise ValueError(f"{bundle_name} is missing required keys: {missing}")
    return bundle


def validate_encoded_feature_metadata(preprocessor, bundle, model_name):
    """Verify encoded feature count, numeric feature positions, and numeric feature names."""
    actual_feature_names = list(preprocessor.get_feature_names_out())
    expected_feature_names = list(bundle["encoded_feature_names"])
    if len(actual_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            f"{model_name} preprocessing produced {len(actual_feature_names)} encoded features; expected {EXPECTED_ENCODED_FEATURE_COUNT}."
        )
    if actual_feature_names != expected_feature_names:
        raise ValueError(f"{model_name} preprocessing feature order does not match saved bundle.")

    numeric_positions = np.asarray(bundle["numeric_positions"], dtype=np.int64)
    if numeric_positions.ndim != 1:
        raise ValueError(f"{model_name} numeric_positions must be one-dimensional.")

    expected_numeric_names = list(bundle["numeric_feature_names"])
    actual_numeric_names = [actual_feature_names[pos] for pos in numeric_positions]
    if len(numeric_positions) != len(expected_numeric_names):
        raise ValueError(
            f"{model_name} numeric position count {len(numeric_positions)} does not match numeric feature name count {len(expected_numeric_names)}."
        )
    if actual_numeric_names != expected_numeric_names:
        raise ValueError(f"{model_name} numeric positions do not match saved numeric feature names.")

    return numeric_positions, expected_feature_names


def validate_probability_array(probabilities, y_true, model_name):
    """Ensure probability outputs are valid one-dimensional finite scores in [0, 1]."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 1:
        raise ValueError(f"{model_name} probabilities must be one-dimensional; found shape {probabilities.shape}.")
    if len(probabilities) != len(y_true):
        raise ValueError(
            f"{model_name} probabilities length mismatch: expected {len(y_true):,}, found {len(probabilities):,}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{model_name} probabilities contain non-finite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{model_name} probabilities must lie between 0 and 1.")
    return probabilities.astype("float64", copy=False)


# ============================================================================
# Prediction Generation
# ============================================================================

def prepare_nn_predictions(nn_bundle, X_val, X_test):
    """Apply NN preprocessing and generate predictions from the saved NN model."""
    print("Progress: Preparing Neural Network predictions...")
    
    preprocessor = nn_bundle["preprocessor"]
    scaler = nn_bundle["scaler"]
    numeric_positions, _ = validate_encoded_feature_metadata(preprocessor, nn_bundle, "Neural Network")

    # Transform without refitting
    X_val_transformed = preprocessor.transform(X_val)  # sparse matrix
    X_test_transformed = preprocessor.transform(X_test)  # sparse matrix

    # Convert to dense and apply scaler to numeric positions only
    X_val_dense = X_val_transformed.astype("float32").toarray() if sp.issparse(X_val_transformed) else np.asarray(X_val_transformed, dtype="float32")
    X_test_dense = X_test_transformed.astype("float32").toarray() if sp.issparse(X_test_transformed) else np.asarray(X_test_transformed, dtype="float32")

    X_val_dense[:, numeric_positions] = scaler.transform(X_val_dense[:, numeric_positions])
    X_test_dense[:, numeric_positions] = scaler.transform(X_test_dense[:, numeric_positions])

    # Generate probabilities
    nn_model = load_keras_model(NN_MODEL_FILE, "Neural Network")
    val_probs = nn_model.predict(X_val_dense, batch_size=512, verbose=0)[:, 0]
    test_probs = nn_model.predict(X_test_dense, batch_size=512, verbose=0)[:, 0]

    return val_probs, test_probs


def prepare_lstm_predictions(lstm_bundle, X_val, X_test):
    """Apply LSTM preprocessing and generate predictions from the saved LSTM model."""
    print("Progress: Preparing LSTM Demo predictions...")
    
    preprocessor = lstm_bundle["preprocessor"]
    scaler = lstm_bundle["scaler"]
    numeric_positions, _ = validate_encoded_feature_metadata(preprocessor, lstm_bundle, "LSTM")
    timesteps = lstm_bundle.get("timesteps", 1)

    if timesteps != EXPECTED_TIMESTEPS:
        raise ValueError(f"Expected LSTM timesteps={EXPECTED_TIMESTEPS}, found {timesteps}")

    # Transform without refitting
    X_val_transformed = preprocessor.transform(X_val)  # sparse matrix
    X_test_transformed = preprocessor.transform(X_test)  # sparse matrix

    # Convert to dense and apply scaler to numeric positions only
    X_val_dense = X_val_transformed.astype("float32").toarray() if sp.issparse(X_val_transformed) else np.asarray(X_val_transformed, dtype="float32")
    X_test_dense = X_test_transformed.astype("float32").toarray() if sp.issparse(X_test_transformed) else np.asarray(X_test_transformed, dtype="float32")

    X_val_dense[:, numeric_positions] = scaler.transform(X_val_dense[:, numeric_positions])
    X_test_dense[:, numeric_positions] = scaler.transform(X_test_dense[:, numeric_positions])

    # Reshape for LSTM: (N, encoded_features) -> (N, timesteps, encoded_features)
    encoded_feature_count = X_val_dense.shape[1]
    X_val_lstm = X_val_dense.reshape(-1, timesteps, encoded_feature_count)
    X_test_lstm = X_test_dense.reshape(-1, timesteps, encoded_feature_count)

    # Generate probabilities
    lstm_model = load_keras_model(LSTM_MODEL_FILE, "LSTM Demo")
    val_probs = lstm_model.predict(X_val_lstm, batch_size=512, verbose=0)[:, 0]
    test_probs = lstm_model.predict(X_test_lstm, batch_size=512, verbose=0)[:, 0]

    return val_probs, test_probs


def load_existing_predictions():
    """Load existing validation predictions from baseline and deep learning models."""
    print("Progress: Loading existing validation predictions...")

    if not BASELINE_VAL_PRED_FILE.exists():
        raise FileNotFoundError(f"Missing baseline validation predictions: {BASELINE_VAL_PRED_FILE}")
    baseline_val = pd.read_parquet(BASELINE_VAL_PRED_FILE)
    print(f"  Baseline sklearn: {baseline_val.shape}")

    if not NN_VAL_PRED_FILE.exists():
        raise FileNotFoundError(f"Missing NN validation predictions: {NN_VAL_PRED_FILE}")
    nn_val = pd.read_parquet(NN_VAL_PRED_FILE)
    print(f"  Neural Network: {nn_val.shape}")

    if not LSTM_VAL_PRED_FILE.exists():
        raise FileNotFoundError(f"Missing LSTM validation predictions: {LSTM_VAL_PRED_FILE}")
    lstm_val = pd.read_parquet(LSTM_VAL_PRED_FILE)
    print(f"  LSTM Demo: {lstm_val.shape}")

    return baseline_val, nn_val, lstm_val


def merge_validation_predictions(val_df, baseline_val, nn_val, lstm_val):
    """Join saved validation predictions by patient_id and validate exact target alignment."""
    print("Progress: Joining saved validation predictions by patient_id...")

    validation_truth = val_df[["patient_id", TARGET]].rename(columns={TARGET: "actual_target"}).copy()
    validation_truth = validation_truth.reset_index(drop=True)

    for name, df in {
        "baseline": baseline_val,
        "neural_network": nn_val,
        "lstm": lstm_val,
    }.items():
        if df["patient_id"].isna().any() or not df["patient_id"].is_unique:
            raise ValueError(f"{name} validation predictions must have unique non-missing patient_id values.")
        if len(df) != EXPECTED_SPLIT_COUNTS["validation"]:
            raise ValueError(f"{name} validation prediction row count mismatch: expected {EXPECTED_SPLIT_COUNTS['validation']:,}, found {len(df):,}")

    merged = validation_truth.merge(
        baseline_val,
        on=["patient_id", "actual_target"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.merge(
        nn_val.rename(columns={"positive_class_probability": "neuralnetwork_probability"}),
        on=["patient_id", "actual_target"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.merge(
        lstm_val.rename(columns={"positive_class_probability": "lstm_probability"}),
        on=["patient_id", "actual_target"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError("Saved validation predictions do not cover the validation split exactly once.")

    expected_ids = validation_truth["patient_id"].tolist()
    if merged["patient_id"].tolist() != expected_ids:
        merged = merged.set_index("patient_id").loc[expected_ids].reset_index()

    return merged


# ============================================================================
# Metrics Calculation
# ============================================================================

def evaluate_metrics(y_true, probabilities, threshold=0.50):
    """Calculate metrics at a fixed threshold."""
    preds = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "ap": average_precision_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]),
    }


def create_metrics_dataframe(val_metrics_dict, test_metrics_dict, model_names):
    """Organize metrics from all models into a DataFrame."""
    rows = []
    for model_name in model_names:
        val_m = val_metrics_dict[model_name]
        test_m = test_metrics_dict[model_name]
        rows.append({
            "model": model_name,
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "val_ap": val_m["ap"],
            "val_tn": int(val_m["confusion_matrix"][0, 0]),
            "val_fp": int(val_m["confusion_matrix"][0, 1]),
            "val_fn": int(val_m["confusion_matrix"][1, 0]),
            "val_tp": int(val_m["confusion_matrix"][1, 1]),
            "test_accuracy": test_m["accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"],
            "test_roc_auc": test_m["roc_auc"],
            "test_ap": test_m["ap"],
            "test_tn": int(test_m["confusion_matrix"][0, 0]),
            "test_fp": int(test_m["confusion_matrix"][0, 1]),
            "test_fn": int(test_m["confusion_matrix"][1, 0]),
            "test_tp": int(test_m["confusion_matrix"][1, 1]),
        })
    return pd.DataFrame(rows)


def create_reference_metrics(y_true):
    """Create always-predict-0 reference metrics using zero scores and zero predictions."""
    zero_probs = np.zeros(len(y_true), dtype="float64")
    return evaluate_metrics(y_true, zero_probs, threshold=CUTOFF)


def summarize_rankings(metrics_df, metric_column):
    ranked = metrics_df.sort_values(metric_column, ascending=False)[["model", metric_column]]
    return [f"{row.model}: {row[metric_column]:.4f}" for _, row in ranked.iterrows()]


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_curves(y_val, y_test, all_val_probs, all_test_probs, model_names, model_colors, selected_model_name):
    """Plot ROC curves for validation and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Validation ROC
    ax = axes[0]
    for model_name in model_names:
        fpr, tpr, _ = roc_curve(y_val, all_val_probs[model_name])
        auc = roc_auc_score(y_val, all_val_probs[model_name])
        linestyle = "-" if model_name == selected_model_name else "--"
        linewidth = 2.5 if model_name == selected_model_name else 1.5
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", color=model_colors[model_name],
                linestyle=linestyle, linewidth=linewidth)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Validation Split")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Test ROC
    ax = axes[1]
    for model_name in model_names:
        fpr, tpr, _ = roc_curve(y_test, all_test_probs[model_name])
        auc = roc_auc_score(y_test, all_test_probs[model_name])
        linestyle = "-" if model_name == selected_model_name else "--"
        linewidth = 2.5 if model_name == selected_model_name else 1.5
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", color=model_colors[model_name],
                linestyle=linestyle, linewidth=linewidth)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Test Split")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(ROC_CURVES_FILE, dpi=150)
    plt.close(fig)
    print(f"Progress: Saved ROC curves to {ROC_CURVES_FILE}")


def plot_pr_curves(y_val, y_test, all_val_probs, all_test_probs, model_names, model_colors, selected_model_name):
    """Plot Precision-Recall curves for validation and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Validation PR
    ax = axes[0]
    for model_name in model_names:
        precision, recall, _ = precision_recall_curve(y_val, all_val_probs[model_name])
        ap = average_precision_score(y_val, all_val_probs[model_name])
        linestyle = "-" if model_name == selected_model_name else "--"
        linewidth = 2.5 if model_name == selected_model_name else 1.5
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})", color=model_colors[model_name],
                linestyle=linestyle, linewidth=linewidth)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Validation Split")
    val_baseline = float(np.mean(y_val))
    ax.axhline(val_baseline, color="black", linestyle=":", linewidth=1.5, alpha=0.8, label=f"Baseline prevalence={val_baseline:.4f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Test PR
    ax = axes[1]
    for model_name in model_names:
        precision, recall, _ = precision_recall_curve(y_test, all_test_probs[model_name])
        ap = average_precision_score(y_test, all_test_probs[model_name])
        linestyle = "-" if model_name == selected_model_name else "--"
        linewidth = 2.5 if model_name == selected_model_name else 1.5
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})", color=model_colors[model_name],
                linestyle=linestyle, linewidth=linewidth)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Test Split")
    test_baseline = float(np.mean(y_test))
    ax.axhline(test_baseline, color="black", linestyle=":", linewidth=1.5, alpha=0.8, label=f"Baseline prevalence={test_baseline:.4f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(PR_CURVES_FILE, dpi=150)
    plt.close(fig)
    print(f"Progress: Saved PR curves to {PR_CURVES_FILE}")


def plot_confusion_matrices(y_val, y_test, all_val_probs, all_test_probs, model_names, threshold=0.50):
    """Plot confusion matrices for validation and test splits side by side."""
    n_models = len(model_names)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8), constrained_layout=True)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    # Compute all confusion matrices to find max count for consistent color scale
    all_cms = []
    for model_name in model_names:
        val_cm = confusion_matrix(y_val, (all_val_probs[model_name] >= threshold).astype(int), labels=[0, 1])
        test_cm = confusion_matrix(y_test, (all_test_probs[model_name] >= threshold).astype(int), labels=[0, 1])
        all_cms.extend([val_cm, test_cm])

    max_count = max(cm.max() for cm in all_cms)

    labels = ["Ineffective (0)", "Effective (1)"]

    for idx, model_name in enumerate(model_names):
        # Validation CM
        val_cm = confusion_matrix(y_val, (all_val_probs[model_name] >= threshold).astype(int), labels=[0, 1])
        sns.heatmap(val_cm, annot=True, fmt=",d", cmap="Blues", cbar=False, square=True,
                    vmin=0, vmax=max_count, ax=axes[0, idx])
        axes[0, idx].set_title(f"{model_name} - Validation")
        axes[0, idx].set_xlabel("Predicted")
        axes[0, idx].set_ylabel("Actual")
        axes[0, idx].set_xticklabels(labels, rotation=0)
        axes[0, idx].set_yticklabels(labels, rotation=0)

        # Test CM
        test_cm = confusion_matrix(y_test, (all_test_probs[model_name] >= threshold).astype(int), labels=[0, 1])
        sns.heatmap(test_cm, annot=True, fmt=",d", cmap="Blues", cbar=False, square=True,
                    vmin=0, vmax=max_count, ax=axes[1, idx])
        axes[1, idx].set_title(f"{model_name} - Test")
        axes[1, idx].set_xlabel("Predicted")
        axes[1, idx].set_ylabel("Actual")
        axes[1, idx].set_xticklabels(labels, rotation=0)
        axes[1, idx].set_yticklabels(labels, rotation=0)

    fig.savefig(CM_FILE, dpi=150)
    plt.close(fig)
    print(f"Progress: Saved confusion matrices to {CM_FILE}")


def plot_metrics_bar_chart(metrics_df, selected_model_name):
    """Plot bar charts comparing metrics across models."""
    metrics_to_plot = ["val_f1", "val_roc_auc", "val_ap", "test_f1", "test_roc_auc", "test_ap"]
    metric_labels = ["F1 (Val)", "ROC-AUC (Val)", "AP (Val)", "F1 (Test)", "ROC-AUC (Test)", "AP (Test)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax_idx, (metric_col, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[ax_idx]
        data = metrics_df.set_index("model")[metric_col].sort_values(ascending=False)

        colors = ["#1f77b4" if name == selected_model_name else "#aec7e8" for name in data.index]
        bars = ax.bar(range(len(data)), data.values, color=colors, edgecolor="black", linewidth=1.5)

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(METRICS_BAR_FILE, dpi=150)
    plt.close(fig)
    print(f"Progress: Saved metrics bar chart to {METRICS_BAR_FILE}")


# ============================================================================
# Report Generation
# ============================================================================

def build_report_lines(base_df, labelled_df, train_df, val_df, test_df, metrics_df,
                       val_metrics_dict, test_metrics_dict, selected_model_name,
                       val_reference_metrics, test_reference_metrics,
                       model_config):
    """Build the final comparison report."""
    lines = []
    lines.append("Scenario 3 Model Evaluation and Comparison")
    lines.append("=========================================")
    lines.append("")
    lines.append("Overview")
    lines.append("--------")
    lines.append(f"Source file: {BASE_FILE}")
    lines.append(f"Split file: {SPLIT_FILE}")
    lines.append(f"Base shape: {base_df.shape}")
    lines.append(f"Labelled rows: {len(labelled_df):,}")
    lines.append("")
    lines.append("Data splits:")
    lines.append(f"- Train: {len(train_df):,} rows, {(train_df[TARGET] == 1).sum():,} positive ({100 * (train_df[TARGET] == 1).sum() / len(train_df):.2f}%)")
    lines.append(f"- Validation: {len(val_df):,} rows, {(val_df[TARGET] == 1).sum():,} positive ({100 * (val_df[TARGET] == 1).sum() / len(val_df):.2f}%)")
    lines.append(f"- Test: {len(test_df):,} rows, {(test_df[TARGET] == 1).sum():,} positive ({100 * (test_df[TARGET] == 1).sum() / len(test_df):.2f}%)")
    lines.append("")
    lines.append("Models")
    lines.append("------")
    lines.append("This comparison evaluates five pre-trained models without retraining or retuning:")
    lines.append("")
    lines.append("1. Decision Tree: Fitted sklearn Decision Tree pipeline")
    lines.append(f"   - File: {DT_MODEL_FILE}")
    lines.append("   - Parameters: max_depth=8, min_samples_split=20, min_samples_leaf=10")
    lines.append("")
    lines.append("2. Random Forest: Fitted sklearn Random Forest pipeline")
    lines.append(f"   - File: {RF_MODEL_FILE}")
    lines.append("   - Parameters: n_estimators=200, max_depth=12, min_samples_split=15")
    lines.append("")
    lines.append("3. XGBoost: Fitted XGBoost pipeline")
    lines.append(f"   - File: {XGB_MODEL_FILE}")
    lines.append("   - Parameters: n_estimators=200, max_depth=6, learning_rate=0.1")
    lines.append("")
    lines.append("4. Neural Network: Keras Dense model reusing Decision Tree's fitted preprocessor")
    lines.append(f"   - File: {NN_MODEL_FILE}")
    lines.append("   - Architecture: Dense(128)→BN→Dense(64)→BN→Dense(32)→Dense(1)")
    lines.append("   - Trained with EarlyStopping on validation ROC-AUC")
    lines.append("")
    lines.append("5. LSTM Demo: Keras LSTM model for educational purposes")
    lines.append(f"   - File: {LSTM_MODEL_FILE}")
    lines.append("   - DISCLAIMER: This model uses exactly ONE timestep per patient.")
    lines.append("     It is an educational sequence wrapper only and does not learn")
    lines.append("     temporal changes over time. Each patient contributes one observation.")
    lines.append("   - Architecture: LSTM(64)→LSTM(32)→Dense(16)→Dense(1)")
    lines.append("")
    lines.append("Model Selection Criterion")
    lines.append("------------------------")
    lines.append(f"Among Decision Tree, Random Forest, XGBoost, and Neural Network,")
    lines.append(f"the model with the highest validation ROC-AUC was selected.")
    lines.append(f"In case of ties, F1 was used as tiebreaker.")
    lines.append(f"LSTM is included as an educational comparison with the caveat above.")
    lines.append("")
    lines.append("VALIDATION-BASED SELECTION (Recorded Before Test Evaluation)")
    lines.append("-----------------------------------------------------------")
    lines.append(f"Selected Model: {selected_model_name}")
    lines.append(f"  - Validation Accuracy: {val_metrics_dict[selected_model_name]['accuracy']:.4f}")
    lines.append(f"  - Validation Precision: {val_metrics_dict[selected_model_name]['precision']:.4f}")
    lines.append(f"  - Validation Recall: {val_metrics_dict[selected_model_name]['recall']:.4f}")
    lines.append(f"  - Validation F1: {val_metrics_dict[selected_model_name]['f1']:.4f}")
    lines.append(f"  - Validation ROC-AUC: {val_metrics_dict[selected_model_name]['roc_auc']:.4f}")
    lines.append("")
    lines.append("Validation Metrics Summary")
    lines.append("--------------------------")
    lines.append(
        f"AlwaysZero reference: Accuracy={val_reference_metrics['accuracy']:.4f}, "
        f"Precision={val_reference_metrics['precision']:.4f}, Recall={val_reference_metrics['recall']:.4f}, "
        f"F1={val_reference_metrics['f1']:.4f}, ROC-AUC={val_reference_metrics['roc_auc']:.4f}, "
        f"AP={val_reference_metrics['ap']:.4f}"
    )
    for _, row in metrics_df.iterrows():
        model_name = row["model"]
        lines.append(f"{model_name}:")
        lines.append(f"  Accuracy={row['val_accuracy']:.4f}, Precision={row['val_precision']:.4f}, "
                     f"Recall={row['val_recall']:.4f}, F1={row['val_f1']:.4f}, "
                     f"ROC-AUC={row['val_roc_auc']:.4f}, AP={row['val_ap']:.4f}")
        lines.append(
            f"  TN={int(row['val_tn'])}, FP={int(row['val_fp'])}, FN={int(row['val_fn'])}, TP={int(row['val_tp'])}"
        )
    lines.append("")
    lines.append("Test Set Evaluation")
    lines.append("-------------------")
    lines.append(
        "This test split was evaluated in earlier scenarios. In this comparison, model selection uses validation results only, and the saved models are evaluated without retraining or retuning."
    )
    lines.append("")
    lines.append(
        f"AlwaysZero reference: Accuracy={test_reference_metrics['accuracy']:.4f}, "
        f"Precision={test_reference_metrics['precision']:.4f}, Recall={test_reference_metrics['recall']:.4f}, "
        f"F1={test_reference_metrics['f1']:.4f}, ROC-AUC={test_reference_metrics['roc_auc']:.4f}, "
        f"AP={test_reference_metrics['ap']:.4f}"
    )
    for _, row in metrics_df.iterrows():
        model_name = row["model"]
        lines.append(f"{model_name}:")
        lines.append(f"  Accuracy={row['test_accuracy']:.4f}, Precision={row['test_precision']:.4f}, "
                     f"Recall={row['test_recall']:.4f}, F1={row['test_f1']:.4f}, "
                     f"ROC-AUC={row['test_roc_auc']:.4f}, AP={row['test_ap']:.4f}")
        lines.append(
            f"  TN={int(row['test_tn'])}, FP={int(row['test_fp'])}, FN={int(row['test_fn'])}, TP={int(row['test_tp'])}"
        )
    lines.append("")
    lines.append("Baseline Comparison")
    lines.append("-------------------")
    lines.append("For context, the AlwaysZero reference predicts the majority class only.")
    lines.append("It can show high accuracy on this imbalanced dataset while having zero recall for the positive class.")
    lines.append("")
    lines.append("Key Insights")
    lines.append("------------")
    lines.append("1. Validation ROC-AUC ranking:")
    for ranking_line in summarize_rankings(metrics_df, "val_roc_auc"):
        lines.append(f"   - {ranking_line}")
    lines.append("2. Test ROC-AUC ranking:")
    for ranking_line in summarize_rankings(metrics_df, "test_roc_auc"):
        lines.append(f"   - {ranking_line}")
    lines.append("3. Precision-recall tradeoffs differ across models even when ROC-AUC is similar.")
    lines.append("4. The LSTM result is educational only because the data provides one observation per patient.")
    lines.append("5. These Scenario 3 models are evaluated on a fixed holdout split and were not refit for this script.")
    lines.append("")
    lines.append("Output Files")
    lines.append("------------")
    lines.append(f"- Metrics CSV: {FINAL_COMPARISON_FILE}")
    lines.append(f"- Test predictions: {FINAL_TEST_PRED_FILE}")
    lines.append(f"- ROC curves: {ROC_CURVES_FILE}")
    lines.append(f"- Precision-Recall curves: {PR_CURVES_FILE}")
    lines.append(f"- Confusion matrices: {CM_FILE}")
    lines.append(f"- Metrics bar chart: {METRICS_BAR_FILE}")
    lines.append("")
    lines.append("Cutoff: 0.50 (probability ≥ 0.50 → predict 1, else 0)")
    lines.append("")
    lines.append("This comparison script was run without retraining, retuning, or refitting models.")
    lines.append("All models were applied to fixed validation and test splits.")

    return lines


# ============================================================================
# Main Execution
# ============================================================================

def run_evaluation():
    """Main evaluation pipeline."""
    print("\n" + "=" * 80)
    print("Scenario 3 Model Evaluation and Comparison - Section 16, Step 9")
    print("=" * 80 + "\n")

    # Load and validate data
    base_df, labelled_df, train_df, val_df, test_df, predictor_columns = load_and_validate_data()

    X_val = val_df[predictor_columns].copy()
    y_val = val_df[TARGET].astype(int).copy()
    X_test = test_df[predictor_columns].copy()
    y_test = test_df[TARGET].astype(int).copy()

    # Load all models
    print("\nProgress: Loading all saved models...")
    dt_pipeline = load_sklearn_pipeline(DT_MODEL_FILE, "Decision Tree")
    rf_pipeline = load_sklearn_pipeline(RF_MODEL_FILE, "Random Forest")
    xgb_pipeline = load_sklearn_pipeline(XGB_MODEL_FILE, "XGBoost")
    nn_bundle = load_preprocessing_bundle(NN_PREPROCESSING_FILE, "Neural Network preprocessing bundle")
    lstm_bundle = load_preprocessing_bundle(LSTM_PREPROCESSING_FILE, "LSTM preprocessing bundle")
    validate_encoded_feature_metadata(nn_bundle["preprocessor"], nn_bundle, "Neural Network")
    validate_encoded_feature_metadata(lstm_bundle["preprocessor"], lstm_bundle, "LSTM")

    # Load existing validation predictions
    baseline_val, nn_val, lstm_val = load_existing_predictions()
    merged_val_predictions = merge_validation_predictions(val_df, baseline_val, nn_val, lstm_val)

    # Organize validation predictions from saved files only
    print("Progress: Organizing validation predictions...")
    all_val_probs = {
        "DecisionTree": merged_val_predictions["decisiontree_probability"].to_numpy(),
        "RandomForest": merged_val_predictions["randomforest_probability"].to_numpy(),
        "XGBoost": merged_val_predictions["xgboost_probability"].to_numpy(),
        "NeuralNetwork": merged_val_predictions["neuralnetwork_probability"].to_numpy(),
        "LSTM": merged_val_predictions["lstm_probability"].to_numpy(),
    }

    # Calculate validation metrics and select the best non-LSTM model before test evaluation
    print("Progress: Calculating validation metrics...")
    for model_name, probabilities in all_val_probs.items():
        all_val_probs[model_name] = validate_probability_array(probabilities, y_val, model_name)
    val_metrics_dict = {
        model_name: evaluate_metrics(y_val, all_val_probs[model_name], CUTOFF)
        for model_name in all_val_probs
    }
    val_reference_metrics = create_reference_metrics(y_val)

    print("Progress: Selecting best model based on validation ROC-AUC...")
    candidates = ["DecisionTree", "RandomForest", "XGBoost", "NeuralNetwork"]
    best_model = max(candidates, key=lambda m: (val_metrics_dict[m]["roc_auc"], val_metrics_dict[m]["f1"]))
    print(f"  Selected: {best_model} with validation ROC-AUC={val_metrics_dict[best_model]['roc_auc']:.4f}, F1={val_metrics_dict[best_model]['f1']:.4f}")

    # Generate test predictions after recording the validation-based selection
    print("Progress: Generating sklearn model predictions on test set...")
    dt_test_probs = dt_pipeline.predict_proba(X_test)[:, 1]
    rf_test_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    xgb_test_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

    # Generate predictions for deep learning models
    _, nn_test_probs = prepare_nn_predictions(nn_bundle, X_val, X_test)
    _, lstm_test_probs = prepare_lstm_predictions(lstm_bundle, X_val, X_test)

    # Organize test predictions
    all_test_probs = {
        "DecisionTree": dt_test_probs,
        "RandomForest": rf_test_probs,
        "XGBoost": xgb_test_probs,
        "NeuralNetwork": nn_test_probs,
        "LSTM": lstm_test_probs,
    }

    # Calculate metrics
    print("Progress: Calculating test metrics for all models...")
    for model_name, probabilities in all_test_probs.items():
        all_test_probs[model_name] = validate_probability_array(probabilities, y_test, model_name)
    test_metrics_dict = {
        model_name: evaluate_metrics(y_test, all_test_probs[model_name], CUTOFF)
        for model_name in all_test_probs
    }
    test_reference_metrics = create_reference_metrics(y_test)

    # Create metrics dataframe
    model_names = ["DecisionTree", "RandomForest", "XGBoost", "NeuralNetwork", "LSTM"]
    metrics_df = create_metrics_dataframe(val_metrics_dict, test_metrics_dict, model_names)

    # Save metrics CSV
    print(f"Progress: Saving metrics CSV to {FINAL_COMPARISON_FILE}...")
    FINAL_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(FINAL_COMPARISON_FILE, index=False)

    # Save test predictions
    print(f"Progress: Saving test predictions to {FINAL_TEST_PRED_FILE}...")
    test_preds = pd.DataFrame({
        "patient_id": test_df["patient_id"].values,
        "actual_target": y_test.values,
        "decisiontree_probability": all_test_probs["DecisionTree"],
        "randomforest_probability": all_test_probs["RandomForest"],
        "xgboost_probability": all_test_probs["XGBoost"],
        "neuralnetwork_probability": all_test_probs["NeuralNetwork"],
        "lstm_probability": all_test_probs["LSTM"],
    })
    FINAL_TEST_PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    test_preds.to_parquet(FINAL_TEST_PRED_FILE, index=False)

    # Generate plots
    model_colors = {
        "DecisionTree": "#1f77b4",
        "RandomForest": "#ff7f0e",
        "XGBoost": "#2ca02c",
        "NeuralNetwork": "#d62728",
        "LSTM": "#9467bd",
    }

    plot_roc_curves(y_val, y_test, all_val_probs, all_test_probs, model_names, model_colors, best_model)
    plot_pr_curves(y_val, y_test, all_val_probs, all_test_probs, model_names, model_colors, best_model)
    plot_confusion_matrices(y_val, y_test, all_val_probs, all_test_probs, model_names, CUTOFF)
    plot_metrics_bar_chart(metrics_df, best_model)

    # Build and save report
    print(f"Progress: Building final report...")
    model_config = {
        "cutoff": CUTOFF,
    }
    report_lines = build_report_lines(base_df, labelled_df, train_df, val_df, test_df,
                                      metrics_df, val_metrics_dict, test_metrics_dict,
                                      selected_model_name=best_model,
                                      val_reference_metrics=val_reference_metrics,
                                      test_reference_metrics=test_reference_metrics,
                                      model_config=model_config)

    FINAL_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Progress: Report written to {FINAL_REPORT_FILE}")

    print("\n" + "=" * 80)
    print("Scenario 3 Model Evaluation and Comparison Complete")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Selected Model: {best_model}")
    print(f"  Validation ROC-AUC: {val_metrics_dict[best_model]['roc_auc']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics_dict[best_model]['roc_auc']:.4f}")
    print(f"\nOutput Files:")
    print(f"  - {FINAL_COMPARISON_FILE}")
    print(f"  - {FINAL_TEST_PRED_FILE}")
    print(f"  - {FINAL_REPORT_FILE}")
    print(f"  - {ROC_CURVES_FILE}")
    print(f"  - {PR_CURVES_FILE}")
    print(f"  - {CM_FILE}")
    print(f"  - {METRICS_BAR_FILE}")


def run_smoke_test():
    """Quick smoke test to validate structure without full evaluation."""
    print("\n" + "=" * 80)
    print("Scenario 3 Model Evaluation and Comparison - Smoke Test")
    print("=" * 80 + "\n")

    print("Progress: Running smoke test (minimal validation)...")
    print("  - Checking file existence...")

    files_to_check = [
        (BASE_FILE, "Validation input"),
        (SPLIT_FILE, "Split assignments"),
        (DT_MODEL_FILE, "Decision Tree model"),
        (RF_MODEL_FILE, "Random Forest model"),
        (XGB_MODEL_FILE, "XGBoost model"),
        (NN_MODEL_FILE, "Neural Network model"),
        (NN_PREPROCESSING_FILE, "NN preprocessing bundle"),
        (LSTM_MODEL_FILE, "LSTM model"),
        (LSTM_PREPROCESSING_FILE, "LSTM preprocessing bundle"),
        (BASELINE_VAL_PRED_FILE, "Baseline validation predictions"),
        (NN_VAL_PRED_FILE, "NN validation predictions"),
        (LSTM_VAL_PRED_FILE, "LSTM validation predictions"),
    ]

    all_exist = True
    for filepath, description in files_to_check:
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {description}: {filepath}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\nERROR: Some required files are missing. Cannot proceed with smoke test.")
        return

    print("\n  - Loading and validating basic data shapes...")
    try:
        base_df = pd.read_parquet(BASE_FILE)
        split_df = pd.read_parquet(SPLIT_FILE)
        print(f"    ✓ Base: {base_df.shape}")
        print(f"    ✓ Splits: {split_df.shape}")

        # Load one model as test
        load_sklearn_pipeline(DT_MODEL_FILE, "Decision Tree (test)")
        print("    ✓ Decision Tree pipeline loaded")

        # Load one preprocessing bundle
        nn_bundle = load_preprocessing_bundle(NN_PREPROCESSING_FILE, "NN preprocessing (test)")
        print(f"    ✓ NN preprocessing bundle loaded with {nn_bundle['encoded_feature_count']} features")

        if nn_bundle["encoded_feature_count"] != EXPECTED_ENCODED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features, found {nn_bundle['encoded_feature_count']}"
            )

        # Load one Keras model
        load_keras_model(NN_MODEL_FILE, "Neural Network (test)")
        print("    ✓ Neural Network model loaded")

        baseline_val, nn_val, lstm_val = load_existing_predictions()
        print(f"    ✓ Validation prediction files loaded: {len(baseline_val):,}, {len(nn_val):,}, {len(lstm_val):,}")

        print("\nProgress: Smoke test completed successfully!")
        print("Ready to run full evaluation with: ./gsk_env/bin/python 15_model_evaluation_comparison.py --train")

    except Exception as e:
        print(f"\nERROR during smoke test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 3 model evaluation and comparison")
    parser.add_argument("--train", action="store_true", help="Run the full validation+test evaluation")
    args = parser.parse_args()

    if args.train:
        run_evaluation()
    else:
        run_smoke_test()
