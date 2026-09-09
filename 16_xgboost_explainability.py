#!/usr/bin/env python3
"""Scenario 3 XGBoost explainability (Section 17, Step 10: SHAP + LIME).

This script explains the saved Scenario 3 XGBoost pipeline without retraining,
retuning, refitting preprocessing, or changing patient splits.

Modes:
- default (no args): check required files and load the saved pipeline only.
- --run: execute complete SHAP and LIME explainability workflow.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ============================================================================
# Paths and constants
# ============================================================================

BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")
XGB_PIPELINE_FILE = Path("models/scenario3_xgboost_pipeline.joblib")
FINAL_TEST_PRED_FILE = Path("outputs/scenario3_final_test_predictions.parquet")

REPORT_FILE = Path("outputs/scenario3_xgboost_explainability_report.txt")
SHAP_SUMMARY_PNG = Path("outputs/scenario3_xgboost_shap_summary.png")
SHAP_VALUES_NPY = Path("outputs/scenario3_xgboost_shap_values.npy")
SHAP_IMPORTANCE_CSV = Path("outputs/scenario3_xgboost_shap_importance.csv")
SHAP_IMPORTANCE_BAR_PNG = Path("outputs/scenario3_xgboost_shap_importance_bar.png")
SHAP_FORCE_PNG = Path("outputs/scenario3_xgboost_shap_force_patient0.png")
SHAP_WATERFALL_PNG = Path("outputs/scenario3_xgboost_shap_waterfall_patient0.png")
LIME_PNG = Path("outputs/scenario3_xgboost_lime_patient0.png")
LIME_HTML = Path("outputs/scenario3_xgboost_lime_patient0.html")
LIME_CSV = Path("outputs/scenario3_xgboost_lime_patient0_contributions.csv")

TARGET = "treatment_outcome"
EXCLUDED_COLUMNS = ["patient_id", TARGET, "admission_date", "adverse_event", "readmission_30d"]
EXPECTED_BASE_SHAPE = (1_050_000, 31)
EXPECTED_LABELLED_ROWS = 985_876
EXPECTED_SPLIT_COUNTS = {"train": 591_525, "validation": 197_175, "test": 197_176}
EXPECTED_BASE_PREDICTOR_COUNT = 26
EXPECTED_ENCODED_FEATURE_COUNT = 121
CUTOFF = 0.50
RANDOM_SEED = 42
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 5000


# ============================================================================
# Utilities
# ============================================================================


def ensure_scenario3_preprocessing_module() -> None:
    """Allow joblib to load the saved pipeline if module alias differs."""
    fixed_module = importlib.import_module("scenario3_preprocessing_fixed")
    sys.modules["scenario3_preprocessing"] = fixed_module


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def validate_probability_array(probabilities: np.ndarray, expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(probabilities)
    if arr.ndim != 1:
        raise ValueError(f"{name} probabilities must be one-dimensional; found shape {arr.shape}.")
    if len(arr) != expected_len:
        raise ValueError(f"{name} probability length mismatch: expected {expected_len:,}, found {len(arr):,}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} probabilities contain non-finite values.")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name} probabilities must be between 0 and 1.")
    return arr.astype("float64", copy=False)


def normalize_shap_values(shap_values):
    """Normalize SHAP output to a 2D numpy array [n_samples, n_features]."""
    if isinstance(shap_values, list):
        # For binary classification some SHAP versions return [class0, class1]
        if len(shap_values) == 2:
            arr = np.asarray(shap_values[1])
        elif len(shap_values) == 1:
            arr = np.asarray(shap_values[0])
        else:
            raise ValueError(f"Unexpected SHAP list output length: {len(shap_values)}")
    else:
        arr = np.asarray(shap_values)

    if arr.ndim != 2:
        raise ValueError(f"Expected SHAP values to be 2D; found shape {arr.shape}")
    return arr


# ============================================================================
# Data and pipeline loading
# ============================================================================


def validate_required_files() -> None:
    required = [
        BASE_FILE,
        SPLIT_FILE,
        XGB_PIPELINE_FILE,
        FINAL_TEST_PRED_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def load_pipeline():
    ensure_scenario3_preprocessing_module()
    pipeline = joblib.load(XGB_PIPELINE_FILE)
    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Saved XGBoost pipeline must contain steps: preprocessor, model")
    return pipeline


def load_and_validate_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load base/split files, validate split integrity, and return train/val/test frames."""
    print("Progress: Loading Scenario 3 base data and split assignments...")
    base_df = pd.read_parquet(BASE_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    if base_df.shape != EXPECTED_BASE_SHAPE:
        raise ValueError(f"Base shape must be {EXPECTED_BASE_SHAPE}; found {base_df.shape}")

    if base_df["patient_id"].isna().any() or not base_df["patient_id"].is_unique:
        raise ValueError("Base patient_id must be unique and non-missing.")
    if split_df["patient_id"].isna().any() or not split_df["patient_id"].is_unique:
        raise ValueError("Split assignment patient_id must be unique and non-missing.")

    if not set(split_df["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Split file contains invalid split labels.")

    split_counts = split_df["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Split counts mismatch. Expected {EXPECTED_SPLIT_COUNTS}, found {split_counts}")

    # Preserve assignment order from the saved split file
    ordered = split_df.reset_index(drop=True).copy()
    ordered["assignment_order"] = np.arange(len(ordered), dtype=np.int64)

    merged = ordered.merge(base_df, on="patient_id", how="inner", validate="one_to_one", sort=False)
    merged = merged.sort_values("assignment_order", kind="stable").reset_index(drop=True)

    labelled = merged[merged[TARGET].notna()].copy()
    if len(labelled) != EXPECTED_LABELLED_ROWS:
        raise ValueError(f"Labelled row count mismatch. Expected {EXPECTED_LABELLED_ROWS:,}, found {len(labelled):,}")

    # Binary target validation
    y = pd.to_numeric(labelled[TARGET], errors="coerce")
    if y.isna().any() or not set(y.unique().tolist()).issubset({0, 1}):
        raise ValueError("Labelled target values must be binary {0,1}.")
    labelled[TARGET] = y.astype(int)

    # Split overlap validation
    train_ids = set(labelled.loc[labelled["split"] == "train", "patient_id"])
    val_ids = set(labelled.loc[labelled["split"] == "validation", "patient_id"])
    test_ids = set(labelled.loc[labelled["split"] == "test", "patient_id"])
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("Split overlap detected among train/validation/test.")

    if len(train_ids) != EXPECTED_SPLIT_COUNTS["train"] or len(val_ids) != EXPECTED_SPLIT_COUNTS["validation"] or len(test_ids) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError("Split membership counts do not match expectations.")

    predictor_columns = [c for c in base_df.columns if c not in EXCLUDED_COLUMNS]
    if len(predictor_columns) != EXPECTED_BASE_PREDICTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BASE_PREDICTOR_COUNT} base predictors after exclusions; found {len(predictor_columns)}"
        )

    train_df = labelled[labelled["split"] == "train"].copy()
    val_df = labelled[labelled["split"] == "validation"].copy()
    test_df = labelled[labelled["split"] == "test"].copy()

    print(
        "Progress: Split validation complete "
        f"(train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,})."
    )

    return base_df, labelled, train_df, val_df, test_df, predictor_columns


def resolve_predictor_columns(pipeline, dataset_predictor_columns: List[str]) -> List[str]:
    """Validate the 26 base predictor names and return saved preprocessor predictor order.

    The dataset may have interleaved numeric/categorical column order, while the
    fitted preprocessor stores numeric-first ordering in predictor_columns_.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    dataset_cols = list(dataset_predictor_columns)
    if len(dataset_cols) != EXPECTED_BASE_PREDICTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BASE_PREDICTOR_COUNT} dataset predictors; found {len(dataset_cols)}"
        )
    if len(set(dataset_cols)) != len(dataset_cols):
        raise ValueError("Dataset predictor columns contain duplicates.")

    if not hasattr(preprocessor, "predictor_columns_"):
        raise ValueError("Saved preprocessor is missing predictor_columns_.")

    saved_cols = list(preprocessor.predictor_columns_)
    if len(saved_cols) != EXPECTED_BASE_PREDICTOR_COUNT:
        raise ValueError(
            f"Saved preprocessor predictor_columns_ must have {EXPECTED_BASE_PREDICTOR_COUNT} columns; found {len(saved_cols)}"
        )
    if len(set(saved_cols)) != len(saved_cols):
        raise ValueError("Saved preprocessor predictor_columns_ contains duplicates.")

    dataset_set = set(dataset_cols)
    saved_set = set(saved_cols)
    missing_in_dataset = sorted(saved_set - dataset_set)
    extra_in_dataset = sorted(dataset_set - saved_set)
    if missing_in_dataset or extra_in_dataset:
        raise ValueError(
            "Base predictor names mismatch between dataset and saved preprocessor. "
            f"missing_in_dataset={missing_in_dataset}, extra_in_dataset={extra_in_dataset}"
        )

    return saved_cols


def validate_pipeline_preprocessing(pipeline, predictor_columns: List[str]) -> Tuple[object, np.ndarray]:
    """Validate preprocessor + encoded feature contract and return feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]

    # Verify predictor contract from fitted preprocessor in order used by pipeline
    if hasattr(preprocessor, "predictor_columns_"):
        if list(preprocessor.predictor_columns_) != list(predictor_columns):
            raise ValueError("Pipeline preprocessor predictor order does not match resolved saved preprocessor order.")

    encoded_feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    if len(encoded_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            f"Encoded feature count mismatch. Expected {EXPECTED_ENCODED_FEATURE_COUNT}, found {len(encoded_feature_names)}"
        )

    print(f"Progress: Verified predictor count={len(predictor_columns)} and encoded feature count={len(encoded_feature_names)}.")
    return preprocessor, encoded_feature_names


def build_encoded_matrices(pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, predictor_columns: List[str]):
    """Transform train and test via saved fitted preprocessor (no refit)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    X_train_base = train_df[predictor_columns].copy()
    X_test_base = test_df[predictor_columns].copy()

    print("Progress: Transforming train/test through saved preprocessor (no refit)...")
    X_train_encoded = preprocessor.transform(X_train_base)
    X_test_encoded = preprocessor.transform(X_test_base)

    if not sp.issparse(X_train_encoded) or not sp.issparse(X_test_encoded):
        raise ValueError("Expected sparse encoded matrices from saved preprocessor.")

    X_train_encoded = X_train_encoded.tocsr()
    X_test_encoded = X_test_encoded.tocsr()

    if X_test_encoded.shape[0] != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError(
            f"Test encoded row count mismatch. Expected {EXPECTED_SPLIT_COUNTS['test']:,}, found {X_test_encoded.shape[0]:,}"
        )
    if X_test_encoded.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            f"Encoded feature count mismatch on test transform. Expected {EXPECTED_ENCODED_FEATURE_COUNT}, found {X_test_encoded.shape[1]}"
        )

    return X_train_base, X_test_base, X_train_encoded, X_test_encoded


def build_single_patient_base_frame(test_df: pd.DataFrame, predictor_columns: List[str], patient_index: int = 0) -> pd.DataFrame:
    """Return first test patient in saved split order with resolved base predictor order."""
    if patient_index < 0 or patient_index >= len(test_df):
        raise IndexError(f"patient_index out of range: {patient_index}")
    return test_df.iloc[[patient_index]][predictor_columns].copy()


def verify_final_test_prediction_alignment(test_df: pd.DataFrame, pipeline_test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate final test file alignment and probability consistency by patient_id."""
    print("Progress: Validating final test prediction file alignment...")
    final_pred = pd.read_parquet(FINAL_TEST_PRED_FILE)

    needed_cols = {"patient_id", "actual_target", "xgboost_probability"}
    missing = sorted(needed_cols - set(final_pred.columns))
    if missing:
        raise ValueError(f"Final test prediction file missing columns: {missing}")

    if final_pred["patient_id"].isna().any() or not final_pred["patient_id"].is_unique:
        raise ValueError("Final test prediction patient_id must be unique and non-missing.")

    truth = test_df[["patient_id", TARGET]].rename(columns={TARGET: "actual_target"}).copy()
    merged = truth.merge(
        final_pred[["patient_id", "actual_target", "xgboost_probability"]],
        on=["patient_id", "actual_target"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError("Final test prediction file does not match test split IDs/targets exactly.")

    expected_ids = truth["patient_id"].tolist()
    if merged["patient_id"].tolist() != expected_ids:
        merged = merged.set_index("patient_id").loc[expected_ids].reset_index()

    saved_probs = validate_probability_array(
        merged["xgboost_probability"].to_numpy(),
        expected_len=EXPECTED_SPLIT_COUNTS["test"],
        name="Saved final XGBoost",
    )

    pipeline_test_probs = validate_probability_array(
        pipeline_test_probs,
        expected_len=EXPECTED_SPLIT_COUNTS["test"],
        name="Pipeline XGBoost",
    )

    max_abs_diff = float(np.max(np.abs(saved_probs - pipeline_test_probs)))
    if not np.allclose(saved_probs, pipeline_test_probs, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "Pipeline XGBoost probabilities do not match saved final test prediction file by patient_id. "
            f"max_abs_diff={max_abs_diff:.3e}"
        )

    print(f"Progress: Final test probabilities aligned (max_abs_diff={max_abs_diff:.3e}).")
    return saved_probs, merged["actual_target"].to_numpy(dtype=int)


# ============================================================================
# SHAP workflow
# ============================================================================


def run_shap(
    xgb_model,
    X_test_encoded: sp.csr_matrix,
    feature_names: np.ndarray,
    test_df: pd.DataFrame,
    pipeline_probs: np.ndarray,
):
    import shap

    print("Progress: Running SHAP TreeExplainer on all test patients (no sampling)...")
    explainer = shap.TreeExplainer(
        xgb_model,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )

    shap_values = None
    if SHAP_VALUES_NPY.exists():
        print(f"Progress: Reusing saved SHAP values from {SHAP_VALUES_NPY}...")
        shap_values = np.load(SHAP_VALUES_NPY)

    if shap_values is None:
        print("Progress: Saved SHAP values not found. This run will recompute SHAP values.")
        # Keep exact SHAP computation with additivity checks enabled.
        shap_values_raw = explainer.shap_values(X_test_encoded, check_additivity=True)
        shap_values = normalize_shap_values(shap_values_raw)
        np.save(SHAP_VALUES_NPY, shap_values)
        print(f"Progress: Saved SHAP values to {SHAP_VALUES_NPY}")

    if shap_values.shape != X_test_encoded.shape:
        raise ValueError(
            f"SHAP shape mismatch. shap_values={shap_values.shape}, encoded_input={X_test_encoded.shape}"
        )

    expected_value = explainer.expected_value
    base_value = float(np.ravel(np.asarray(expected_value))[0])

    # Verify sigmoid(expected + sum(shap)) matches pipeline probability.
    log_odds = base_value + shap_values.sum(axis=1)
    probs_from_shap = sigmoid(log_odds)
    probs_from_shap = validate_probability_array(probs_from_shap, len(pipeline_probs), "SHAP reconstructed")

    max_prob_diff = float(np.max(np.abs(probs_from_shap - pipeline_probs)))
    if not np.allclose(probs_from_shap, pipeline_probs, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "SHAP additivity reconstruction mismatch against pipeline probabilities. "
            f"max_abs_diff={max_prob_diff:.3e}"
        )

    print(f"Progress: SHAP additivity verified (max_abs_diff={max_prob_diff:.3e}).")

    # Global SHAP summary plot (all 197,176 test patients)
    # Use dense numeric feature values ONLY for plotting colors.
    # Keep sparse matrix for model prediction + SHAP calculations.
    print("Progress: Building dense feature-value array for SHAP summary color mapping...")
    X_test_plot_dense = X_test_encoded.astype(np.float32).toarray()
    if X_test_plot_dense.shape != shap_values.shape:
        raise ValueError(
            f"Dense plot matrix shape mismatch. dense={X_test_plot_dense.shape}, shap_values={shap_values.shape}"
        )
    if X_test_plot_dense.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name alignment mismatch for SHAP plot. dense_features={X_test_plot_dense.shape[1]}, names={len(feature_names)}"
        )

    print("Progress: Saving SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_plot_dense,
        feature_names=feature_names,
        show=False,
    )
    plt.gca().set_xlabel("SHAP value (log-odds contribution)")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP importance table + bar chart
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_log_odds": mean_abs,
        }
    ).sort_values("mean_abs_shap_log_odds", ascending=False)
    importance_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)

    top20 = importance_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature"], top20["mean_abs_shap_log_odds"], color="#2ca02c")
    plt.xlabel("Mean |SHAP value| (log-odds units)")
    plt.ylabel("Feature")
    plt.title("XGBoost SHAP Global Importance (Top 20, Test Set)")
    plt.tight_layout()
    plt.savefig(SHAP_IMPORTANCE_BAR_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    # First test patient in saved split order
    patient_index = 0
    patient_id = int(test_df.iloc[patient_index]["patient_id"])
    actual_target = int(test_df.iloc[patient_index][TARGET])
    probability = float(pipeline_probs[patient_index])
    prediction = int(probability >= CUTOFF)

    patient_shap = shap_values[patient_index]
    patient_encoded_sparse = X_test_encoded[patient_index]
    patient_encoded_dense = patient_encoded_sparse.toarray().ravel()

    # Force plot (matplotlib)
    print("Progress: Saving SHAP force plot for first test patient...")
    plt.figure(figsize=(14, 3.8))
    shap.force_plot(
        base_value,
        patient_shap,
        patient_encoded_dense,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_FORCE_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall plot
    print("Progress: Saving SHAP waterfall plot for first test patient...")
    exp = shap.Explanation(
        values=patient_shap,
        base_values=base_value,
        data=patient_encoded_dense,
        feature_names=list(feature_names),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_WATERFALL_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "base_value": base_value,
        "max_reconstruction_diff": max_prob_diff,
        "patient_index": patient_index,
        "patient_id": patient_id,
        "patient_actual_target": actual_target,
        "patient_probability": probability,
        "patient_prediction": prediction,
    }


# ============================================================================
# LIME workflow
# ============================================================================


def build_lime_reference(
    train_df: pd.DataFrame,
    predictor_columns: List[str],
    preprocessor,
):
    """Build LIME training reference data using train patients only and saved stats."""
    from scenario3_preprocessing_fixed import BASE_NUMERIC_COLUMNS, BASE_CATEGORICAL_COLUMNS

    print("Progress: Preparing LIME reference data from training patients only...")
    train_base = train_df[predictor_columns].copy()

    numeric_cols = [c for c in predictor_columns if c in BASE_NUMERIC_COLUMNS]
    categorical_cols = [c for c in predictor_columns if c in BASE_CATEGORICAL_COLUMNS]

    # Reuse saved preprocessor normalization + training imputation statistics.
    normalized_numeric, normalized_categorical = preprocessor._normalize_base_inputs(train_base)
    imputed_numeric = preprocessor._impute_numeric_from_stats(normalized_numeric)
    imputed_categorical = preprocessor._impute_categorical_from_stats(normalized_categorical)

    train_base = pd.DataFrame(index=train_base.index)
    for col in numeric_cols:
        train_base[col] = pd.to_numeric(imputed_numeric[col], errors="coerce").astype("float64")
    for col in categorical_cols:
        train_base[col] = imputed_categorical[col].astype("object")

    category_maps: Dict[str, List[str]] = {}
    category_to_code: Dict[str, Dict[str, int]] = {}

    for col in categorical_cols:
        categories = pd.Series(train_base[col].astype(str).unique()).sort_values(kind="stable").tolist()
        category_maps[col] = categories
        category_to_code[col] = {cat: idx for idx, cat in enumerate(categories)}

    lime_matrix = np.zeros((len(train_base), len(predictor_columns)), dtype="float64")
    for j, col in enumerate(predictor_columns):
        if col in numeric_cols:
            lime_matrix[:, j] = pd.to_numeric(train_base[col], errors="coerce").astype("float64").to_numpy()
        else:
            lime_matrix[:, j] = train_base[col].astype(str).map(category_to_code[col]).astype("float64").to_numpy()

    categorical_positions = [predictor_columns.index(col) for col in categorical_cols]
    categorical_names = {predictor_columns.index(col): category_maps[col] for col in categorical_cols}

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "category_maps": category_maps,
        "category_to_code": category_to_code,
        "categorical_positions": categorical_positions,
        "categorical_names": categorical_names,
        "lime_matrix": lime_matrix,
    }


def make_lime_prediction_wrapper(
    pipeline,
    predictor_columns: List[str],
    preprocessor,
    lime_meta,
):
    """Build wrapper that decodes LIME categorical integer codes back to strings."""
    numeric_cols = set(lime_meta["numeric_cols"])
    category_maps = lime_meta["category_maps"]

    def wrapper(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(predictor_columns):
            raise ValueError(
                f"LIME wrapper expected {len(predictor_columns)} features, found {X.shape[1]}"
            )

        out = pd.DataFrame(columns=predictor_columns, index=np.arange(X.shape[0]))

        for j, col in enumerate(predictor_columns):
            col_values = X[:, j]
            if col in numeric_cols:
                s = pd.to_numeric(pd.Series(col_values), errors="coerce").replace([np.inf, -np.inf], np.nan)
                s = s.fillna(float(preprocessor.numeric_medians_[col]))
                out[col] = s.astype("float64")
            else:
                labels = category_maps[col]
                mode_val = str(preprocessor.categorical_modes_[col])
                decoded = []
                for v in col_values:
                    if not np.isfinite(v):
                        decoded.append(mode_val)
                        continue
                    code = int(np.round(v))
                    code = max(0, min(code, len(labels) - 1))
                    decoded.append(labels[code])
                out[col] = pd.Series(decoded, dtype="object")

        probs_pos = pipeline.predict_proba(out)[:, 1]
        probs_pos = np.asarray(probs_pos, dtype="float64")
        if probs_pos.ndim != 1:
            raise ValueError(f"Wrapper probabilities must be 1D; found shape {probs_pos.shape}")
        if not np.isfinite(probs_pos).all():
            raise ValueError("Wrapper probabilities contain non-finite values.")
        if (probs_pos < 0).any() or (probs_pos > 1).any():
            raise ValueError("Wrapper probabilities must lie within [0, 1].")
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    return wrapper


def normalize_single_patient_for_lime(patient_df: pd.DataFrame, predictor_columns: List[str], preprocessor) -> pd.DataFrame:
    """Normalize and impute one patient exactly as saved preprocessor would for base columns."""
    row = patient_df[predictor_columns].copy()

    normalized_numeric, normalized_categorical = preprocessor._normalize_base_inputs(row)
    imputed_numeric = preprocessor._impute_numeric_from_stats(normalized_numeric)
    imputed_categorical = preprocessor._impute_categorical_from_stats(normalized_categorical)

    result = pd.DataFrame(index=row.index)
    for col in predictor_columns:
        if col in imputed_numeric.columns:
            result[col] = pd.to_numeric(imputed_numeric[col], errors="coerce").astype("float64")
        else:
            result[col] = imputed_categorical[col].astype("object")
    return result


def extend_lime_category_maps_for_patient(patient_norm_df: pd.DataFrame, predictor_columns: List[str], lime_meta) -> None:
    """Extend category mappings for unseen patient categories without changing training reference distribution."""
    categorical_cols = lime_meta["categorical_cols"]
    for col in categorical_cols:
        value = str(patient_norm_df.iloc[0][col])
        mapping = lime_meta["category_to_code"][col]
        labels = lime_meta["category_maps"][col]
        if value not in mapping:
            mapping[value] = len(labels)
            labels.append(value)


def encode_single_patient_for_lime(patient_norm_df: pd.DataFrame, predictor_columns: List[str], lime_meta):
    """Encode a normalized+imputed single patient in 26-feature space used by LIME."""
    row = patient_norm_df[predictor_columns].copy()
    numeric_cols = set(lime_meta["numeric_cols"])
    category_to_code = lime_meta["category_to_code"]

    vec = np.zeros(len(predictor_columns), dtype="float64")

    for j, col in enumerate(predictor_columns):
        value = row.iloc[0][col]
        if col in numeric_cols:
            v = pd.to_numeric(pd.Series([value]), errors="coerce").replace([np.inf, -np.inf], np.nan).iloc[0]
            if pd.isna(v):
                raise ValueError(f"Numeric value remained missing after normalization/imputation for column {col}")
            vec[j] = float(v)
        else:
            value = str(value)
            if value not in category_to_code[col]:
                raise ValueError(f"Categorical value {value!r} not present in LIME category map for column {col}")
            vec[j] = float(category_to_code[col][value])

    return vec


def prepare_lime_context(
    pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor_columns: List[str],
    patient_index: int,
):
    """Prepare LIME structures and verify unperturbed probability round-trip before SHAP."""
    from lime import lime_tabular

    preprocessor = pipeline.named_steps["preprocessor"]
    lime_meta = build_lime_reference(train_df, predictor_columns, preprocessor)

    patient_df = test_df.iloc[[patient_index]].copy()
    patient_norm_df = normalize_single_patient_for_lime(patient_df, predictor_columns, preprocessor)
    extend_lime_category_maps_for_patient(patient_norm_df, predictor_columns, lime_meta)

    # refresh categorical_names after potential extensions for unseen patient categories
    lime_meta["categorical_names"] = {
        predictor_columns.index(col): lime_meta["category_maps"][col]
        for col in lime_meta["categorical_cols"]
    }

    patient_vector = encode_single_patient_for_lime(patient_norm_df, predictor_columns, lime_meta)
    wrapper = make_lime_prediction_wrapper(pipeline, predictor_columns, preprocessor, lime_meta)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=lime_meta["lime_matrix"],
        feature_names=predictor_columns,
        class_names=["Ineffective", "Effective"],
        categorical_features=lime_meta["categorical_positions"],
        categorical_names=lime_meta["categorical_names"],
        mode="classification",
        random_state=RANDOM_SEED,
    )

    return {
        "explainer": explainer,
        "wrapper": wrapper,
        "patient_vector": patient_vector,
        "patient_df": patient_df,
        "patient_norm_df": patient_norm_df,
        "lime_meta": lime_meta,
    }


def run_lime(
    lime_context,
    patient_pipeline_prob: float,
):
    explainer = lime_context["explainer"]
    wrapper = lime_context["wrapper"]
    patient_vector = lime_context["patient_vector"]

    # Verify unperturbed wrapper prediction matches original pipeline probability.
    wrapper_prob = float(wrapper(patient_vector.reshape(1, -1))[0, 1])
    if not np.isclose(wrapper_prob, patient_pipeline_prob, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "LIME wrapper probability for unperturbed patient does not match pipeline probability. "
            f"wrapper={wrapper_prob:.12f}, pipeline={patient_pipeline_prob:.12f}"
        )

    print("Progress: Running LIME explanation for first test patient...")
    explanation = explainer.explain_instance(
        patient_vector,
        wrapper,
        labels=(1,),
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )

    # Save visualization artifacts
    fig = explanation.as_pyplot_figure(label=1)
    fig.tight_layout()
    fig.savefig(LIME_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    explanation.save_to_file(str(LIME_HTML))

    items = explanation.as_list(label=1)
    lime_df = pd.DataFrame(items, columns=["feature_rule", "weight"])
    lime_df["direction"] = np.where(lime_df["weight"] >= 0, "toward Effective (1)", "toward Ineffective (0)")
    lime_df.to_csv(LIME_CSV, index=False)

    local_score = float(explanation.score)
    local_pred = float(explanation.local_pred[0]) if explanation.local_pred is not None else np.nan

    return {
        "wrapper_probability": wrapper_prob,
        "local_fit_score": local_score,
        "local_approx_probability": local_pred,
        "lime_items": items,
    }


# ============================================================================
# Reporting
# ============================================================================


def write_report(
    base_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor_columns: List[str],
    feature_names: np.ndarray,
    shap_info: Dict[str, float],
    lime_info: Dict[str, float],
):
    lines: List[str] = []
    lines.append("Scenario 3 XGBoost Explainability Report (Step 10)")
    lines.append("===================================================")
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("- Model explained: saved Scenario 3 XGBoost pipeline (no retraining or retuning).")
    lines.append("- Neural Network remains the validation-selected model; XGBoost is explained here because it is the PDF example.")
    lines.append("- Split assignments and datasets were reused exactly as saved.")
    lines.append("")
    lines.append("Data checks")
    lines.append("-----------")
    lines.append(f"- Base dataset shape: {base_df.shape}")
    lines.append(f"- Train/Validation/Test counts: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    lines.append(f"- Base predictors used: {len(predictor_columns)}")
    lines.append(f"- Encoded features used by model: {len(feature_names)}")
    lines.append(f"- Test patients used for global SHAP: {len(test_df):,} (all test patients; no sampling)")
    lines.append("")
    lines.append("SHAP results")
    lines.append("------------")
    lines.append("- SHAP values are reported in log-odds units (not percentage points).")
    lines.append("- TreeExplainer used model_output='raw' and feature_perturbation='tree_path_dependent'.")
    lines.append("- SHAP additivity checks were enabled.")
    lines.append(
        f"- Probability reconstruction check: sigmoid(expected_value + sum(SHAP)) matched pipeline probabilities "
        f"with max absolute difference {shap_info['max_reconstruction_diff']:.3e}."
    )
    lines.append("")
    lines.append("Individual patient explained")
    lines.append("----------------------------")
    lines.append(f"- Patient index in saved test split order: {shap_info['patient_index']}")
    lines.append(f"- Patient ID: {shap_info['patient_id']}")
    lines.append(f"- Actual target: {shap_info['patient_actual_target']}")
    lines.append(f"- Predicted probability (Effective=1): {shap_info['patient_probability']:.6f}")
    lines.append(f"- Predicted class at cutoff 0.50: {shap_info['patient_prediction']}")
    lines.append("")
    lines.append("LIME results")
    lines.append("------------")
    lines.append("- LIME used the same patient as SHAP.")
    lines.append("- LIME reference data was built from training patients only.")
    lines.append("- LIME perturbs 26 base predictors and decodes categorical integer codes back to original labels before pipeline prediction.")
    lines.append("- LIME is a local approximation of model behavior, not the exact model itself.")
    lines.append(f"- LIME local fit score: {lime_info['local_fit_score']:.6f}")
    lines.append(f"- LIME local approximation for class 1 probability: {lime_info['local_approx_probability']:.6f}")
    lines.append(f"- Actual model probability for class 1: {shap_info['patient_probability']:.6f}")
    lines.append("")
    lines.append("Output files")
    lines.append("------------")
    lines.append(f"- {SHAP_SUMMARY_PNG}")
    lines.append(f"- {SHAP_IMPORTANCE_CSV}")
    lines.append(f"- {SHAP_IMPORTANCE_BAR_PNG}")
    lines.append(f"- {SHAP_FORCE_PNG}")
    lines.append(f"- {SHAP_WATERFALL_PNG}")
    lines.append(f"- {LIME_PNG}")
    lines.append(f"- {LIME_HTML}")
    lines.append(f"- {LIME_CSV}")
    lines.append(f"- {REPORT_FILE}")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append("These outputs describe model behavior on this dataset and split. They do not establish clinical causation.")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Mode runners
# ============================================================================


def run_check_only() -> None:
    print("Progress: Running default checks (files + pipeline load only)...")
    validate_required_files()
    pipeline = load_pipeline()
    model_name = type(pipeline.named_steps["model"]).__name__
    print(f"Progress: Loaded saved pipeline successfully. Model step: {model_name}")
    print("Done: Default checks completed. Use --run for full SHAP/LIME explainability.")


def run_full() -> None:
    print("Progress: Starting full XGBoost explainability workflow...")
    validate_required_files()

    pipeline = load_pipeline()
    xgb_model = pipeline.named_steps["model"]

    base_df, labelled_df, train_df, val_df, test_df, dataset_predictor_columns = load_and_validate_splits()

    # Resolve and enforce saved preprocessor predictor order for all inputs.
    predictor_columns = resolve_predictor_columns(pipeline, dataset_predictor_columns)

    preprocessor, feature_names = validate_pipeline_preprocessing(pipeline, predictor_columns)

    X_train_base, X_test_base, X_train_encoded, X_test_encoded = build_encoded_matrices(
        pipeline, train_df, test_df, predictor_columns
    )

    # Pipeline probabilities on test for alignment checks
    print("Progress: Generating XGBoost test probabilities from saved pipeline...")
    pipeline_probs = pipeline.predict_proba(X_test_base)[:, 1]
    pipeline_probs = validate_probability_array(pipeline_probs, EXPECTED_SPLIT_COUNTS["test"], "Pipeline XGBoost")

    # Unperturbed LIME wrapper probability check BEFORE full-test SHAP calculation.
    patient_index = 0
    patient_base = build_single_patient_base_frame(test_df, predictor_columns, patient_index=patient_index)
    patient_pipeline_prob = float(pipeline.predict_proba(patient_base)[:, 1][0])
    lime_context = prepare_lime_context(
        pipeline=pipeline,
        train_df=train_df,
        test_df=test_df,
        predictor_columns=predictor_columns,
        patient_index=patient_index,
    )
    pre_shap_wrapper_prob = float(lime_context["wrapper"](lime_context["patient_vector"].reshape(1, -1))[0, 1])
    if not np.isclose(pre_shap_wrapper_prob, patient_pipeline_prob, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "Pre-SHAP LIME wrapper probability check failed. "
            f"wrapper={pre_shap_wrapper_prob:.12f}, pipeline={patient_pipeline_prob:.12f}"
        )
    print("Progress: Pre-SHAP LIME unperturbed probability round-trip verified.")

    saved_probs, saved_targets = verify_final_test_prediction_alignment(test_df, pipeline_probs)

    shap_info = run_shap(
        xgb_model=xgb_model,
        X_test_encoded=X_test_encoded,
        feature_names=feature_names,
        test_df=test_df,
        pipeline_probs=saved_probs,
    )

    lime_info = run_lime(
        lime_context=lime_context,
        patient_pipeline_prob=float(shap_info["patient_probability"]),
    )

    write_report(
        base_df=base_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        predictor_columns=predictor_columns,
        feature_names=feature_names,
        shap_info=shap_info,
        lime_info=lime_info,
    )

    print(f"Done: Explainability outputs written with prefix outputs/scenario3_xgboost_ and report {REPORT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 3 XGBoost explainability (SHAP + LIME)")
    parser.add_argument("--run", action="store_true", help="Run full SHAP/LIME explainability workflow")
    args = parser.parse_args()

    if args.run:
        run_full()
    else:
        run_check_only()
