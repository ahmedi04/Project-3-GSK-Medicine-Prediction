#!/usr/bin/env python3
"""Scenario 3 neural network training for Section 14, Step 7.

This script prepares the full Scenario 3 labelled split assignment dataset,
reuses the fitted Scenario 3 Decision Tree preprocessor without refitting,
standardizes only the numeric encoded features for Keras, and defines the PDF
architecture and training workflow.

The default entry point is conservative: it validates inputs and can run a
small smoke test, but full-dataset neural-network training is only executed
when explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")
DT_PIPELINE_FILE = Path("models/scenario3_decision_tree_pipeline.joblib")
NN_MODEL_FILE = Path("models/scenario3_neural_network.keras")
NN_PREPROCESSING_FILE = Path("models/scenario3_nn_preprocessing.joblib")
REPORT_FILE = Path("outputs/scenario3_nn_report.txt")
METRICS_FILE = Path("outputs/scenario3_nn_metrics.csv")
HISTORY_FILE = Path("outputs/scenario3_nn_training_history.csv")
HISTORY_PNG = Path("outputs/scenario3_nn_training_history.png")
CM_FILE = Path("outputs/scenario3_nn_validation_confusion_matrix.png")
PRED_FILE = Path("outputs/scenario3_nn_validation_predictions.parquet")

TARGET = "treatment_outcome"
EXCLUDED_COLUMNS = ["patient_id", TARGET, "admission_date", "adverse_event", "readmission_30d"]
EXPECTED_SPLIT_COUNTS = {"train": 591_525, "validation": 197_175, "test": 197_176}
EXPECTED_LABELLED_ROWS = 985_876
EXPECTED_BASE_SHAPE = (1_050_000, 31)
EXPECTED_LABEL_VALUES = {0, 1}
EXPECTED_ENCODED_FEATURE_COUNT = 121
RANDOM_SEED = 42
PDF_BATCH_SIZE = 512
PDF_MAX_EPOCHS = 100


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_scenario3_preprocessing_module() -> None:
    """Make the fitted Decision Tree pipeline importable even if the local module is broken."""
    import importlib

    fixed_module = importlib.import_module("scenario3_preprocessing_fixed")
    sys.modules["scenario3_preprocessing"] = fixed_module


def load_decision_tree_pipeline():
    ensure_scenario3_preprocessing_module()
    if not DT_PIPELINE_FILE.exists():
        raise FileNotFoundError(f"Missing fitted Decision Tree pipeline: {DT_PIPELINE_FILE}")
    pipeline = joblib.load(DT_PIPELINE_FILE)
    if "preprocessor" not in pipeline.named_steps:
        raise SystemExit("Decision Tree pipeline does not contain a fitted preprocessor step.")
    return pipeline


def validate_inputs(base_df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    if base_df.shape != EXPECTED_BASE_SHAPE:
        raise SystemExit(f"Scenario 3 base shape must be {EXPECTED_BASE_SHAPE}; found {base_df.shape}.")

    if base_df["patient_id"].isna().any() or not base_df["patient_id"].is_unique:
        raise SystemExit("Scenario 3 base patient_id values must be non-missing and unique.")
    if split_df["patient_id"].isna().any() or not split_df["patient_id"].is_unique:
        raise SystemExit("Scenario 3 split assignments patient_id values must be non-missing and unique.")

    if not set(split_df["split"]).issubset({"train", "validation", "test"}):
        raise SystemExit("Scenario 3 split assignments contain invalid split labels.")

    actual_counts = split_df["split"].value_counts().to_dict()
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(
            f"Scenario 3 split counts mismatch. Expected {EXPECTED_SPLIT_COUNTS}, found {actual_counts}."
        )

    target_series = pd.to_numeric(base_df[TARGET], errors="coerce")
    labelled_count = int(target_series.notna().sum())
    missing_count = int(target_series.isna().sum())
    if labelled_count != EXPECTED_LABELLED_ROWS or missing_count != (len(base_df) - EXPECTED_LABELLED_ROWS):
        raise SystemExit(
            "Scenario 3 target counts do not match expectations: "
            f"labelled={labelled_count:,}, missing={missing_count:,}."
        )

    valid_labels = set(target_series.dropna().astype(int).unique().tolist())
    if valid_labels - EXPECTED_LABEL_VALUES:
        raise SystemExit(f"Scenario 3 target contains invalid labels: {sorted(valid_labels)}")


def join_assignments(base_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
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

    labelled = merged[merged[TARGET].notna()].copy()
    if len(labelled) != EXPECTED_LABELLED_ROWS:
        raise SystemExit(
            f"Labelled merged rows mismatch: expected {EXPECTED_LABELLED_ROWS:,}, found {len(labelled):,}."
        )

    split_sets = {
        name: set(labelled.loc[labelled["split"] == name, "patient_id"])
        for name in ["train", "validation", "test"]
    }
    overlaps = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Scenario 3 split overlaps detected: {overlaps}")

    if len(split_sets["train"]) != EXPECTED_SPLIT_COUNTS["train"]:
        raise SystemExit("Scenario 3 train membership count does not match the saved Step 09 recipe.")
    if len(split_sets["validation"]) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise SystemExit("Scenario 3 validation membership count does not match the saved Step 09 recipe.")
    if len(split_sets["test"]) != EXPECTED_SPLIT_COUNTS["test"]:
        raise SystemExit("Scenario 3 test membership count does not match the saved Step 09 recipe.")

    labelled_patient_ids = set(labelled["patient_id"])
    assigned_patient_ids = set(split_df["patient_id"])
    if labelled_patient_ids != assigned_patient_ids:
        raise SystemExit("Scenario 3 split assignments do not cover the labelled patients exactly once.")

    labelled[TARGET] = pd.to_numeric(labelled[TARGET], errors="coerce").astype(int)
    if labelled[TARGET].isna().any() or not set(labelled[TARGET].unique().tolist()).issubset(EXPECTED_LABEL_VALUES):
        raise SystemExit("Scenario 3 labelled targets contain invalid binary values.")

    return labelled


def split_summary(name: str, y: pd.Series) -> str:
    total = len(y)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    pos_pct = (positive / total) * 100 if total else 0.0
    neg_pct = (negative / total) * 100 if total else 0.0
    return f"{name}: rows={total:,}, negative(0)={negative:,} ({neg_pct:.3f}%), positive(1)={positive:,} ({pos_pct:.3f}%)"


def evaluate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.50) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
    }


def ensure_sparse_csr(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


def align_preprocessor_features(preprocessor, features: pd.DataFrame) -> pd.DataFrame:
    expected_columns = list(preprocessor.feature_numeric_columns_) + list(preprocessor.feature_categorical_columns_)
    missing = [column for column in expected_columns if column not in features.columns]
    if missing:
        raise SystemExit(f"Preprocessed feature frame is missing columns: {missing}")
    return features[expected_columns].copy()


def prepare_encoded_matrices(preprocessor, X_train: pd.DataFrame, X_val: pd.DataFrame):
    train_sparse = ensure_sparse_csr(preprocessor.transform(X_train))
    val_sparse = ensure_sparse_csr(preprocessor.transform(X_val))

    encoded_feature_names = list(preprocessor.get_feature_names_out())
    if len(encoded_feature_names) != train_sparse.shape[1]:
        raise SystemExit(
            f"Encoded feature name count mismatch: {len(encoded_feature_names)} vs {train_sparse.shape[1]} matrix columns."
        )
    if train_sparse.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Scenario 3 expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded columns; found {train_sparse.shape[1]}."
        )

    feature_df = align_preprocessor_features(preprocessor, preprocessor.transform_features(X_train.iloc[:1]))
    numeric_feature_names = list(preprocessor.feature_numeric_columns_)
    numeric_positions = np.array(
        [index for index, feature_name in enumerate(encoded_feature_names) if feature_name in numeric_feature_names],
        dtype=np.int32,
    )
    if numeric_positions.size != len(numeric_feature_names):
        raise SystemExit("Unable to resolve the numeric feature positions from the encoded feature names.")
    expected_positions = np.arange(len(numeric_feature_names), dtype=np.int32)
    if not np.array_equal(numeric_positions, expected_positions):
        raise SystemExit(
            "Scenario 3 numeric features are not in the expected leading encoded positions; "
            "the preprocessing bundle assumptions need to be revisited."
        )

    train_dense = train_sparse.toarray().astype(np.float32, copy=False)
    val_dense = val_sparse.toarray().astype(np.float32, copy=False)

    scaler = StandardScaler()
    scaler.fit(train_dense[:, numeric_positions])
    train_dense[:, numeric_positions] = scaler.transform(train_dense[:, numeric_positions]).astype(np.float32)
    val_dense[:, numeric_positions] = scaler.transform(val_dense[:, numeric_positions]).astype(np.float32)

    if not np.isfinite(train_dense).all() or not np.isfinite(val_dense).all():
        raise SystemExit("Scaled Scenario 3 matrices contain non-finite values.")

    bundle = {
        "preprocessor": preprocessor,
        "scaler": scaler,
        "numeric_positions": numeric_positions,
        "encoded_feature_names": encoded_feature_names,
        "numeric_feature_names": numeric_feature_names,
        "encoded_feature_count": train_sparse.shape[1],
    }
    return train_dense, val_dense, encoded_feature_names, bundle


def build_nn_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="scenario3_inputs")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="scenario3_neural_network")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, object]]:
    set_global_seed(RANDOM_SEED)
    model = build_nn_model(X_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=5,
        verbose=1,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PDF_MAX_EPOCHS,
        batch_size=PDF_BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=2,
    )
    best_epoch = int(np.argmax(history.history["val_auc"]) + 1)
    fit_summary = {
        "actual_epochs_completed": int(len(history.history["loss"])),
        "best_epoch": best_epoch,
        "batch_size": PDF_BATCH_SIZE,
    }
    return model, history, fit_summary


def save_bundle(bundle: Dict[str, object]) -> None:
    NN_PREPROCESSING_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, NN_PREPROCESSING_FILE)


def save_history_plot(history: tf.keras.callbacks.History) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "epoch"
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history_df.reset_index().to_csv(HISTORY_FILE, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    epoch_axis = np.arange(1, len(history_df) + 1)

    axes[0].plot(epoch_axis, history_df["loss"], label="train_loss", linewidth=2)
    axes[0].plot(epoch_axis, history_df["val_loss"], label="val_loss", linewidth=2)
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_axis, history_df["auc"], label="train_auc", linewidth=2)
    axes[1].plot(epoch_axis, history_df["val_auc"], label="val_auc", linewidth=2)
    axes[1].set_title("Training and Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Scenario 3 Neural Network Training History")
    fig.savefig(HISTORY_PNG, dpi=150)
    plt.close(fig)


def save_validation_confusion_matrix(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    predictions = (probabilities >= 0.50).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["Ineffective (0)", "Effective (1)"],
        yticklabels=["Ineffective (0)", "Effective (1)"],
        ax=ax,
    )
    ax.set_title("Scenario 3 Validation Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    CM_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FILE, dpi=150)
    plt.close(fig)
    return cm


def build_report(
    base_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_bundle: Dict[str, object],
    class_weights: Dict[int, float],
    fit_summary: Dict[str, object],
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    history: tf.keras.callbacks.History,
) -> str:
    preprocessor = feature_bundle["preprocessor"]
    lines: List[str] = []
    lines.append("Scenario 3 Neural Network Report")
    lines.append("================================")
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
    lines.append("- Test data is preserved for the final comparison stage and is not used for training or selection.")
    lines.append("")
    lines.append("Preprocessing and feature handling:")
    lines.append("- Reused the fitted Decision Tree preprocessor without refitting.")
    lines.append(f"- Encoded feature count: {feature_bundle['encoded_feature_count']}")
    lines.append(f"- Numeric feature count before one-hot expansion: {len(preprocessor.feature_numeric_columns_)}")
    lines.append(f"- Categorical feature count before one-hot expansion: {len(preprocessor.feature_categorical_columns_)}")
    lines.append("- Numeric encoded positions were standardized with StandardScaler fitted only on training data.")
    lines.append("- One-hot indicators were left in their 0/1 form.")
    lines.append("")
    lines.append("Neural network architecture:")
    lines.append("- Input -> Dense(128, relu) -> BatchNormalization -> Dropout(0.3)")
    lines.append("- Dense(64, relu) -> BatchNormalization -> Dropout(0.3)")
    lines.append("- Dense(32, relu) -> Dropout(0.2) -> Dense(1, sigmoid)")
    lines.append("- Optimizer: Adam(lr=0.001)")
    lines.append("- Loss: binary_crossentropy")
    lines.append("- Metrics: accuracy, AUC(name='auc')")
    lines.append(f"- Batch size: {fit_summary['batch_size']} (runtime adaptation from the PDF's 64; every epoch still uses all training records)")
    lines.append(f"- Maximum epochs: {PDF_MAX_EPOCHS}")
    lines.append(f"- Actual epochs completed: {fit_summary['actual_epochs_completed']}")
    lines.append(f"- Selected best epoch: {fit_summary['best_epoch']}")
    lines.append("")
    lines.append("Class weights from training labels only:")
    lines.append(f"- class_0={class_weights[0]:.6f}, class_1={class_weights[1]:.6f}")
    lines.append("")
    lines.append("Training and validation metrics at cutoff 0.50:")
    for split_name, metrics in [("Train", train_metrics), ("Validation", val_metrics)]:
        lines.append(
            f"- {split_name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel().tolist()
        lines.append(f"  Confusion matrix [tn, fp, fn, tp]: {[tn, fp, fn, tp]}")
    lines.append("")
    lines.append("Saved artifacts:")
    lines.append(f"- Model: {NN_MODEL_FILE}")
    lines.append(f"- Preprocessing bundle: {NN_PREPROCESSING_FILE}")
    lines.append(f"- Metrics CSV: {METRICS_FILE}")
    lines.append(f"- Training history CSV: {HISTORY_FILE}")
    lines.append(f"- Training history figure: {HISTORY_PNG}")
    lines.append(f"- Validation confusion matrix: {CM_FILE}")
    lines.append(f"- Validation predictions: {PRED_FILE}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- The test split remains untouched for the final comparison stage.")
    lines.append("- The preprocessor bundle stores the fitted preprocessor, fitted scaler, numeric positions, and encoded feature names.")
    lines.append("- Standardization was applied only to the numeric encoded columns; the one-hot indicators remained 0/1.")
    lines.append("- All arrays were cast to float32 before model training and evaluation.")
    return "\n".join(lines)


def save_metrics_csv(
    class_weights: Dict[int, float],
    fit_summary: Dict[str, object],
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    encoded_feature_count: int,
    architecture: str,
) -> None:
    rows = []
    for split_name, metrics in [("train", train_metrics), ("validation", val_metrics)]:
        rows.append(
            {
                "split": split_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "tn": int(metrics["confusion_matrix"][0, 0]),
                "fp": int(metrics["confusion_matrix"][0, 1]),
                "fn": int(metrics["confusion_matrix"][1, 0]),
                "tp": int(metrics["confusion_matrix"][1, 1]),
                "batch_size": fit_summary["batch_size"],
                "actual_epochs_completed": fit_summary["actual_epochs_completed"],
                "best_epoch": fit_summary["best_epoch"],
                "encoded_feature_count": encoded_feature_count,
                "architecture": architecture,
                "class_weight_0": class_weights[0],
                "class_weight_1": class_weights[1],
            }
        )
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_FILE, index=False)


def save_validation_predictions(val_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    prediction_df = pd.DataFrame(
        {
            "patient_id": val_df["patient_id"].values,
            "actual_target": val_df[TARGET].astype(int).values,
            "positive_class_probability": probabilities.astype("float64"),
        }
    )
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(PRED_FILE, index=False)


def run_full_training() -> None:
    set_global_seed(RANDOM_SEED)

    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {BASE_FILE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split assignment file: {SPLIT_FILE}")

    print("Progress: Loading Scenario 3 inputs...")
    base_df = pd.read_parquet(BASE_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    validate_inputs(base_df, split_df)
    labelled_df = join_assignments(base_df, split_df)

    train_df = labelled_df[labelled_df["split"] == "train"].copy()
    val_df = labelled_df[labelled_df["split"] == "validation"].copy()
    test_df = labelled_df[labelled_df["split"] == "test"].copy()

    predictor_columns = [column for column in base_df.columns if column not in EXCLUDED_COLUMNS]
    if len(predictor_columns) != 26:
        raise SystemExit(f"Expected 26 base predictors, found {len(predictor_columns)}.")

    X_train = train_df[predictor_columns].copy()
    y_train = train_df[TARGET].astype(int).to_numpy(dtype=np.int32, copy=False)
    X_val = val_df[predictor_columns].copy()
    y_val = val_df[TARGET].astype(int).to_numpy(dtype=np.int32, copy=False)

    print("Progress: Loading fitted Decision Tree preprocessor...")
    pipeline = load_decision_tree_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]

    print("Progress: Transforming Scenario 3 predictors...")
    train_dense, val_dense, encoded_feature_names, bundle = prepare_encoded_matrices(preprocessor, X_train, X_val)

    if len(encoded_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Scenario 3 expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features, found {len(encoded_feature_names)}."
        )

    scaler = bundle["scaler"]
    numeric_positions = bundle["numeric_positions"]

    save_bundle(bundle)

    print("Progress: Calculating class weights...")
    class_weights = compute_class_weights(y_train)

    print("Progress: Building and training the Keras model...")
    model, history, fit_summary = train_neural_network(train_dense, y_train, val_dense, y_val, class_weights)

    print("Progress: Evaluating restored best weights...")
    train_probabilities = model.predict(train_dense, batch_size=PDF_BATCH_SIZE, verbose=0).reshape(-1)
    val_probabilities = model.predict(val_dense, batch_size=PDF_BATCH_SIZE, verbose=0).reshape(-1)

    train_metrics = evaluate_metrics(y_train, train_probabilities, threshold=0.50)
    val_metrics = evaluate_metrics(y_val, val_probabilities, threshold=0.50)

    print("Progress: Saving model and reports...")
    NN_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(NN_MODEL_FILE)
    save_validation_predictions(val_df, val_probabilities)
    save_history_plot(history)
    validation_cm = save_validation_confusion_matrix(y_val, val_probabilities)

    architecture = "Input -> Dense(128,relu) -> BN -> Dropout(0.3) -> Dense(64,relu) -> BN -> Dropout(0.3) -> Dense(32,relu) -> Dropout(0.2) -> Dense(1,sigmoid)"
    save_metrics_csv(class_weights, fit_summary, train_metrics, val_metrics, bundle["encoded_feature_count"], architecture)

    report_text = build_report(
        base_df=base_df,
        labelled_df=labelled_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_bundle=bundle,
        class_weights=class_weights,
        fit_summary=fit_summary,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        history=history,
    )
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("Progress: Scenario 3 neural-network training complete.")
    print(f"Validation confusion matrix: {validation_cm.tolist()}")
    print(f"Best epoch: {fit_summary['best_epoch']}")


def run_smoke_test() -> None:
    set_global_seed(RANDOM_SEED)
    print("Progress: Verifying TensorFlow/Keras with a tiny synthetic training run...")
    x_small = np.random.RandomState(RANDOM_SEED).normal(size=(32, 8)).astype(np.float32)
    y_small = (x_small[:, 0] + x_small[:, 1] > 0).astype(np.int32)

    inputs = tf.keras.Input(shape=(8,), dtype=tf.float32)
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    smoke_model = tf.keras.Model(inputs, outputs)
    smoke_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    history = smoke_model.fit(x_small, y_small, epochs=2, batch_size=8, verbose=0)

    smoke_path = Path("/tmp/scenario3_nn_smoke_test.keras")
    smoke_model.save(smoke_path)
    reloaded = tf.keras.models.load_model(smoke_path)
    reloaded.predict(x_small[:4], verbose=0)
    smoke_path.unlink(missing_ok=True)
    print(f"Progress: Tiny Keras smoke test completed with epochs={len(history.history['loss'])}.")

    print("Progress: Verifying the saved Scenario 3 Decision Tree preprocessor loads...")
    pipeline = load_decision_tree_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded_feature_names = list(preprocessor.get_feature_names_out())
    if len(encoded_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded feature names, found {len(encoded_feature_names)}."
        )
    if not hasattr(preprocessor, "encoded_feature_count_"):
        raise SystemExit("Loaded preprocessor is missing encoded_feature_count_.")
    print(
        "Progress: Preprocessor loaded successfully with "
        f"{preprocessor.encoded_feature_count_} encoded features and {len(preprocessor.feature_numeric_columns_)} numeric columns."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 3 neural network workflow.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the full Scenario 3 neural-network training workflow.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the lightweight TensorFlow and preprocessor smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        run_full_training()
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
