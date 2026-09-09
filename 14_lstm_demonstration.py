#!/usr/bin/env python3
"""Scenario 3 LSTM demonstration for Section 15, Step 8.

Important limitation:
- The prepared Scenario 3 dataset contains one row per patient and does not
  represent repeated visits or temporal sequences.
- This script therefore uses a single timestep per patient only, strictly as
  an educational LSTM demonstration.
- It does not reshape rows into invented visit histories or combine patients
  into sequences.

The default execution path runs a lightweight smoke test only. Full training is
available behind --train and is not executed unless explicitly requested.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.utils.class_weight import compute_class_weight


BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")
NN_PREPROCESSING_FILE = Path("models/scenario3_nn_preprocessing.joblib")
LSTM_MODEL_FILE = Path("models/scenario3_lstm_demo.keras")
LSTM_PREPROCESSING_BUNDLE_FILE = Path("models/scenario3_lstm_demo_preprocessing.joblib")
REPORT_FILE = Path("outputs/scenario3_lstm_demo_report.txt")
METRICS_FILE = Path("outputs/scenario3_lstm_demo_metrics.csv")
HISTORY_FILE = Path("outputs/scenario3_lstm_demo_training_history.csv")
HISTORY_PNG = Path("outputs/scenario3_lstm_demo_training_history.png")
CM_FILE = Path("outputs/scenario3_lstm_demo_validation_confusion_matrix.png")
PRED_FILE = Path("outputs/scenario3_lstm_demo_validation_predictions.parquet")
SMOKE_MODEL_FILE = Path("/tmp/scenario3_lstm_demo_smoke_test.keras")

TARGET = "treatment_outcome"
EXCLUDED_COLUMNS = ["patient_id", TARGET, "admission_date", "adverse_event", "readmission_30d"]
EXPECTED_SPLIT_COUNTS = {"train": 591_525, "validation": 197_175, "test": 197_176}
EXPECTED_LABELLED_ROWS = 985_876
EXPECTED_BASE_SHAPE = (1_050_000, 31)
EXPECTED_LABEL_VALUES = {0, 1}
EXPECTED_ENCODED_FEATURE_COUNT = 121
EXPECTED_TIMESTEPS = 1
RANDOM_SEED = 42
PDF_BATCH_SIZE = 512
PDF_MAX_EPOCHS = 50


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_preprocessing_bundle(bundle_path: Path = NN_PREPROCESSING_FILE) -> Dict[str, object]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing Scenario 3 neural-network preprocessing bundle: {bundle_path}")
    bundle = joblib.load(bundle_path)
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
        raise SystemExit(f"Scenario 3 preprocessing bundle is missing required keys: {missing}")
    return bundle


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

    labelled[TARGET] = pd.to_numeric(labelled[TARGET], errors="coerce")
    if labelled[TARGET].isna().any() or not set(labelled[TARGET].astype(int).unique().tolist()).issubset(EXPECTED_LABEL_VALUES):
        raise SystemExit("Scenario 3 labelled targets contain invalid binary values.")
    labelled[TARGET] = labelled[TARGET].astype(int)

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


def validate_encoded_feature_names(bundle: Dict[str, object], preprocessor) -> List[str]:
    encoded_feature_names = list(bundle["encoded_feature_names"])
    preprocessor_names = list(preprocessor.get_feature_names_out())
    if encoded_feature_names != preprocessor_names:
        raise SystemExit("Scenario 3 encoded feature names/order do not match the saved preprocessing bundle.")
    if len(encoded_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Scenario 3 expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features; found {len(encoded_feature_names)}."
        )
    return encoded_feature_names


def standardize_encoded_matrices(bundle: Dict[str, object], X_train: pd.DataFrame, X_val: pd.DataFrame):
    preprocessor = bundle["preprocessor"]
    scaler = bundle["scaler"]
    numeric_positions = np.asarray(bundle["numeric_positions"], dtype=np.int32)
    numeric_feature_names = list(bundle["numeric_feature_names"])
    encoded_feature_names = validate_encoded_feature_names(bundle, preprocessor)

    train_sparse = ensure_sparse_csr(preprocessor.transform(X_train)).astype(np.float32)
    val_sparse = ensure_sparse_csr(preprocessor.transform(X_val)).astype(np.float32)

    if train_sparse.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT or val_sparse.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Scenario 3 expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded columns; found {train_sparse.shape[1]} and {val_sparse.shape[1]}."
        )

    derived_positions = np.array(
        [index for index, feature_name in enumerate(encoded_feature_names) if feature_name in numeric_feature_names],
        dtype=np.int32,
    )
    if not np.array_equal(derived_positions, numeric_positions):
        raise SystemExit("Scenario 3 saved numeric positions do not match the encoded feature names/order.")

    train_dense = train_sparse.toarray().astype(np.float32, copy=False)
    val_dense = val_sparse.toarray().astype(np.float32, copy=False)

    train_dense[:, numeric_positions] = scaler.transform(train_dense[:, numeric_positions]).astype(np.float32)
    val_dense[:, numeric_positions] = scaler.transform(val_dense[:, numeric_positions]).astype(np.float32)

    if not np.isfinite(train_dense).all() or not np.isfinite(val_dense).all():
        raise SystemExit("Scaled Scenario 3 matrices contain non-finite values.")

    train_seq = train_dense.reshape((-1, EXPECTED_TIMESTEPS, EXPECTED_ENCODED_FEATURE_COUNT)).astype(np.float32, copy=False)
    val_seq = val_dense.reshape((-1, EXPECTED_TIMESTEPS, EXPECTED_ENCODED_FEATURE_COUNT)).astype(np.float32, copy=False)

    if train_seq.shape[0] != len(X_train) or val_seq.shape[0] != len(X_val):
        raise SystemExit("Scenario 3 sequence reshaping changed the patient count.")

    bundle_out = {
        "preprocessor": preprocessor,
        "scaler": scaler,
        "numeric_positions": numeric_positions,
        "encoded_feature_names": encoded_feature_names,
        "numeric_feature_names": numeric_feature_names,
        "encoded_feature_count": EXPECTED_ENCODED_FEATURE_COUNT,
        "timesteps": EXPECTED_TIMESTEPS,
    }
    return train_seq, val_seq, bundle_out


def build_lstm_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(EXPECTED_TIMESTEPS, input_dim), dtype=tf.float32, name="scenario3_lstm_inputs")
    x = tf.keras.layers.LSTM(64, return_sequences=True, stateful=False)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=False, stateful=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="scenario3_lstm_demo")
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


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
):
    set_global_seed(RANDOM_SEED)
    model = build_lstm_model(X_train.shape[2])
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
    LSTM_PREPROCESSING_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, LSTM_PREPROCESSING_BUNDLE_FILE)


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

    fig.suptitle("Scenario 3 LSTM Demonstration Training History")
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
    ax.set_title("Scenario 3 LSTM Validation Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    CM_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FILE, dpi=150)
    plt.close(fig)
    return cm


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
                "timesteps": EXPECTED_TIMESTEPS,
            }
        )
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_FILE, index=False)


def build_report(
    base_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle: Dict[str, object],
    class_weights: Dict[int, float],
    fit_summary: Dict[str, object],
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    architecture: str,
) -> str:
    lines: List[str] = []
    lines.append("Scenario 3 LSTM Demonstration Report")
    lines.append("====================================")
    lines.append(f"Source file: {BASE_FILE}")
    lines.append(f"Split file: {SPLIT_FILE}")
    lines.append(f"Base shape: {base_df.shape}")
    lines.append(f"Labelled rows: {len(labelled_df):,}")
    lines.append(f"Missing-target rows: {int(base_df[TARGET].isna().sum()):,}")
    lines.append("")
    lines.append("Important limitation:")
    lines.append("- The dataset has one row per patient, so this LSTM uses exactly one timestep per patient.")
    lines.append("- It is an educational sequence wrapper only; it does not learn changes over time.")
    lines.append("- No patient rows were combined into sequences and no visits were invented.")
    lines.append("")
    lines.append("Split counts and class distributions:")
    lines.append(f"- {split_summary('Train', train_df[TARGET])}")
    lines.append(f"- {split_summary('Validation', val_df[TARGET])}")
    lines.append(f"- {split_summary('Test', test_df[TARGET])}")
    lines.append("- Test data was preserved for later comparison and not used for training or selection.")
    lines.append("")
    lines.append("Preprocessing and feature handling:")
    lines.append("- Reused the saved Scenario 3 preprocessing bundle without refitting either the preprocessor or scaler.")
    lines.append(f"- Timesteps recorded in the preprocessing bundle: {bundle['timesteps']}")
    lines.append(f"- Encoded feature count: {bundle['encoded_feature_count']}")
    lines.append(f"- Numeric feature positions: {len(bundle['numeric_positions'])}")
    lines.append("- Numeric encoded columns were standardized using the saved scaler only; one-hot indicators remained unchanged.")
    lines.append("- All arrays were cast to float32 before dense conversion and sequence reshaping.")
    lines.append("")
    lines.append("LSTM architecture:")
    lines.append(f"- Input(shape=(1, {EXPECTED_ENCODED_FEATURE_COUNT}))")
    lines.append("- LSTM(64, return_sequences=True, stateful=False)")
    lines.append("- Dropout(0.3)")
    lines.append("- LSTM(32, return_sequences=False, stateful=False)")
    lines.append("- Dropout(0.3)")
    lines.append("- Dense(16, activation='relu')")
    lines.append("- Dropout(0.2)")
    lines.append("- Dense(1, activation='sigmoid')")
    lines.append("- Optimizer: Adam(lr=0.001)")
    lines.append("- Loss: binary_crossentropy")
    lines.append("- Metrics: accuracy, AUC(name='auc')")
    lines.append(f"- Batch size: {fit_summary['batch_size']} (runtime adaptation from the PDF's 32; every epoch still uses all training records)")
    lines.append(f"- Maximum epochs: {PDF_MAX_EPOCHS}")
    lines.append(f"- Actual epochs completed: {fit_summary['actual_epochs_completed']}")
    lines.append(f"- Selected best epoch: {fit_summary['best_epoch']}")
    lines.append("")
    lines.append("Class weights from training labels only:")
    lines.append(f"- class_0={class_weights[0]:.6f}, class_1={class_weights[1]:.6f}")
    lines.append("")
    lines.append("Metrics at cutoff 0.50:")
    for split_name, metrics in [("Train", train_metrics), ("Validation", val_metrics)]:
        lines.append(
            f"- {split_name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel().tolist()
        lines.append(f"  Confusion matrix [tn, fp, fn, tp]: {[tn, fp, fn, tp]}")
    lines.append("")
    lines.append("Saved artifacts:")
    lines.append(f"- Model: {LSTM_MODEL_FILE}")
    lines.append(f"- Preprocessing bundle: {LSTM_PREPROCESSING_BUNDLE_FILE}")
    lines.append(f"- Metrics CSV: {METRICS_FILE}")
    lines.append(f"- Training history CSV: {HISTORY_FILE}")
    lines.append(f"- Training history figure: {HISTORY_PNG}")
    lines.append(f"- Validation confusion matrix: {CM_FILE}")
    lines.append(f"- Validation predictions: {PRED_FILE}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- The test split remains untouched for the final comparison stage.")
    lines.append("- The preprocessing bundle stores the fitted preprocessor, fitted scaler, numeric positions, encoded feature names, and timesteps=1.")
    lines.append("- This demonstration keeps the patient order and target alignment unchanged throughout the workflow.")
    return "\n".join(lines)


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

    print("Progress: Loading saved neural-network preprocessing bundle...")
    bundle = load_preprocessing_bundle()
    preprocessor = bundle["preprocessor"]
    scaler = bundle["scaler"]
    if not hasattr(preprocessor, "transform") or not hasattr(scaler, "transform"):
        raise SystemExit("Loaded preprocessing bundle does not expose the required fitted components.")

    print("Progress: Transforming Scenario 3 predictors for LSTM input...")
    train_seq, val_seq, bundle_out = standardize_encoded_matrices(bundle, X_train, X_val)

    if bundle_out["encoded_feature_count"] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Scenario 3 expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features, found {bundle_out['encoded_feature_count']}."
        )
    if bundle_out["timesteps"] != EXPECTED_TIMESTEPS:
        raise SystemExit(f"Scenario 3 expected timesteps={EXPECTED_TIMESTEPS}, found {bundle_out['timesteps']}.")

    save_bundle(bundle_out)

    print("Progress: Calculating class weights...")
    class_weights = compute_class_weights(y_train)

    print("Progress: Building and training the LSTM model...")
    model, history, fit_summary = train_lstm_model(train_seq, y_train, val_seq, y_val, class_weights)

    print("Progress: Evaluating restored best weights...")
    train_probabilities = model.predict(train_seq, batch_size=PDF_BATCH_SIZE, verbose=0).reshape(-1)
    val_probabilities = model.predict(val_seq, batch_size=PDF_BATCH_SIZE, verbose=0).reshape(-1)

    train_metrics = evaluate_metrics(y_train, train_probabilities, threshold=0.50)
    val_metrics = evaluate_metrics(y_val, val_probabilities, threshold=0.50)

    print("Progress: Saving model and reports...")
    LSTM_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_FILE)
    save_validation_predictions(val_df, val_probabilities)
    save_history_plot(history)
    validation_cm = save_validation_confusion_matrix(y_val, val_probabilities)

    architecture = (
        "Input(shape=(1, 121)) -> LSTM(64, return_sequences=True) -> Dropout(0.3) -> "
        "LSTM(32, return_sequences=False) -> Dropout(0.3) -> Dense(16,relu) -> Dropout(0.2) -> Dense(1,sigmoid)"
    )
    save_metrics_csv(class_weights, fit_summary, train_metrics, val_metrics, bundle_out["encoded_feature_count"], architecture)

    report_text = build_report(
        base_df=base_df,
        labelled_df=labelled_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        bundle=bundle_out,
        class_weights=class_weights,
        fit_summary=fit_summary,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        architecture=architecture,
    )
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("Progress: Scenario 3 LSTM training complete.")
    print(f"Validation confusion matrix: {validation_cm.tolist()}")
    print(f"Best epoch: {fit_summary['best_epoch']}")


def run_smoke_test() -> None:
    set_global_seed(RANDOM_SEED)
    print("Progress: Verifying TensorFlow/Keras with a tiny synthetic LSTM training run...")
    x_small = np.random.RandomState(RANDOM_SEED).normal(size=(24, 1, 8)).astype(np.float32)
    y_small = (x_small[:, 0, 0] + x_small[:, 0, 1] > 0).astype(np.int32)

    inputs = tf.keras.Input(shape=(1, 8), dtype=tf.float32)
    x = tf.keras.layers.LSTM(4, return_sequences=False)(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    smoke_model = tf.keras.Model(inputs, outputs)
    smoke_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    history = smoke_model.fit(x_small, y_small, epochs=2, batch_size=4, verbose=0)

    smoke_model.save(SMOKE_MODEL_FILE)
    reloaded = tf.keras.models.load_model(SMOKE_MODEL_FILE)
    reloaded.predict(x_small[:3], verbose=0)
    SMOKE_MODEL_FILE.unlink(missing_ok=True)
    print(f"Progress: Tiny LSTM smoke test completed with epochs={len(history.history['loss'])}.")

    print("Progress: Verifying the saved Scenario 3 preprocessing bundle loads...")
    bundle = load_preprocessing_bundle()
    preprocessor = bundle["preprocessor"]
    scaler = bundle["scaler"]
    encoded_feature_names = list(bundle["encoded_feature_names"])
    numeric_positions = np.asarray(bundle["numeric_positions"], dtype=np.int32)
    if len(encoded_feature_names) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded feature names, found {len(encoded_feature_names)}."
        )
    if len(numeric_positions) != len(bundle["numeric_feature_names"]):
        raise SystemExit("Loaded preprocessing bundle has inconsistent numeric position metadata.")
    if not hasattr(preprocessor, "transform") or not hasattr(scaler, "transform"):
        raise SystemExit("Loaded preprocessing bundle is missing a fitted preprocessor or scaler.")
    print(
        "Progress: Preprocessing bundle loaded successfully with "
        f"{len(encoded_feature_names)} encoded features and {len(numeric_positions)} numeric positions."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 3 LSTM demonstration workflow.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the lightweight TensorFlow/Keras and preprocessing smoke tests.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the full Scenario 3 LSTM demonstration workflow.",
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
