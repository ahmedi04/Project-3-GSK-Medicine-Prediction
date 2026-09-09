#!/usr/bin/env python3
"""
Scenario 2 baseline: 60/20/20 split for Decision Tree, Random Forest, and XGBoost.

This script is intentionally separate from the existing 80/20 scripts so all previous
results stay preserved.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


TARGET = "treatment_outcome"
SOURCE_FILE = Path("outputs/data_engineered.parquet")
OUT_CSV = Path("outputs/scenario2_602020_baseline_metrics.csv")
OUT_REPORT = Path("outputs/scenario2_602020_baseline_report.txt")
OUT_CM = Path("outputs/scenario2_602020_confusion_matrices.png")


def pct_positive(y):
    """Return positive-class percentage for quick class-balance checks."""
    return float(y.mean() * 100.0)


def evaluate_split_metrics(y_true, y_pred, y_prob):
    """Calculate standard binary metrics for one split."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    return out


def regenerate_confusion_matrix_chart_from_csv(csv_path=OUT_CSV, image_path=OUT_CM):
    """Recreate the saved confusion-matrix chart directly from the metrics CSV."""
    print("Progress: Regenerating confusion-matrix chart from existing CSV values...")
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    metrics_df = pd.read_csv(csv_path)
    title_map = {
        "DecisionTree": "Decision Tree",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
    }
    model_order = ["DecisionTree", "RandomForest", "XGBoost"]
    tick_labels = ["Ineffective (0)", "Effective (1)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, model_name in zip(axes, model_order):
        row = metrics_df.loc[metrics_df["model"] == model_name]
        if row.empty:
            raise ValueError(f"Missing confusion-matrix values for model: {model_name}")
        row = row.iloc[0]
        cm = np.array([
            [int(row["test_tn"]), int(row["test_fp"])],
            [int(row["test_fn"]), int(row["test_tp"])],
        ])
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
        )
        ax.set_title(title_map[model_name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticklabels(tick_labels, rotation=0)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(f"Progress: Replaced confusion-matrix chart at {image_path}")


def main():
    print("Progress: Loading full engineered dataset...")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    df = pd.read_parquet(SOURCE_FILE)

    # 2) Confirm exact source shape
    expected_shape = (1_050_000, 38)
    if df.shape != expected_shape:
        raise SystemExit(
            f"Shape check failed. Expected {expected_shape}, got {df.shape}."
        )
    print(f"Progress: Source shape verified: {df.shape}")

    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column: {TARGET}")

    # 4) Remove only rows with missing target
    missing_target_rows = int(df[TARGET].isna().sum())
    if missing_target_rows != 64_124:
        raise SystemExit(
            f"Expected 64,124 missing `{TARGET}` rows, found {missing_target_rows}."
        )
    print(f"Progress: Dropping {missing_target_rows:,} rows with missing target...")
    df_labelled = df[df[TARGET].notna()].copy()

    # Ensure binary numeric target (0/1)
    y = pd.to_numeric(df_labelled[TARGET], errors="coerce")
    if y.isna().any():
        raise SystemExit("Labelled data still has invalid target values after numeric coercion.")
    unique_y = set(y.unique().tolist())
    if not unique_y.issubset({0, 1}):
        raise SystemExit(f"Target must be binary 0/1, found values: {sorted(unique_y)}")
    y = y.astype(int)

    # 5) Exclude leakage/identifier/date columns
    excluded = {
        "patient_id": "identifier",
        TARGET: "target",
        "admission_date": "raw date",
        "adverse_event": "post-treatment outcome/data leakage",
        "readmission_30d": "post-treatment outcome/data leakage",
    }
    predictor_cols = [c for c in df_labelled.columns if c not in excluded]

    # 6) Confirm exactly 33 predictor columns remain
    if len(predictor_cols) != 33:
        raise SystemExit(
            f"Predictor count check failed. Expected 33, got {len(predictor_cols)}"
        )
    print("Progress: Predictor count verified at 33 columns.")

    X = df_labelled[predictor_cols].copy()

    # 7) Stratified 60/20/20 split using two train_test_split operations
    print("Progress: Creating stratified 60/20/20 train/validation/test split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # validation should be 20% of full data => 25% of remaining 80%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(
        f"Rows -> train: {len(X_train):,}, validation: {len(X_val):,}, test: {len(X_test):,}"
    )
    print(
        "Target % positive -> "
        f"train: {pct_positive(y_train):.3f}%, "
        f"validation: {pct_positive(y_val):.3f}%, "
        f"test: {pct_positive(y_test):.3f}%"
    )

    # 8) Preprocessing fit only on training data
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    # 9) Models with specified baseline parameters
    models = {
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
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.3,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Train, evaluate, and capture test confusion matrices
    rows = []
    test_conf_mats = {}

    for name, model in models.items():
        print(f"Progress: Training {name}...")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)

        # 10) Use normal 0.50 threshold
        train_prob = pipe.predict_proba(X_train)[:, 1]
        val_prob = pipe.predict_proba(X_val)[:, 1]
        test_prob = pipe.predict_proba(X_test)[:, 1]

        train_pred = (train_prob >= 0.50).astype(int)
        val_pred = (val_prob >= 0.50).astype(int)
        test_pred = (test_prob >= 0.50).astype(int)

        # 11) Metrics on all splits
        train_m = evaluate_split_metrics(y_train, train_pred, train_prob)
        val_m = evaluate_split_metrics(y_val, val_pred, val_prob)
        test_m = evaluate_split_metrics(y_test, test_pred, test_prob)

        tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
        test_conf_mats[name] = (tn, fp, fn, tp)

        row = {
            "model": name,
            "train_accuracy": train_m["accuracy"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_roc_auc": train_m["roc_auc"],
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "test_accuracy": test_m["accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"],
            "test_roc_auc": test_m["roc_auc"],
            "test_tn": tn,
            "test_fp": fp,
            "test_fn": fn,
            "test_tp": tp,
            "train_minus_val_f1": train_m["f1"] - val_m["f1"],
            "val_minus_test_f1": val_m["f1"] - test_m["f1"],
            "train_minus_val_auc": train_m["roc_auc"] - val_m["roc_auc"],
            "val_minus_test_auc": val_m["roc_auc"] - test_m["roc_auc"],
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # 12) Save three-panel confusion matrix image (test split)
    print("Progress: Saving three-panel test confusion matrix figure...")
    # 13) Save comparison CSV
    print("Progress: Saving baseline metrics CSV...")
    metrics_df.to_csv(OUT_CSV, index=False)

    # Recreate the chart from the saved CSV values so it can also be regenerated later
    regenerate_confusion_matrix_chart_from_csv(OUT_CSV, OUT_CM)

    # 14) Build readable report
    print("Progress: Writing baseline scenario report...")
    lines = []
    lines.append("Scenario 2 Baseline Report (60/20/20 split)")
    lines.append("===========================================")
    lines.append(f"Source file: {SOURCE_FILE}")
    lines.append(f"Source rows/cols: {df.shape[0]:,} / {df.shape[1]}")
    lines.append(f"Rows with missing {TARGET}: {missing_target_rows:,}")
    lines.append(f"Labelled rows used: {len(df_labelled):,}")
    lines.append("")
    lines.append("Excluded columns from predictors:")
    for c, reason in excluded.items():
        lines.append(f"- {c}: {reason}")
    lines.append(f"Predictor columns before preprocessing: {len(predictor_cols)}")
    lines.append("")
    lines.append("Split sizes and class distributions:")
    lines.append(
        f"- Train: {len(X_train):,} rows | positive={pct_positive(y_train):.3f}% | negative={100-pct_positive(y_train):.3f}%"
    )
    lines.append(
        f"- Validation: {len(X_val):,} rows | positive={pct_positive(y_val):.3f}% | negative={100-pct_positive(y_val):.3f}%"
    )
    lines.append(
        f"- Test: {len(X_test):,} rows | positive={pct_positive(y_test):.3f}% | negative={100-pct_positive(y_test):.3f}%"
    )
    lines.append("")
    lines.append("Metrics by model (train / validation / test):")

    for _, r in metrics_df.iterrows():
        lines.append("")
        lines.append(f"Model: {r['model']}")
        lines.append(
            "- Train: "
            f"accuracy={r['train_accuracy']:.4f}, precision={r['train_precision']:.4f}, "
            f"recall={r['train_recall']:.4f}, f1={r['train_f1']:.4f}, roc_auc={r['train_roc_auc']:.4f}"
        )
        lines.append(
            "- Validation: "
            f"accuracy={r['val_accuracy']:.4f}, precision={r['val_precision']:.4f}, "
            f"recall={r['val_recall']:.4f}, f1={r['val_f1']:.4f}, roc_auc={r['val_roc_auc']:.4f}"
        )
        lines.append(
            "- Test: "
            f"accuracy={r['test_accuracy']:.4f}, precision={r['test_precision']:.4f}, "
            f"recall={r['test_recall']:.4f}, f1={r['test_f1']:.4f}, roc_auc={r['test_roc_auc']:.4f}"
        )
        lines.append(
            f"- Test confusion matrix [tn, fp, fn, tp]: [{int(r['test_tn'])}, {int(r['test_fp'])}, {int(r['test_fn'])}, {int(r['test_tp'])}]"
        )
        lines.append(
            f"- Differences: train-validation f1={r['train_minus_val_f1']:.4f}, validation-test f1={r['val_minus_test_f1']:.4f}, "
            f"train-validation auc={r['train_minus_val_auc']:.4f}, validation-test auc={r['val_minus_test_auc']:.4f}"
        )

    lines.append("")
    lines.append("Basic overfitting observations:")
    lines.append(
        "- Larger train-vs-validation drops in F1/AUC suggest stronger overfitting; "
        "smaller gaps suggest better generalization."
    )
    lines.append(
        "- This script creates an additional 60/20/20 scenario only; existing 80/20 "
        "results, reports, charts, and saved models were preserved."
    )
    lines.append("")
    lines.append(f"Saved comparison CSV: {OUT_CSV}")
    lines.append(f"Saved confusion matrix figure: {OUT_CM}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Progress: Report saved to {OUT_REPORT}")
    print("Done.")


if __name__ == "__main__":
    main()
