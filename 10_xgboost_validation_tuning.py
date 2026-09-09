#!/usr/bin/env python3
"""Validation-based XGBoost tuning for the GSK Medicine Prediction project."""
from pathlib import Path
import joblib
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
from xgboost import XGBClassifier


TARGET = "treatment_outcome"
SOURCE_FILE = Path("outputs/data_engineered.parquet")
RESULTS_CSV = Path("outputs/xgboost_tuning_all_results.csv")
REPORT_FILE = Path("outputs/xgboost_tuning_report.txt")
CURVE_FILE = Path("outputs/xgboost_validation_threshold_curve.png")
CM_FILE = Path("outputs/xgboost_tuned_confusion_matrices.png")
MODEL_FILE = Path("models/xgboost_tuned_pipeline.joblib")


def build_one_hot_encoder():
    """Create a version-compatible OneHotEncoder with sparse output when possible."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def split_summary(name, y_values):
    """Return count text and class percentages for one split."""
    pos_count = int((y_values == 1).sum())
    neg_count = int((y_values == 0).sum())
    total = len(y_values)
    pos_pct = (pos_count / total) * 100 if total else 0.0
    neg_pct = (neg_count / total) * 100 if total else 0.0
    return (
        f"{name}: rows={total:,}, negative(0)={neg_count:,} ({neg_pct:.3f}%), "
        f"positive(1)={pos_count:,} ({pos_pct:.3f}%)"
    )


def metrics_at_threshold(y_true, probabilities, threshold):
    """Calculate binary metrics at a given probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
    }


def make_confusion_matrix_plot(cm_050, cm_selected, threshold_selected):
    """Save two clearly labelled test confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    titles = ["Test at threshold 0.50", f"Test at threshold {threshold_selected:.2f}"]
    matrices = [cm_050, cm_selected]
    labels = ["Ineffective (0)", "Effective (1)"]

    for ax, title, cm in zip(axes, titles, matrices):
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)

    CM_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FILE, dpi=150)
    plt.close(fig)


def make_threshold_curve_plot(result_rows, selected_threshold):
    """Plot validation precision, recall, and F1 across thresholds."""
    result_df = pd.DataFrame(result_rows).sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(result_df["threshold"], result_df["validation_precision"], label="Precision")
    ax.plot(result_df["threshold"], result_df["validation_recall"], label="Recall")
    ax.plot(result_df["threshold"], result_df["validation_f1"], label="F1-score")
    ax.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected threshold = {selected_threshold:.2f}")
    ax.set_title("Validation metrics across thresholds for selected XGBoost model")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0.30, 0.80)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()

    CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CURVE_FILE, dpi=150)
    plt.close(fig)


def main():
    print("Progress: Loading engineered dataset...")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    df = pd.read_parquet(SOURCE_FILE)

    print("Progress: Verifying source dataset shape...")
    expected_shape = (1_050_000, 38)
    if df.shape != expected_shape:
        raise SystemExit(f"Expected source shape {expected_shape}, found {df.shape}.")

    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column: {TARGET}")

    print("Progress: Keeping only labelled rows for supervised tuning...")
    df_labelled = df[df[TARGET].notna()].copy()
    expected_labelled_rows = 985_876
    if len(df_labelled) != expected_labelled_rows:
        raise SystemExit(
            f"Expected {expected_labelled_rows:,} labelled rows, found {len(df_labelled):,}."
        )

    y = pd.to_numeric(df_labelled[TARGET], errors="coerce")
    if y.isna().any():
        raise SystemExit("Target contains non-numeric values after removing missing rows.")
    y = y.astype(int)

    excluded_columns = {
        "patient_id": "identifier",
        TARGET: "target",
        "admission_date": "raw date",
        "adverse_event": "post-treatment outcome/data leakage",
        "readmission_30d": "post-treatment outcome/data leakage",
    }
    predictor_columns = [c for c in df_labelled.columns if c not in excluded_columns]
    if len(predictor_columns) != 33:
        raise SystemExit(
            f"Expected 33 predictor columns, found {len(predictor_columns)}."
        )

    X = df_labelled[predictor_columns].copy()

    print("Progress: Reproducing the stratified 60/20/20 split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(split_summary("Train", y_train))
    print(split_summary("Validation", y_val))
    print(split_summary("Test", y_test))

    print("Progress: Identifying numeric and categorical predictors...")
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in X_train.columns if c not in numeric_columns]

    print("Progress: Fitting preprocessing on training data only...")
    # ColumnTransformer expects estimator objects, so the categorical steps are wrapped in a pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", build_one_hot_encoder()),
                ]),
                categorical_columns,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Progress: Training the baseline XGBoost model...")
    baseline_model = XGBClassifier(
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
    )
    baseline_model.fit(X_train_processed, y_train)
    baseline_val_prob = baseline_model.predict_proba(X_val_processed)[:, 1]
    baseline_val_050 = metrics_at_threshold(y_val, baseline_val_prob, 0.50)

    tuning_configs = [
        {
            "config_number": 1,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "gamma": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2,
            "scale_pos_weight": 2.3,
        },
        {
            "config_number": 2,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 10,
            "gamma": 0.2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 3,
            "scale_pos_weight": 2.3,
        },
        {
            "config_number": 3,
            "n_estimators": 250,
            "learning_rate": 0.07,
            "max_depth": 4,
            "min_child_weight": 10,
            "gamma": 0.2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 3,
            "scale_pos_weight": 1.8,
        },
    ]

    print("Progress: Evaluating tuned XGBoost configurations on validation thresholds...")
    thresholds = np.round(np.arange(0.30, 0.801, 0.01), 2)
    all_results = []
    best_choice = None
    best_model = None
    best_config = None
    selected_train_prob = None
    selected_val_prob = None

    for config in tuning_configs:
        print(f"Progress: Training XGBoost configuration {config['config_number']}...")
        model_params = {k: v for k, v in config.items() if k != "config_number"}
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **model_params,
        )
        model.fit(X_train_processed, y_train)

        train_prob = model.predict_proba(X_train_processed)[:, 1]
        val_prob = model.predict_proba(X_val_processed)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)

        config_rows = []
        for threshold in thresholds:
            validation_metrics = metrics_at_threshold(y_val, val_prob, threshold)
            row = {
                "configuration_number": config["config_number"],
                "threshold": float(threshold),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
                "validation_roc_auc": val_auc,
            }
            all_results.append(row)
            config_rows.append(row)

            qualifies = (
                validation_metrics["precision"] >= 0.50
                and validation_metrics["recall"] >= 0.10
            )

            if best_choice is None:
                best_choice = {**row, "qualifies": qualifies}
                best_model = model
                best_config = config.copy()
                selected_train_prob = train_prob
                selected_val_prob = val_prob
                continue

            current = {**row, "qualifies": qualifies}
            if best_choice["qualifies"]:
                if current["qualifies"] and current["validation_f1"] > best_choice["validation_f1"]:
                    best_choice = current
                    best_model = model
                    best_config = config.copy()
                    selected_train_prob = train_prob
                    selected_val_prob = val_prob
            else:
                if current["qualifies"]:
                    best_choice = current
                    best_model = model
                    best_config = config.copy()
                    selected_train_prob = train_prob
                    selected_val_prob = val_prob
                elif current["validation_f1"] > best_choice["validation_f1"]:
                    best_choice = current
                    best_model = model
                    best_config = config.copy()
                    selected_train_prob = train_prob
                    selected_val_prob = val_prob

    if best_choice is None or best_model is None or best_config is None:
        raise SystemExit("No tuning results were produced.")

    precision_target_achieved = bool(best_choice["qualifies"])
    if not precision_target_achieved:
        print("Progress: No validation choice achieved precision >= 0.50 with recall >= 0.10.")

    print("Progress: Evaluating the selected model on validation, training, and untouched test data...")
    selected_threshold = float(best_choice["threshold"])
    selected_test_prob = best_model.predict_proba(X_test_processed)[:, 1]
    selected_train_metrics = metrics_at_threshold(y_train, selected_train_prob, selected_threshold)
    selected_val_metrics = metrics_at_threshold(y_val, selected_val_prob, selected_threshold)
    test_metrics_050 = metrics_at_threshold(y_test, selected_test_prob, 0.50)
    test_metrics_selected = metrics_at_threshold(y_test, selected_test_prob, selected_threshold)

    print("Progress: Saving tuning results CSV...")
    results_df = pd.DataFrame(all_results)
    baseline_row = pd.DataFrame([
        {
            "configuration_number": 0,
            "threshold": 0.50,
            "validation_accuracy": baseline_val_050["accuracy"],
            "validation_precision": baseline_val_050["precision"],
            "validation_recall": baseline_val_050["recall"],
            "validation_f1": baseline_val_050["f1"],
            "validation_roc_auc": baseline_val_050["roc_auc"],
        }
    ])
    results_df = pd.concat([baseline_row, results_df], ignore_index=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print("Progress: Saving validation threshold curve...")
    selected_curve_plot_rows = [r for r in all_results if r["configuration_number"] == best_config["config_number"]]
    make_threshold_curve_plot(selected_curve_plot_rows, selected_threshold)

    print("Progress: Saving test confusion matrices...")
    make_confusion_matrix_plot(
        test_metrics_050["confusion_matrix"],
        test_metrics_selected["confusion_matrix"],
        selected_threshold,
    )

    print("Progress: Saving the selected tuned pipeline object...")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": best_model,
            "selected_threshold": selected_threshold,
            "excluded_columns": excluded_columns,
            "predictor_columns": predictor_columns,
        },
        MODEL_FILE,
    )

    print("Progress: Writing the tuning report...")
    lines = []
    lines.append("XGBoost Validation Tuning Report")
    lines.append("===============================")
    lines.append(f"Source dataset: {SOURCE_FILE}")
    lines.append(f"Verified source shape: {df.shape}")
    lines.append(f"Labelled rows used: {len(df_labelled):,}")
    lines.append("")
    lines.append("Excluded columns and reasons:")
    for column, reason in excluded_columns.items():
        lines.append(f"- {column}: {reason}")
    lines.append(f"Predictor columns remaining: {len(predictor_columns)}")
    lines.append("")
    lines.append("Split sizes and class distributions:")
    lines.append(f"- {split_summary('Train', y_train)}")
    lines.append(f"- {split_summary('Validation', y_val)}")
    lines.append(f"- {split_summary('Test', y_test)}")
    lines.append("")
    lines.append("Preprocessing explanation:")
    lines.append("- Numeric columns use median imputation.")
    lines.append("- Categorical columns use most-frequent imputation followed by one-hot encoding.")
    lines.append("- Preprocessing was fit only on the training set, then reused for validation and test.")
    lines.append("")
    lines.append("Baseline XGBoost validation results at threshold 0.50:")
    lines.append(
        f"- accuracy={baseline_val_050['accuracy']:.4f}, precision={baseline_val_050['precision']:.4f}, "
        f"recall={baseline_val_050['recall']:.4f}, f1={baseline_val_050['f1']:.4f}, "
        f"roc_auc={baseline_val_050['roc_auc']:.4f}"
    )
    lines.append("")
    lines.append("Tested tuned configurations:")
    for config in tuning_configs:
        lines.append(f"- Configuration {config['config_number']}: {config}")
    lines.append("")
    lines.append(f"Selected configuration: {best_config['config_number']}")
    lines.append(f"Selected threshold: {selected_threshold:.2f}")
    lines.append(f"Precision target achieved? {'Yes' if precision_target_achieved else 'No'}")
    if not precision_target_achieved:
        lines.append("- No candidate reached validation precision >= 0.50 while also keeping recall >= 0.10.")
        lines.append("- The fallback choice is the highest validation F1-score across all tested configurations and thresholds.")
    lines.append("")
    lines.append("Selected model validation results:")
    lines.append(
        f"- accuracy={selected_val_metrics['accuracy']:.4f}, precision={selected_val_metrics['precision']:.4f}, "
        f"recall={selected_val_metrics['recall']:.4f}, f1={selected_val_metrics['f1']:.4f}, "
        f"roc_auc={selected_val_metrics['roc_auc']:.4f}"
    )
    lines.append("")
    lines.append("Untouched test results at threshold 0.50:")
    lines.append(
        f"- accuracy={test_metrics_050['accuracy']:.4f}, precision={test_metrics_050['precision']:.4f}, "
        f"recall={test_metrics_050['recall']:.4f}, f1={test_metrics_050['f1']:.4f}, "
        f"roc_auc={test_metrics_050['roc_auc']:.4f}"
    )
    lines.append(
        f"- confusion matrix [tn, fp, fn, tp]: {test_metrics_050['confusion_matrix'].ravel().tolist()}"
    )
    lines.append("")
    lines.append(f"Untouched test results at selected threshold {selected_threshold:.2f}:")
    lines.append(
        f"- accuracy={test_metrics_selected['accuracy']:.4f}, precision={test_metrics_selected['precision']:.4f}, "
        f"recall={test_metrics_selected['recall']:.4f}, f1={test_metrics_selected['f1']:.4f}, "
        f"roc_auc={test_metrics_selected['roc_auc']:.4f}"
    )
    lines.append(
        f"- confusion matrix [tn, fp, fn, tp]: {test_metrics_selected['confusion_matrix'].ravel().tolist()}"
    )
    lines.append("")
    lines.append("Overfitting comparison between training and validation for the selected model:")
    lines.append(
        f"- Train at selected threshold: accuracy={selected_train_metrics['accuracy']:.4f}, "
        f"precision={selected_train_metrics['precision']:.4f}, recall={selected_train_metrics['recall']:.4f}, "
        f"f1={selected_train_metrics['f1']:.4f}, roc_auc={selected_train_metrics['roc_auc']:.4f}"
    )
    lines.append(
        f"- Validation at selected threshold: accuracy={selected_val_metrics['accuracy']:.4f}, "
        f"precision={selected_val_metrics['precision']:.4f}, recall={selected_val_metrics['recall']:.4f}, "
        f"f1={selected_val_metrics['f1']:.4f}, roc_auc={selected_val_metrics['roc_auc']:.4f}"
    )
    lines.append(
        f"- Train-validation F1 difference: {selected_train_metrics['f1'] - selected_val_metrics['f1']:.4f}"
    )
    lines.append(
        f"- Train-validation ROC-AUC difference: {selected_train_metrics['roc_auc'] - selected_val_metrics['roc_auc']:.4f}"
    )
    lines.append("")
    lines.append("Plain-language conclusion:")
    lines.append(
        "- Raising the decision threshold usually improves precision but often lowers recall. "
        "This tuning process uses validation data only, so the final test evaluation remains fair."
    )
    lines.append(
        "- The selected setting aims to improve precision while still keeping some ability to catch effective cases."
    )
    lines.append("")
    lines.append(f"Saved CSV: {RESULTS_CSV}")
    lines.append(f"Saved report: {REPORT_FILE}")
    lines.append(f"Saved threshold curve: {CURVE_FILE}")
    lines.append(f"Saved confusion matrices: {CM_FILE}")
    lines.append(f"Saved tuned pipeline object: {MODEL_FILE}")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Progress: Report written to {REPORT_FILE}")
    print("Done. Review the script before running it.")


if __name__ == "__main__":
    main()
