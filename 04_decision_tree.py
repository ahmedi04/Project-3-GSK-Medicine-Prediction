
#!/usr/bin/env python3
"""
Step 4 - Decision Tree baseline

Loads the engineered dataset, removes rows with missing treatment_outcome,
builds a leakage-safe preprocessing pipeline fitted on training data only,
trains a Decision Tree classifier, evaluates results, and saves the report,
plots and trained pipeline.
"""

from pathlib import Path
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def ensure_dirs():
    """Create folders for reports, charts and models."""
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def main():
    ensure_dirs()

    # Input and output paths
    src = Path("outputs/data_engineered.parquet")
    report_path = Path("outputs/step4_decision_tree_report.txt")
    cm_path = Path("outputs/step4_decision_tree_confusion_matrix.png")
    fi_path = Path("outputs/step4_decision_tree_feature_importance.png")
    model_path = Path("models/decision_tree_pipeline.joblib")

    print("Checking engineered dataset exists...")

    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")

    # Load the complete Parquet dataset
    print("Loading engineered dataset...")
    df = pd.read_parquet(src)

    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    # Confirm that the target column exists
    if "treatment_outcome" not in df.columns:
        raise SystemExit("treatment_outcome column was not found.")

    # Remove only rows where the target is missing
    print("Removing rows with missing treatment_outcome...")

    missing_mask = df["treatment_outcome"].isna()
    missing_count = int(missing_mask.sum())

    df_labelled = df.loc[~missing_mask].copy()
    labelled_rows = len(df_labelled)

    print(f"{missing_count:,} missing-target rows removed")
    print(f"{labelled_rows:,} labelled rows used")

    # Convert the target into integer values
    df_labelled["treatment_outcome"] = (
        df_labelled["treatment_outcome"].astype(int)
    )

    y = df_labelled["treatment_outcome"]

    # Target must contain exactly classes 0 and 1
    unique_values = set(y.unique())

    if unique_values != {0, 1}:
        raise SystemExit(
            "treatment_outcome must contain exactly classes 0 and 1. "
            f"Found: {sorted(unique_values)}"
        )

    # Exclude ID, target, date and post-treatment outcome columns.
    # These columns could cause data leakage or are not useful predictors.
    excluded_columns = [
        "patient_id",
        "treatment_outcome",
        "admission_date",
        "adverse_event",
        "readmission_30d",
    ]

    X = df_labelled.drop(
        columns=[
            column
            for column in excluded_columns
            if column in df_labelled.columns
        ]
    )

    # Confirm excluded columns are not present in the predictors
    present_excluded = [
        column for column in excluded_columns if column in X.columns
    ]

    if present_excluded:
        raise SystemExit(
            "Excluded columns are still present in the predictors: "
            f"{present_excluded}"
        )

    # Automatically identify numeric and categorical predictors
    numeric_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Make sure no unsupported columns were silently left out
    recognized_columns = set(numeric_columns + categorical_columns)
    unsupported_columns = [
        column for column in X.columns
        if column not in recognized_columns
    ]

    if unsupported_columns:
        raise SystemExit(
            "Unsupported predictor data types found in columns: "
            f"{unsupported_columns}"
        )

    print(f"Numeric predictors: {len(numeric_columns)}")
    print(f"Categorical predictors: {len(categorical_columns)}")

    # Create an 80% training and 20% testing split.
    # Stratify keeps the target class percentages similar in both sets.
    print("Splitting data into train and test sets (80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    # Confirm both target classes exist in both datasets
    if set(y_train.unique()) != {0, 1}:
        raise SystemExit(
            "The training set does not contain both target classes."
        )

    if set(y_test.unique()) != {0, 1}:
        raise SystemExit(
            "The testing set does not contain both target classes."
        )

    # Numeric missing values are filled with training medians
    numeric_transformer = SimpleImputer(strategy="median")

    # Categorical missing values are filled with the most common value.
    # One-hot encoding converts categories into numeric dummy columns.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    # Apply the correct preprocessing to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    # Create the Decision Tree using the provided project parameters
    classifier = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )

    # The pipeline keeps preprocessing and modelling together
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Fit preprocessing and the model only on training data
    print("Training the Decision Tree model...")
    pipeline.fit(X_train, y_train)

    # Generate predictions
    print("Generating predictions...")

    y_train_prediction = pipeline.predict(X_train)
    y_test_prediction = pipeline.predict(X_test)

    y_train_probability = pipeline.predict_proba(X_train)[:, 1]
    y_test_probability = pipeline.predict_proba(X_test)[:, 1]

    if len(y_test_prediction) != len(y_test):
        raise SystemExit(
            "The number of predictions does not match the test rows."
        )

    # Function used to calculate evaluation metrics
    def evaluate(y_true, y_prediction, y_probability):
        return {
            "accuracy": accuracy_score(y_true, y_prediction),
            "precision": precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_true,
                y_probability,
            ),
        }

    print("Calculating training metrics...")
    training_metrics = evaluate(
        y_train,
        y_train_prediction,
        y_train_probability,
    )

    print("Calculating testing metrics...")
    testing_metrics = evaluate(
        y_test,
        y_test_prediction,
        y_test_probability,
    )

    accuracy_difference = (
        training_metrics["accuracy"]
        - testing_metrics["accuracy"]
    )

    # Create the confusion matrix in fixed 0, 1 order
    matrix = confusion_matrix(
        y_test,
        y_test_prediction,
        labels=[0, 1],
    )

    # Extract feature names after preprocessing
    print("Extracting feature names...")

    fitted_preprocessor = pipeline.named_steps["preprocessor"]

    try:
        processed_feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )
    except Exception as error:
        raise SystemExit(
            f"Could not obtain processed feature names: {error}"
        )

    feature_names = [
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        for name in processed_feature_names
    ]

    feature_importances = (
        pipeline.named_steps["classifier"].feature_importances_
    )

    if len(feature_importances) != len(feature_names):
        raise SystemExit(
            "The number of feature importances does not match "
            "the number of feature names."
        )

    importance_results = list(
        zip(
            feature_names,
            feature_importances.tolist(),
        )
    )

    sorted_importances = sorted(
        importance_results,
        key=lambda item: item[1],
        reverse=True,
    )

    # Create the feature-importance chart
    print("Saving feature-importance chart...")

    number_to_plot = min(15, len(sorted_importances))

    if number_to_plot > 0:
        top_features = sorted_importances[:number_to_plot]

        chart_names = [
            name for name, value in reversed(top_features)
        ]

        chart_values = [
            value for name, value in reversed(top_features)
        ]

        plt.figure(
            figsize=(9, max(4, number_to_plot * 0.35))
        )

        plt.barh(
            chart_names,
            chart_values,
            color="steelblue",
        )

        plt.title(
            f"Top {number_to_plot} Decision Tree Feature Importances"
        )

        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()

    # Create the confusion-matrix chart
    print("Saving confusion-matrix chart...")

    plt.figure(figsize=(6, 5))
    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    plt.title("Decision Tree Confusion Matrix - Test Data")
    plt.colorbar()

    tick_positions = np.arange(2)

    plt.xticks(
        tick_positions,
        ["Ineffective (0)", "Effective (1)"],
    )

    plt.yticks(
        tick_positions,
        ["Ineffective (0)", "Effective (1)"],
    )

    threshold = matrix.max() / 2

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                format(matrix[row, column], "d"),
                horizontalalignment="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    plt.ylabel("Actual outcome")
    plt.xlabel("Predicted outcome")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Save the complete trained pipeline
    print("Saving trained Decision Tree pipeline...")
    joblib.dump(pipeline, model_path)

    # Write a readable text report
    print("Writing Decision Tree report...")

    target_counts = y.value_counts()

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("Step 4 - Decision Tree Report\n")
        report.write("================================\n\n")

        report.write(f"Source file: {src}\n")
        report.write(f"Rows loaded: {total_rows:,}\n")
        report.write(
            f"Missing-target rows removed: {missing_count:,}\n"
        )
        report.write(f"Labelled rows used: {labelled_rows:,}\n\n")

        report.write("Target distribution:\n")

        for target_value in [0, 1]:
            count = int(target_counts.get(target_value, 0))
            percentage = (
                count / labelled_rows * 100
                if labelled_rows > 0
                else 0
            )

            report.write(
                f" - {target_value}: "
                f"{count:,} ({percentage:.2f}%)\n"
            )

        report.write("\nExcluded columns:\n")

        for column in excluded_columns:
            report.write(f" - {column}\n")

        report.write("\n")

        report.write(
            f"Numeric predictors: {len(numeric_columns)}\n"
        )

        report.write(
            f"Categorical predictors: "
            f"{len(categorical_columns)}\n\n"
        )

        report.write(
            f"Training shape: {X_train.shape}\n"
        )

        report.write(
            f"Testing shape: {X_test.shape}\n\n"
        )

        report.write("Model parameters:\n")
        report.write(
            json.dumps(
                classifier.get_params(),
                indent=2,
            )
        )
        report.write("\n\n")

        report.write("Training metrics:\n")

        for metric_name, metric_value in training_metrics.items():
            report.write(
                f" - {metric_name}: {metric_value:.4f}\n"
            )

        report.write("\nTesting metrics:\n")

        for metric_name, metric_value in testing_metrics.items():
            report.write(
                f" - {metric_name}: {metric_value:.4f}\n"
            )

        report.write("\nClassification report - Test data:\n")

        report.write(
            classification_report(
                y_test,
                y_test_prediction,
                labels=[0, 1],
                target_names=[
                    "Ineffective (0)",
                    "Effective (1)",
                ],
                zero_division=0,
            )
        )

        report.write("\nConfusion matrix - Test data:\n")
        report.write(np.array2string(matrix))
        report.write("\n\n")

        report.write(
            "Training/testing accuracy difference: "
            f"{accuracy_difference:.6f}\n\n"
        )

        report.write("Top 15 feature importances:\n")

        for feature_name, importance in sorted_importances[:15]:
            report.write(
                f" - {feature_name}: {importance:.6f}\n"
            )

        report.write("\nNotes:\n")
        report.write(
            "- Balanced class weights were used because "
            "the target is imbalanced.\n"
        )
        report.write(
            "- Preprocessing was fitted only on training data "
            "to prevent data leakage.\n"
        )
        report.write(
            "- ID, date and post-treatment outcome columns "
            "were excluded from model training.\n"
        )

    print("Step 4 Decision Tree completed successfully.")
    print(f"Report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Feature importance saved to: {fi_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()