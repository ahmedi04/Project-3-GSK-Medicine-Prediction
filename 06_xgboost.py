#!/usr/bin/env python3
"""
Step 6A - XGBoost Classifier

Loads the engineered Parquet dataset, removes rows with missing
treatment_outcome, applies leakage-safe preprocessing and trains an
XGBoost classifier using the same 80/20 split as the previous models.

Creates:
 - outputs/step6_xgboost_report.txt
 - outputs/step6_xgboost_confusion_matrix.png
 - outputs/step6_xgboost_feature_importance.png
 - models/xgboost_pipeline.joblib
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
from xgboost import XGBClassifier


def ensure_directories():
    """Create folders needed for reports, charts and models."""
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def calculate_metrics(y_true, y_prediction, y_probability):
    """Calculate the main classification metrics."""
    try:
        roc_auc = roc_auc_score(y_true, y_probability)
    except Exception:
        roc_auc = float("nan")

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
        "roc_auc": roc_auc,
    }


def main():
    ensure_directories()

    # Input and output paths
    source_path = Path("outputs/data_engineered.parquet")
    report_path = Path("outputs/step6_xgboost_report.txt")
    confusion_matrix_path = Path(
        "outputs/step6_xgboost_confusion_matrix.png"
    )
    feature_importance_path = Path(
        "outputs/step6_xgboost_feature_importance.png"
    )
    model_path = Path("models/xgboost_pipeline.joblib")

    # Confirm that the input dataset exists
    print("Checking engineered dataset exists...")

    if not source_path.exists():
        raise SystemExit(f"Input file not found: {source_path}")

    # Load the complete dataset without using chunks
    print("Loading engineered Parquet dataset...")
    dataframe = pd.read_parquet(source_path)

    total_rows = len(dataframe)
    print(f"Total rows loaded: {total_rows:,}")

    # Confirm the target column exists
    if "treatment_outcome" not in dataframe.columns:
        raise SystemExit(
            "The treatment_outcome column was not found."
        )

    # Remove only rows where treatment_outcome is missing
    print("Removing rows with missing treatment_outcome...")

    missing_target_mask = dataframe[
        "treatment_outcome"
    ].isna()

    missing_target_count = int(
        missing_target_mask.sum()
    )

    labelled_data = dataframe.loc[
        ~missing_target_mask
    ].copy()

    labelled_rows = len(labelled_data)

    print(
        f"{missing_target_count:,} missing-target rows removed"
    )
    print(f"{labelled_rows:,} labelled rows used")

    # Convert the target into integers
    labelled_data["treatment_outcome"] = (
        labelled_data["treatment_outcome"].astype(int)
    )

    y = labelled_data["treatment_outcome"]

    # Confirm the target contains exactly 0 and 1
    target_values = set(y.unique())

    if target_values != {0, 1}:
        raise SystemExit(
            "treatment_outcome must contain exactly classes "
            f"0 and 1. Found: {sorted(target_values)}"
        )

    # Exclude identifiers, dates and post-treatment outcomes
    # to prevent data leakage
    excluded_columns = [
        "patient_id",
        "treatment_outcome",
        "admission_date",
        "adverse_event",
        "readmission_30d",
    ]

    X = labelled_data.drop(
        columns=[
            column
            for column in excluded_columns
            if column in labelled_data.columns
        ]
    )

    # Confirm that excluded columns are not predictors
    excluded_still_present = [
        column
        for column in excluded_columns
        if column in X.columns
    ]

    if excluded_still_present:
        raise SystemExit(
            "Excluded columns are still present in predictors: "
            f"{excluded_still_present}"
        )

    # Identify numeric and categorical predictors
    numeric_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Check for unsupported predictor data types
    recognized_columns = set(
        numeric_columns + categorical_columns
    )

    unsupported_columns = [
        column
        for column in X.columns
        if column not in recognized_columns
    ]

    if unsupported_columns:
        raise SystemExit(
            "Unsupported predictor data types found: "
            f"{unsupported_columns}"
        )

    print(f"Numeric predictors: {len(numeric_columns)}")
    print(
        f"Categorical predictors: "
        f"{len(categorical_columns)}"
    )

    # Use the same split as the previous models
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

    # Confirm both target classes exist
    if set(y_train.unique()) != {0, 1}:
        raise SystemExit(
            "The training set does not contain both classes."
        )

    if set(y_test.unique()) != {0, 1}:
        raise SystemExit(
            "The testing set does not contain both classes."
        )

    # Count training target classes
    training_class_counts = y_train.value_counts()

    negative_count = int(
        training_class_counts.get(0, 0)
    )

    positive_count = int(
        training_class_counts.get(1, 0)
    )

    if positive_count == 0:
        raise SystemExit(
            "Training data contains no positive target records."
        )

    # Use the exact scale_pos_weight provided in the PDF
    scale_pos_weight = 2.3

    print(f"Training class 0 count: {negative_count:,}")
    print(f"Training class 1 count: {positive_count:,}")
    print(
        f"PDF-provided scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # Fill missing numeric predictors using training medians
    numeric_transformer = SimpleImputer(
        strategy="median"
    )

    # Fill categorical gaps and one-hot encode categories
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
                    sparse_output=True,
                    dtype=np.float32,
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
        sparse_threshold=1.0,
    )

    # XGBoost settings from the project PDF
    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.3,
        random_state=42,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
    )

    # Keep preprocessing and model training together
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Fit preprocessing and XGBoost only on training data
    print("Training the XGBoost model...")
    print("This may take several minutes.")

    pipeline.fit(X_train, y_train)

    # Generate predictions and probabilities
    print("Generating training and testing predictions...")

    y_train_prediction = pipeline.predict(X_train)
    y_test_prediction = pipeline.predict(X_test)

    y_train_probability = pipeline.predict_proba(
        X_train
    )[:, 1]

    y_test_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Confirm prediction count
    if len(y_test_prediction) != len(y_test):
        raise SystemExit(
            "Prediction count does not match test rows."
        )

    # Calculate training and testing metrics
    print("Calculating training metrics...")

    training_metrics = calculate_metrics(
        y_train,
        y_train_prediction,
        y_train_probability,
    )

    print("Calculating testing metrics...")

    testing_metrics = calculate_metrics(
        y_test,
        y_test_prediction,
        y_test_probability,
    )

    accuracy_difference = (
        training_metrics["accuracy"]
        - testing_metrics["accuracy"]
    )

    # Create confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_test_prediction,
        labels=[0, 1],
    )

    # Extract processed feature names
    print("Extracting feature names...")

    fitted_preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    try:
        processed_feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )
    except Exception as error:
        raise SystemExit(
            f"Could not obtain feature names: {error}"
        )

    feature_names = [
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        for name in processed_feature_names
    ]

    feature_importances = pipeline.named_steps[
        "classifier"
    ].feature_importances_

    if len(feature_names) != len(feature_importances):
        raise SystemExit(
            "Feature names and importance counts do not match."
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

    # Save feature-importance chart
    print("Saving feature-importance chart...")

    number_to_plot = min(
        15,
        len(sorted_importances),
    )

    if number_to_plot > 0:
        top_features = sorted_importances[
            :number_to_plot
        ]

        chart_names = [
            name
            for name, value in reversed(top_features)
        ]

        chart_values = [
            value
            for name, value in reversed(top_features)
        ]

        plt.figure(
            figsize=(
                9,
                max(4, number_to_plot * 0.35),
            )
        )

        plt.barh(
            chart_names,
            chart_values,
            color="darkorange",
        )

        plt.title(
            f"Top {number_to_plot} "
            "XGBoost Feature Importances"
        )

        plt.xlabel("Importance")
        plt.tight_layout()

        plt.savefig(
            feature_importance_path,
            dpi=150,
        )

        plt.close()

    # Save confusion-matrix chart
    print("Saving confusion-matrix chart...")

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap=plt.cm.Oranges,
    )

    plt.title(
        "XGBoost Confusion Matrix - Test Data"
    )

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
                format(
                    matrix[row, column],
                    "d",
                ),
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

    plt.savefig(
        confusion_matrix_path,
        dpi=150,
    )

    plt.close()

    # Save the complete trained pipeline
    print("Saving trained XGBoost pipeline...")

    joblib.dump(
        pipeline,
        model_path,
    )

    # Write the XGBoost report
    print("Writing XGBoost report...")

    target_counts = y.value_counts()

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("Step 6A - XGBoost Report\n")
        report.write("================================\n\n")

        report.write(
            f"Source file: {source_path}\n"
        )

        report.write(
            f"Rows loaded: {total_rows:,}\n"
        )

        report.write(
            f"Missing-target rows removed: "
            f"{missing_target_count:,}\n"
        )

        report.write(
            f"Labelled rows used: "
            f"{labelled_rows:,}\n\n"
        )

        report.write("Target distribution:\n")

        for target_value in [0, 1]:
            count = int(
                target_counts.get(
                    target_value,
                    0,
                )
            )

            percentage = (
                count / labelled_rows * 100
                if labelled_rows > 0
                else 0
            )

            report.write(
                f" - {target_value}: "
                f"{count:,} "
                f"({percentage:.2f}%)\n"
            )

        report.write("\nExcluded columns:\n")

        for column in excluded_columns:
            report.write(f" - {column}\n")

        report.write("\n")

        report.write(
            f"Numeric predictors: "
            f"{len(numeric_columns)}\n"
        )

        report.write(
            f"Categorical predictors: "
            f"{len(categorical_columns)}\n\n"
        )

        report.write(
            f"Training shape: "
            f"{X_train.shape}\n"
        )

        report.write(
            f"Testing shape: "
            f"{X_test.shape}\n\n"
        )

        report.write("Training class counts:\n")

        report.write(
            f" - Class 0: {negative_count:,}\n"
        )

        report.write(
            f" - Class 1: {positive_count:,}\n"
        )

        report.write(
            "PDF-provided scale_pos_weight: "
            f"{scale_pos_weight:.6f}\n\n"
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

        for (
            metric_name,
            metric_value,
        ) in training_metrics.items():
            report.write(
                f" - {metric_name}: "
                f"{metric_value:.4f}\n"
            )

        report.write("\nTesting metrics:\n")

        for (
            metric_name,
            metric_value,
        ) in testing_metrics.items():
            report.write(
                f" - {metric_name}: "
                f"{metric_value:.4f}\n"
            )

        report.write(
            "\nClassification report - Test data:\n"
        )

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

        report.write(
            "\nConfusion matrix - Test data:\n"
        )

        report.write(
            np.array2string(matrix)
        )

        report.write("\n\n")

        report.write(
            "Training/testing accuracy difference: "
            f"{accuracy_difference:.6f}\n\n"
        )

        report.write(
            "Top 15 feature importances:\n"
        )

        for (
            feature_name,
            importance,
        ) in sorted_importances[:15]:
            report.write(
                f" - {feature_name}: "
                f"{importance:.6f}\n"
            )

        report.write("\nNotes:\n")

        report.write(
            "- The same train/test split settings used "
            "for the other models were used.\n"
        )

        report.write(
            "- scale_pos_weight=2.3 was taken "
            "from the project PDF.\n"
        )

        report.write(
            "- Preprocessing was fitted only on "
            "training data to prevent data leakage.\n"
        )

        report.write(
            "- ID, date and post-treatment outcome "
            "columns were excluded from training.\n"
        )

    print("Step 6A XGBoost completed successfully.")
    print(f"Report saved to: {report_path}")

    print(
        "Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    print(
        "Feature importance saved to: "
        f"{feature_importance_path}"
    )

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()