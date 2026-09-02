#!/usr/bin/env python3
"""
Step 1 - Exploratory Data Analysis

This script:

1. Reads the complete raw Parquet dataset.
2. Examines rows, columns and data types.
3. Calculates missing values and duplicate rows.
4. Produces descriptive statistics.
5. Examines Treatment_Outcome class balance.
6. Checks positive and negative infinity values.
7. Creates distribution and correlation charts.

The original dataset is not modified.

Outputs:
- outputs/step1_eda_summary.txt
- outputs/step1_clinical_distributions.png
- outputs/step1_correlation_matrix.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Use a non-interactive plotting backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def convert_bytes_to_mb(number_of_bytes):
    """Convert bytes into megabytes."""
    return float(number_of_bytes) / (1024 ** 2)


def main():
    # ---------------------------------------------------------
    # 1. File paths
    # ---------------------------------------------------------
    source_path = Path(
        "outputs/clinical_data_raw.parquet"
    )

    output_directory = Path("outputs")

    summary_path = output_directory / (
        "step1_eda_summary.txt"
    )

    distributions_path = output_directory / (
        "step1_clinical_distributions.png"
    )

    correlation_path = output_directory / (
        "step1_correlation_matrix.png"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 2. Check that the Parquet file exists
    # ---------------------------------------------------------
    if not source_path.exists():
        raise SystemExit(
            f"Parquet dataset not found: {source_path}"
        )

    # ---------------------------------------------------------
    # 3. Read the complete Parquet dataset
    # ---------------------------------------------------------
    print("Reading the complete Parquet dataset...")

    df = pd.read_parquet(source_path)

    print("Dataset loaded successfully.")

    number_of_rows, number_of_columns = df.shape

    print(
        f"Dataset shape: "
        f"{number_of_rows} rows and "
        f"{number_of_columns} columns"
    )

    # ---------------------------------------------------------
    # 4. Basic dataset information
    # ---------------------------------------------------------
    first_five_rows = df.head(5)
    column_data_types = df.dtypes

    # Calculate full DataFrame memory usage.
    memory_bytes = df.memory_usage(
        deep=True
    ).sum()

    memory_mb = convert_bytes_to_mb(
        memory_bytes
    )

    # ---------------------------------------------------------
    # 5. Missing-value analysis
    # ---------------------------------------------------------
    print("Calculating missing values...")

    missing_counts = df.isna().sum()

    if number_of_rows > 0:
        missing_percentages = (
            missing_counts / number_of_rows
        ) * 100
    else:
        missing_percentages = pd.Series(
            0.0,
            index=df.columns,
        )

    # ---------------------------------------------------------
    # 6. Exact duplicate-row count
    # ---------------------------------------------------------
    print("Checking exact duplicate rows...")

    exact_duplicate_count = int(
        df.duplicated().sum()
    )

    # ---------------------------------------------------------
    # 7. Numeric descriptive statistics
    # ---------------------------------------------------------
    numeric_columns = list(
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    if numeric_columns:
        numeric_description = (
            df[numeric_columns]
            .describe()
            .transpose()
        )
    else:
        numeric_description = pd.DataFrame()

    # ---------------------------------------------------------
    # 8. Treatment outcome balance
    # ---------------------------------------------------------
    # The raw Parquet file uses this exact capitalization.
    target_column = "Treatment_Outcome"

    if target_column not in df.columns:
        raise SystemExit(
            f"Target column not found: {target_column}"
        )

    target_counts = df[
        target_column
    ].value_counts(dropna=False)

    if number_of_rows > 0:
        target_percentages = (
            target_counts / number_of_rows
        ) * 100
    else:
        target_percentages = pd.Series(
            dtype="float64"
        )

    # ---------------------------------------------------------
    # 9. Check infinity values
    # ---------------------------------------------------------
    print("Checking infinity values...")

    positive_infinity_counts = {}
    negative_infinity_counts = {}

    for column in numeric_columns:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        positive_infinity_counts[column] = int(
            np.isposinf(numeric_values).sum()
        )

        negative_infinity_counts[column] = int(
            np.isneginf(numeric_values).sum()
        )

    # ---------------------------------------------------------
    # 10. Write the complete EDA summary
    # ---------------------------------------------------------
    print("Writing EDA summary...")

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(
            "Step 1 - Exploratory Data Analysis Summary\n"
        )

        summary_file.write(
            "=========================================\n\n"
        )

        summary_file.write(
            f"Source file: {source_path}\n\n"
        )

        summary_file.write(
            f"Dataset rows: {number_of_rows}\n"
        )

        summary_file.write(
            f"Dataset columns: {number_of_columns}\n"
        )

        summary_file.write(
            f"DataFrame memory usage: "
            f"{memory_mb:.2f} MB\n\n"
        )

        summary_file.write(
            "Column names and data types:\n"
        )

        for column, data_type in (
            column_data_types.items()
        ):
            summary_file.write(
                f" - {column}: {data_type}\n"
            )

        summary_file.write("\n")

        summary_file.write("First five rows:\n")

        summary_file.write(
            first_five_rows.to_string(
                index=False
            )
        )

        summary_file.write("\n\n")

        summary_file.write(
            "Missing-value counts and percentages:\n"
        )

        for column in df.columns:
            count = int(
                missing_counts[column]
            )

            percentage = float(
                missing_percentages[column]
            )

            summary_file.write(
                f" - {column}: "
                f"missing={count}, "
                f"percentage={percentage:.2f}%\n"
            )

        summary_file.write("\n")

        summary_file.write(
            f"Exact duplicate rows: "
            f"{exact_duplicate_count}\n\n"
        )

        summary_file.write(
            "Numeric descriptive statistics:\n"
        )

        if not numeric_description.empty:
            summary_file.write(
                numeric_description.to_string()
            )
        else:
            summary_file.write(
                "No numeric columns were found."
            )

        summary_file.write("\n\n")

        summary_file.write(
            "Treatment_Outcome counts "
            "and percentages:\n"
        )

        for value, count in (
            target_counts.items()
        ):
            percentage = (
                int(count)
                / number_of_rows
                * 100
            )

            value_label = (
                "<MISSING>"
                if pd.isna(value)
                else str(value)
            )

            summary_file.write(
                f" - {value_label}: "
                f"{int(count)} "
                f"({percentage:.2f}%)\n"
            )

        summary_file.write("\n")

        summary_file.write(
            "Infinity counts by numeric column:\n"
        )

        for column in numeric_columns:
            summary_file.write(
                f" - {column}: "
                f"positive_infinity="
                f"{positive_infinity_counts[column]}, "
                f"negative_infinity="
                f"{negative_infinity_counts[column]}\n"
            )

    print(
        f"EDA summary saved to: {summary_path}"
    )

    # ---------------------------------------------------------
    # 11. Create a sample for charts
    # ---------------------------------------------------------
    # The complete dataset was used for all calculations above.
    # Sampling is used only to make plotting faster.
    sample_size = min(
        number_of_rows,
        100_000,
    )

    chart_sample = df.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    # Replace infinity only inside the chart sample.
    # The original DataFrame is not modified.
    chart_numeric_columns = list(
        chart_sample.select_dtypes(
            include=[np.number]
        ).columns
    )

    for column in chart_numeric_columns:
        chart_sample[column] = (
            chart_sample[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # Do not plot identifier columns.
    identifier_columns = {
        "Patient_ID",
        "patient_id_1",
    }

    plot_columns = [
        column
        for column in chart_numeric_columns
        if column not in identifier_columns
    ]

    # ---------------------------------------------------------
    # 12. Numeric distribution charts
    # ---------------------------------------------------------
    print("Creating distribution charts...")

    if plot_columns:
        chart_columns_per_row = 3

        chart_rows = (
            len(plot_columns)
            + chart_columns_per_row
            - 1
        ) // chart_columns_per_row

        figure, axes = plt.subplots(
            chart_rows,
            chart_columns_per_row,
            figsize=(
                chart_columns_per_row * 5,
                chart_rows * 3,
            ),
        )

        axes_array = np.array(
            axes
        ).reshape(-1)

        for index, column in enumerate(
            plot_columns
        ):
            axis = axes_array[index]

            valid_values = (
                chart_sample[column]
                .dropna()
            )

            if valid_values.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                sns.histplot(
                    valid_values,
                    bins=50,
                    kde=False,
                    ax=axis,
                    color="teal",
                )

            axis.set_title(column)

        # Hide unused chart positions.
        for index in range(
            len(plot_columns),
            len(axes_array),
        ):
            axes_array[index].axis("off")

        figure.tight_layout()

        figure.savefig(
            distributions_path,
            dpi=150,
        )

        plt.close(figure)

        print(
            f"Distribution charts saved to: "
            f"{distributions_path}"
        )
    else:
        print(
            "No numeric columns available "
            "for distribution charts."
        )

    # ---------------------------------------------------------
    # 13. Correlation heatmap
    # ---------------------------------------------------------
    print("Creating correlation heatmap...")

    if len(plot_columns) >= 2:
        correlation_matrix = chart_sample[
            plot_columns
        ].corr()

        figure_width = max(
            10,
            len(plot_columns) * 0.5,
        )

        figure_height = max(
            8,
            len(plot_columns) * 0.5,
        )

        figure, axis = plt.subplots(
            figsize=(
                figure_width,
                figure_height,
            )
        )

        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="RdYlGn",
            center=0,
            ax=axis,
        )

        axis.set_title(
            "Clinical Feature Correlation Matrix"
        )

        figure.tight_layout()

        figure.savefig(
            correlation_path,
            dpi=150,
        )

        plt.close(figure)

        print(
            f"Correlation matrix saved to: "
            f"{correlation_path}"
        )
    else:
        print(
            "Not enough numeric columns "
            "for a correlation heatmap."
        )

    # ---------------------------------------------------------
    # 14. Final confirmation
    # ---------------------------------------------------------
    print("Step 1 EDA completed successfully.")
    print(
        f"Rows analyzed: {number_of_rows}"
    )
    print(
        f"Columns analyzed: {number_of_columns}"
    )
    print(
        "The original Parquet dataset "
        "was not modified."
    )


if __name__ == "__main__":
    main()