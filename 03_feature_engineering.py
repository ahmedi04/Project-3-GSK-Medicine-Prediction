#!/usr/bin/env python3
"""
Step 3 - Feature Engineering

Reads:
    outputs/data_cleaned.parquet

Creates seven new clinical features:
    1. kidney_stage
    2. bmi_category
    3. age_group
    4. liver_risk
    5. polypharmacy
    6. elderly_high_dose
    7. de_ritis_ratio

Saves:
    outputs/data_engineered.parquet
    outputs/step3_feature_engineering_report.txt

The script processes the complete Parquet dataset without chunking.
It does not perform one-hot encoding or feature scaling yet.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # ---------------------------------------------------------
    # File paths
    # ---------------------------------------------------------
    source_path = Path("outputs/data_cleaned.parquet")
    output_path = Path("outputs/data_engineered.parquet")
    report_path = Path("outputs/step3_feature_engineering_report.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check that the cleaned dataset exists
    if not source_path.exists():
        raise SystemExit(
            f"Input file not found: {source_path}\n"
            "Run 02_data_cleaning.py first."
        )

    # ---------------------------------------------------------
    # Load the complete cleaned dataset
    # ---------------------------------------------------------
    print("Reading cleaned Parquet dataset...")
    df = pd.read_parquet(source_path)

    input_rows, input_columns = df.shape
    print(
        f"Dataset loaded: {input_rows} rows and "
        f"{input_columns} columns"
    )

    # Step 3 expects the 31-column cleaned dataset
    if input_columns != 31:
        raise SystemExit(
            f"Expected 31 input columns, but found {input_columns}."
        )

    # Columns required for feature engineering
    required_columns = [
        "patient_id",
        "age",
        "bmi",
        "egfr",
        "alt_enzyme",
        "ast_enzyme",
        "concurrent_drugs",
        "dosage_mg",
        "treatment_outcome",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise SystemExit(
            f"Required columns are missing: {missing_columns}"
        )

    # Verify patient IDs before feature engineering
    if not df["patient_id"].is_unique:
        raise SystemExit(
            "patient_id is not unique in the cleaned dataset."
        )

    # Work on a copy
    df_engineered = df.copy()

    # ---------------------------------------------------------
    # 1. Kidney stage from eGFR
    # ---------------------------------------------------------
    print("Creating kidney_stage...")

    kidney_conditions = [
        df_engineered["egfr"] >= 90,
        (df_engineered["egfr"] >= 60)
        & (df_engineered["egfr"] < 90),
        (df_engineered["egfr"] >= 30)
        & (df_engineered["egfr"] < 60),
        df_engineered["egfr"] < 30,
    ]

    kidney_labels = [
        "Normal",
        "Mild",
        "Moderate",
        "Severe",
    ]

    df_engineered["kidney_stage"] = np.select(
        kidney_conditions,
        kidney_labels,
        default="Unknown",
    )

    # ---------------------------------------------------------
    # 2. BMI category
    # ---------------------------------------------------------
    print("Creating bmi_category...")

    bmi_conditions = [
        df_engineered["bmi"] < 18.5,
        (df_engineered["bmi"] >= 18.5)
        & (df_engineered["bmi"] < 25),
        (df_engineered["bmi"] >= 25)
        & (df_engineered["bmi"] < 30),
        df_engineered["bmi"] >= 30,
    ]

    bmi_labels = [
        "Underweight",
        "Normal",
        "Overweight",
        "Obese",
    ]

    df_engineered["bmi_category"] = np.select(
        bmi_conditions,
        bmi_labels,
        default="Unknown",
    )

    # ---------------------------------------------------------
    # 3. Age group
    # ---------------------------------------------------------
    print("Creating age_group...")

    age_conditions = [
        df_engineered["age"] < 18,
        (df_engineered["age"] >= 18)
        & (df_engineered["age"] < 35),
        (df_engineered["age"] >= 35)
        & (df_engineered["age"] < 65),
        (df_engineered["age"] >= 65)
        & (df_engineered["age"] < 80),
        df_engineered["age"] >= 80,
    ]

    age_labels = [
        "Pediatric",
        "Young Adult",
        "Middle Aged",
        "Senior",
        "Elderly",
    ]

    df_engineered["age_group"] = np.select(
        age_conditions,
        age_labels,
        default="Unknown",
    )

    # ---------------------------------------------------------
    # 4. Liver-risk flag
    # ---------------------------------------------------------
    print("Creating liver_risk...")

    df_engineered["liver_risk"] = (
        (df_engineered["alt_enzyme"] > 40)
        | (df_engineered["ast_enzyme"] > 40)
    ).astype("int8")

    # ---------------------------------------------------------
    # 5. Polypharmacy flag
    # ---------------------------------------------------------
    print("Creating polypharmacy...")

    df_engineered["polypharmacy"] = (
        df_engineered["concurrent_drugs"] >= 5
    ).astype("int8")

    # ---------------------------------------------------------
    # 6. Elderly high-dose flag
    # ---------------------------------------------------------
    print("Creating elderly_high_dose...")

    dosage_median = df_engineered["dosage_mg"].median(
        skipna=True
    )

    if pd.isna(dosage_median):
        raise SystemExit(
            "Cannot calculate the dosage_mg median."
        )

    dosage_median = float(dosage_median)

    df_engineered["elderly_high_dose"] = (
        (df_engineered["age"] >= 65)
        & (df_engineered["dosage_mg"] > dosage_median)
    ).astype("int8")

    # ---------------------------------------------------------
    # 7. De Ritis ratio: AST divided by ALT
    # ---------------------------------------------------------
    print("Creating de_ritis_ratio...")

    alt = pd.to_numeric(
        df_engineered["alt_enzyme"],
        errors="coerce",
    )

    ast = pd.to_numeric(
        df_engineered["ast_enzyme"],
        errors="coerce",
    )

    # ALT must be greater than zero because division by zero
    # would make the ratio undefined.
    valid_alt = alt > 0

    de_ritis_ratio = pd.Series(
        np.nan,
        index=df_engineered.index,
        dtype="float64",
    )

    de_ritis_ratio.loc[valid_alt] = (
        ast.loc[valid_alt] / alt.loc[valid_alt]
    )

    # Remove any positive or negative infinity values
    de_ritis_ratio = de_ritis_ratio.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Calculate the median using only valid ratios
    valid_ratios = de_ritis_ratio.dropna()

    if valid_ratios.empty:
        raise SystemExit(
            "No valid De Ritis ratios were available."
        )

    ratio_median_before_capping = float(
        valid_ratios.median()
    )

    # Fill undefined ratios using the valid median
    de_ritis_ratio = de_ritis_ratio.fillna(
        ratio_median_before_capping
    )

    # ---------------------------------------------------------
    # Cap extreme De Ritis ratios with the IQR method
    # ---------------------------------------------------------
    ratio_q1 = float(de_ritis_ratio.quantile(0.25))
    ratio_q3 = float(de_ritis_ratio.quantile(0.75))
    ratio_iqr = ratio_q3 - ratio_q1

    original_lower_limit = (
        ratio_q1 - (1.5 * ratio_iqr)
    )
    upper_limit = ratio_q3 + (1.5 * ratio_iqr)

    # The ratio cannot be negative
    lower_limit = max(0.0, original_lower_limit)

    ratios_below_limit = int(
        (de_ritis_ratio < lower_limit).sum()
    )
    ratios_above_limit = int(
        (de_ritis_ratio > upper_limit).sum()
    )

    de_ritis_ratio = de_ritis_ratio.clip(
        lower=lower_limit,
        upper=upper_limit,
    )

    df_engineered["de_ritis_ratio"] = (
        de_ritis_ratio.astype("float64")
    )

    # Statistics after capping
    ratio_min_after = float(
        df_engineered["de_ritis_ratio"].min()
    )
    ratio_max_after = float(
        df_engineered["de_ritis_ratio"].max()
    )
    ratio_median_after = float(
        df_engineered["de_ritis_ratio"].median()
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------
    print("Validating the engineered dataset...")

    new_columns = [
        "kidney_stage",
        "bmi_category",
        "age_group",
        "liver_risk",
        "polypharmacy",
        "elderly_high_dose",
        "de_ritis_ratio",
    ]

    missing_new_columns = [
        column
        for column in new_columns
        if column not in df_engineered.columns
    ]

    if missing_new_columns:
        raise SystemExit(
            f"New features are missing: {missing_new_columns}"
        )

    output_rows, output_columns = df_engineered.shape

    if output_rows != input_rows:
        raise SystemExit(
            f"Row count changed from {input_rows} "
            f"to {output_rows}."
        )

    if output_columns != 38:
        raise SystemExit(
            f"Expected 38 output columns, "
            f"but found {output_columns}."
        )

    if not df_engineered["patient_id"].is_unique:
        raise SystemExit(
            "patient_id is no longer unique."
        )

    if df_engineered["de_ritis_ratio"].isna().any():
        raise SystemExit(
            "Missing values remain in de_ritis_ratio."
        )

    # Count missing target values.
    # These rows are preserved until model training.
    missing_target_count = int(
        df_engineered["treatment_outcome"]
        .isna()
        .sum()
    )

    # ---------------------------------------------------------
    # Save engineered Parquet dataset
    # ---------------------------------------------------------
    print("Saving engineered Parquet dataset...")

    df_engineered.to_parquet(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Create feature-engineering report
    # ---------------------------------------------------------
    print("Writing feature-engineering report...")

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "Step 3 - Feature Engineering Report\n"
        )
        report.write(
            "===================================\n\n"
        )

        report.write(
            f"Source file: {source_path}\n"
        )
        report.write(
            f"Output file: {output_path}\n\n"
        )

        report.write(
            f"Input shape: {input_rows} rows, "
            f"{input_columns} columns\n"
        )
        report.write(
            f"Output shape: {output_rows} rows, "
            f"{output_columns} columns\n\n"
        )

        report.write(
            f"Global dosage_mg median: "
            f"{dosage_median}\n\n"
        )

        report.write(
            "Kidney-stage category counts:\n"
        )
        report.write(
            df_engineered["kidney_stage"]
            .value_counts(dropna=False)
            .to_string()
        )
        report.write("\n\n")

        report.write(
            "BMI-category counts:\n"
        )
        report.write(
            df_engineered["bmi_category"]
            .value_counts(dropna=False)
            .to_string()
        )
        report.write("\n\n")

        report.write(
            "Age-group counts:\n"
        )
        report.write(
            df_engineered["age_group"]
            .value_counts(dropna=False)
            .to_string()
        )
        report.write("\n\n")

        report.write("Binary feature counts:\n")
        report.write(
            " - liver_risk=1: "
            f"{int(df_engineered['liver_risk'].sum())}\n"
        )
        report.write(
            " - polypharmacy=1: "
            f"{int(df_engineered['polypharmacy'].sum())}\n"
        )
        report.write(
            " - elderly_high_dose=1: "
            f"{int(df_engineered['elderly_high_dose'].sum())}\n\n"
        )

        report.write(
            "De Ritis ratio statistics:\n"
        )
        report.write(
            f" - valid median before capping: "
            f"{ratio_median_before_capping}\n"
        )
        report.write(
            f" - Q1: {ratio_q1}\n"
        )
        report.write(
            f" - Q3: {ratio_q3}\n"
        )
        report.write(
            f" - IQR: {ratio_iqr}\n"
        )
        report.write(
            f" - original lower IQR limit: "
            f"{original_lower_limit}\n"
        )
        report.write(
            f" - final non-negative lower limit: "
            f"{lower_limit}\n"
        )
        report.write(
            f" - upper IQR limit: {upper_limit}\n"
        )
        report.write(
            f" - ratios below lower limit: "
            f"{ratios_below_limit}\n"
        )
        report.write(
            f" - ratios above upper limit: "
            f"{ratios_above_limit}\n"
        )
        report.write(
            f" - total ratios capped: "
            f"{ratios_below_limit + ratios_above_limit}\n"
        )
        report.write(
            f" - median after capping: "
            f"{ratio_median_after}\n"
        )
        report.write(
            f" - minimum after capping: "
            f"{ratio_min_after}\n"
        )
        report.write(
            f" - maximum after capping: "
            f"{ratio_max_after}\n\n"
        )

        report.write(
            f"Missing treatment_outcome values preserved: "
            f"{missing_target_count}\n"
        )
        report.write(
            "These rows must be removed before supervised "
            "model training.\n\n"
        )

        report.write(
            f"patient_id unique after feature engineering: "
            f"{df_engineered['patient_id'].is_unique}\n"
        )
        report.write(
            "No rows were removed during feature engineering.\n"
        )
        report.write(
            "One-hot encoding and feature scaling were not "
            "performed to avoid data leakage before splitting.\n"
        )

    print("Step 3 feature engineering completed successfully.")
    print(
        f"Rows: {output_rows}"
    )
    print(
        f"Columns: {output_columns}"
    )
    print(
        f"Engineered dataset saved to: {output_path}"
    )
    print(
        f"Report saved to: {report_path}"
    )


if __name__ == "__main__":
    main()