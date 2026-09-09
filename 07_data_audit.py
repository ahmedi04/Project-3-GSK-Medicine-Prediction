"""
Audit script for dataset lineage and row/column checks.

Do NOT run this script automatically. It reads parquet metadata
and only selected columns with pandas to validate shapes, duplicates,
and column lineage between raw -> cleaned -> engineered datasets.

Output: writes a plain-text report to outputs/data_audit_report.txt
"""
import re
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd


OUTPUT_REPORT = Path("outputs/data_audit_report.txt")


def snake_case(name: str) -> str:
    """Convert a column name to a lowercase snake_case-like form.

    This is a lightweight normalizer for comparing column names.
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name)
    return name


def parquet_metadata(path: Path):
    """Return (num_rows, list_of_columns) using pyarrow metadata only."""
    pf = pq.ParquetFile(str(path))
    num_rows = pf.metadata.num_rows
    # schema_arrow is safe and won't load the dataset
    try:
        cols = list(pf.schema_arrow.names)
    except Exception:
        # fallback to parquet metadata schema
        cols = [c.name for c in pf.schema]
    return int(num_rows), cols


def read_single_column(path: Path, column: str):
    """Read only a single column with pandas (pyarrow engine)."""
    # pandas will use engine available in environment; specify columns to avoid full read
    ser = pd.read_parquet(path, columns=[column])[column]
    return ser


def parse_exact_duplicates_from_eda(eda_path: Path):
    """Parse the exact duplicate rows count from step1_eda_summary.txt.

    Matches the specific text:
    "Exact duplicate rows (every column identical): <number>"
    """
    txt = eda_path.read_text(encoding="utf-8")
    # Accept either "Exact duplicate rows: 70017" or
    # "Exact duplicate rows (every column identical): 70017"
    m = re.search(r"Exact duplicate rows(?: \(every column identical\))?:\s*([\d,]+)", txt)
    if not m:
        return None
    # remove commas if present and return int
    return int(m.group(1).replace(",", ""))


def main():
    # Files we will inspect
    raw_fp = Path("outputs/clinical_data_raw.parquet")
    cleaned_fp = Path("outputs/data_cleaned.parquet")
    eng_fp = Path("outputs/data_engineered.parquet")
    eda_summary = Path("outputs/step1_eda_summary.txt")

    # Begin progress prints
    print("Progress: Gathering parquet metadata (no full loads)...")

    report_lines = []
    report_lines.append("DATA AUDIT REPORT")
    report_lines.append("=================")

    # 1. Use pyarrow metadata to obtain counts and columns
    meta_results = {}
    for label, fp in ("Raw", raw_fp), ("Cleaned", cleaned_fp), ("Engineered", eng_fp):
        if not fp.exists():
            report_lines.append(f"ERROR: {fp} not found")
            meta_results[label] = (None, [])
            continue
        rows, cols = parquet_metadata(fp)
        meta_results[label] = (rows, cols)
        report_lines.append(f"{label} metadata: rows={rows:,}, columns={len(cols)}")

    # 2. Validate expected shapes
    expected = {
        "Raw": (1155000, 41),
        "Cleaned": (1050000, 31),
        "Engineered": (1050000, 38),
    }
    report_lines.append("")
    report_lines.append("Expected shapes validation:")
    shape_pass = True
    for label in ("Raw", "Cleaned", "Engineered"):
        got = meta_results[label][0]
        got_cols = len(meta_results[label][1])
        exp_rows, exp_cols = expected[label]
        if got is None:
            report_lines.append(f"- {label}: MISSING file")
            shape_pass = False
            continue
        ok_rows = got == exp_rows
        ok_cols = got_cols == exp_cols
        report_lines.append(f"- {label}: rows -> expected {exp_rows:,}, got {got:,} -> {'OK' if ok_rows else 'MISMATCH'}; columns -> expected {exp_cols}, got {got_cols} -> {'OK' if ok_cols else 'MISMATCH'}")
        if not (ok_rows and ok_cols):
            shape_pass = False

    # 3. Read only raw Patient_ID column
    print("Progress: Reading Patient_ID from raw (pandas, single column)...")
    try:
        raw_pid = read_single_column(raw_fp, "Patient_ID")
        raw_total = int(len(raw_pid))
        raw_unique = int(raw_pid.nunique(dropna=False))
        raw_repeated = int(raw_pid.duplicated().sum())
        report_lines.append("")
        report_lines.append("Raw Patient_ID summary:")
        report_lines.append(f"- total rows: {raw_total:,}")
        report_lines.append(f"- unique Patient_IDs: {raw_unique:,}")
        report_lines.append(f"- repeated Patient_ID rows (duplicated().sum()): {raw_repeated:,}")
    except Exception as e:
        report_lines.append(f"ERROR reading raw Patient_ID: {e}")
        raw_total = raw_unique = raw_repeated = None

    # 4. Read only cleaned patient_id
    print("Progress: Reading patient_id from cleaned (pandas, single column)...")
    try:
        cleaned_pid = read_single_column(cleaned_fp, "patient_id")
        cleaned_total = int(len(cleaned_pid))
        cleaned_unique = int(cleaned_pid.nunique(dropna=False))
        cleaned_is_unique = cleaned_pid.is_unique
        cleaned_repeated = int(cleaned_pid.duplicated().sum())
        report_lines.append("")
        report_lines.append("Cleaned patient_id summary:")
        report_lines.append(f"- total rows: {cleaned_total:,}")
        report_lines.append(f"- unique patient_id: {cleaned_unique:,}")
        report_lines.append(f"- is_unique: {cleaned_is_unique}")
        report_lines.append(f"- repeated patient_id rows: {cleaned_repeated:,}")
    except Exception as e:
        report_lines.append(f"ERROR reading cleaned patient_id: {e}")
        cleaned_total = cleaned_unique = None
        cleaned_is_unique = False
        cleaned_repeated = None

    # 5. Verify reduction of 105,000 rows matches additional repeated Patient_ID rows
    report_lines.append("")
    report_lines.append("Reduction check (raw -> cleaned):")
    reduction_ok = False
    try:
        if raw_total is not None and cleaned_total is not None:
            reduction = raw_total - cleaned_total
            report_lines.append(f"Observed row reduction raw -> cleaned: {reduction:,}")
            # If cleaned patient_id is unique then duplicates removed should equal raw_repeated
            if cleaned_is_unique:
                report_lines.append(f"Raw repeated Patient_ID rows: {raw_repeated:,}")
                reduction_ok = (reduction == raw_repeated)
                report_lines.append(f"Reduction equals raw repeated rows? {'YES' if reduction_ok else 'NO'}")
            else:
                report_lines.append("Cannot assert duplicate-driven reduction because cleaned patient_id is not unique.")
    except Exception as e:
        report_lines.append(f"ERROR computing reduction check: {e}")

    # 6. Read treatment_outcome from cleaned and engineered and count missing values
    print("Progress: Reading treatment_outcome from cleaned and engineered to count missing values...")
    try:
        to_cleaned = read_single_column(cleaned_fp, "treatment_outcome")
        missing_cleaned = int(to_cleaned.isna().sum())
    except Exception as e:
        missing_cleaned = None
        report_lines.append(f"ERROR reading cleaned treatment_outcome: {e}")

    try:
        to_eng = read_single_column(eng_fp, "treatment_outcome")
        missing_eng = int(to_eng.isna().sum())
    except Exception as e:
        missing_eng = None
        report_lines.append(f"ERROR reading engineered treatment_outcome: {e}")

    report_lines.append("")
    report_lines.append("Missing values in treatment_outcome:")
    report_lines.append(f"- cleaned: {missing_cleaned if missing_cleaned is not None else 'ERROR'}")
    report_lines.append(f"- engineered: {missing_eng if missing_eng is not None else 'ERROR'}")

    # 7. Column lineage narrative
    report_lines.append("")
    report_lines.append("Column lineage and transformations:")
    report_lines.append("- Merged duplicate columns:\n  patient_id_1 -> Patient_ID -> patient_id; age_1 -> Age -> age; gender_1 -> Gender -> gender; drug_name_1 -> Drug_Name -> drug_name")
    report_lines.append("- Helper columns used then removed: Weight_lbs (used to complete weight_kg), blood_pressure (used to derive systolic_bp and diastolic_bp)")
    report_lines.append("- Irrelevant columns removed: Notes, Extra_Col_1, Extra_Col_2, unnamed_0")
    report_lines.append("- All remaining clinical columns were retained and standardized to lowercase snake_case names.")

    # 8. Verify important columns present in cleaned and engineered
    important = ["patient_id", "treatment_outcome", "admission_date", "adverse_event", "readmission_30d"]
    report_lines.append("")
    report_lines.append("Important columns presence check in cleaned and engineered:")
    presence_ok = True
    for label, fp in (("Cleaned", cleaned_fp), ("Engineered", eng_fp)):
        cols = [snake_case(c) for c in meta_results[label][1]]
        missing = [c for c in important if c not in cols]
        if missing:
            presence_ok = False
            report_lines.append(f"- {label}: MISSING {missing}")
        else:
            report_lines.append(f"- {label}: all important columns present")

    # Exact expected cleaned columns (31) — validate no unexpected/missing
    expected_cleaned_cols = [
        "patient_id", "age", "gender", "ethnicity", "weight_kg", "height_cm", "bmi",
        "systolic_bp", "diastolic_bp", "heart_rate", "temperature_f", "hemoglobin",
        "wbc_count", "alt_enzyme", "ast_enzyme", "creatinine", "egfr", "hba1c",
        "total_cholesterol", "drug_name", "dosage_mg", "duration_days", "route",
        "concurrent_drugs", "diagnosis", "smoking_status", "alcohol_use",
        "admission_date", "treatment_outcome", "adverse_event", "readmission_30d",
    ]
    report_lines.append("")
    report_lines.append("Cleaned dataset exact-column validation (expecting 31 columns):")
    cleaned_cols_norm = [snake_case(c) for c in meta_results["Cleaned"][1]]
    cleaned_cols_set = set(cleaned_cols_norm)
    expected_set = set(expected_cleaned_cols)
    missing_in_cleaned = [c for c in expected_cleaned_cols if c not in cleaned_cols_set]
    unexpected_in_cleaned = [c for c in cleaned_cols_norm if c not in expected_set]
    if missing_in_cleaned:
        report_lines.append(f"- Missing expected cleaned columns ({len(missing_in_cleaned)}): {missing_in_cleaned}")
    else:
        report_lines.append("- No expected cleaned columns missing")
    if unexpected_in_cleaned:
        report_lines.append(f"- Unexpected cleaned columns ({len(unexpected_in_cleaned)}): {unexpected_in_cleaned}")
    else:
        report_lines.append("- No unexpected cleaned columns found")
    cleaned_columns_ok = (len(missing_in_cleaned) == 0 and len(unexpected_in_cleaned) == 0)

    # 9. Verify engineered features
    eng_features = [
        "kidney_stage",
        "bmi_category",
        "age_group",
        "liver_risk",
        "polypharmacy",
        "elderly_high_dose",
        "de_ritis_ratio",
    ]
    eng_cols = [snake_case(c) for c in meta_results["Engineered"][1]]
    missing_eng_features = [f for f in eng_features if f not in eng_cols]
    report_lines.append("")
    if missing_eng_features:
        report_lines.append(f"Engineered data missing expected features: {missing_eng_features}")
        features_ok = False
    else:
        report_lines.append("Engineered data contains all listed features.")
        features_ok = True

    # 10. Read exact raw duplicate count from EDA summary
    report_lines.append("")
    if eda_summary.exists():
        raw_exact_dup = parse_exact_duplicates_from_eda(eda_summary)
        if raw_exact_dup is None:
            report_lines.append("Could not parse exact duplicate count from step1_eda_summary.txt")
        else:
            report_lines.append(f"Exact duplicate rows reported in EDA summary: {raw_exact_dup:,}")
    else:
        raw_exact_dup = None
        report_lines.append(f"EDA summary file {eda_summary} not found")

    # 11. Explain cleaned exact duplicate rows must be zero when patient_id is unique
    report_lines.append("")
    report_lines.append("Note: If `patient_id` is unique in cleaned data, there cannot be any rows that are exact duplicates across every column because duplicate rows would necessarily share the same patient_id.")

    # 12. PASS/FAIL and conclusion
    report_lines.append("")
    report_lines.append("PASS/FAIL Summary:")
    checks = {
        "shapes_match_expected": shape_pass,
        "reduction_matches_removed_duplicates": reduction_ok,
        "important_columns_present": presence_ok,
        "engineered_features_present": features_ok,
        "cleaned_columns_exact_match": cleaned_columns_ok,
    }
    for k, v in checks.items():
        report_lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")

    overall = all(checks.values())
    report_lines.append("")
    if overall:
        report_lines.append("CONCLUSION: PASS — No important columns appear missing and removed rows are explained by duplicate Patient_ID rows.")
    else:
        report_lines.append("CONCLUSION: FAIL — One or more checks failed. See details above for missing columns or unexplained rows.")

    # 13. Save report
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Progress: Report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    # Run the audit when invoked as a script
    main()
