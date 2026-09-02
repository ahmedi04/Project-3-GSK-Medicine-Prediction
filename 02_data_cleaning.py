#!/usr/bin/env python3
"""
Full-table data cleaning (no chunking) for the clinical dataset.

Reads `outputs/clinical_data_raw.parquet`, applies structural and value
cleaning rules described in Step 2 of the project, and writes a cleaned
Parquet file plus a detailed report. The script is written with
beginner-friendly comments and prints progress to the terminal.
"""
from pathlib import Path
import re
from collections import Counter
import numpy as np
import pandas as pd


def find_col(df, names):
    """Find the first column in df whose lower-stripped name matches one of the names list."""
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        key = n.lower()
        if key in cols:
            return cols[key]
    return None


def to_na_like(x):
    """Return True if x is a textual missing placeholder."""
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    if s == "":
        return True
    if s in {"-", "--", "na", "n/a", "unknown", "<missing>"}:
        return True
    return False


def extract_dosage_mg(val):
    """Extract a numeric mg dosage from strings like '100 mg', '2x50mg', '50mg daily'."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    # 1) Prefer an explicit '<number>mg' pattern (allow surrounding text)
    m = re.search(r"(?:^|\D)(\d+\.?\d*)\s*mg(?:\D|$)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    # 2) Only accept the whole stripped value if it's purely numeric
    stripped = s.strip()
    if re.fullmatch(r"\d+\.?\d*", stripped):
        try:
            return float(stripped)
        except Exception:
            return np.nan
    # Otherwise do not guess from arbitrary text
    return np.nan


def main():
    src = Path("outputs/clinical_data_raw.parquet")
    out = Path("outputs/data_cleaned.parquet")
    report = Path("outputs/step2_cleaning_report.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading full dataset (this will use memory)...")
    df = pd.read_parquet(src)

    report_lines = []

    # Record original shape
    orig_rows, orig_cols = df.shape
    report_lines.append(f"Original shape: rows={orig_rows}, columns={orig_cols}")

    # Exact full-row duplicates
    full_dup_count = int(df.duplicated().sum())
    report_lines.append(f"Exact full-row duplicates: {full_dup_count}")

    # Unique Patient_ID count and repeated patient rows
    # Try common variants for patient id column
    pid_col = find_col(df, ["Patient_ID", "patient_id", "patientid"])
    if pid_col is None:
        raise SystemExit("No Patient_ID column found in input file.")

    unique_pids = df[pid_col].nunique(dropna=True)
    total_pid_rows = df[pid_col].notna().sum()
    # Count repeated patient_id rows (rows where Patient_ID appears more than once)
    pid_counts = df[pid_col].astype(str).value_counts()
    # number of additional rows beyond first occurrences
    repeated_pid_rows = int((pid_counts - 1)[(pid_counts - 1) > 0].sum())
    report_lines.append(f"Unique Patient_IDs: {unique_pids}")
    report_lines.append(f"Repeated Patient_ID rows (additional copies beyond first): {repeated_pid_rows}")
    # Estimate non-identical repeated rows (rows beyond exact full-row duplicates)
    non_identical_repeated_rows = max(repeated_pid_rows - full_dup_count, 0)
    report_lines.append(f"Non-identical repeated rows (estimated): {non_identical_repeated_rows}")

    # For patient duplicates: keep first occurrence of each Patient_ID
    # Combine duplicate/redundant columns BEFORE removing repeated Patient_ID rows
    # This ensures we preserve useful values from secondary columns when we keep the first row
    print("Combining duplicate columns (age_1, gender_1, drug_name_1, Weight_lbs -> weight_kg) before deduplication...")
    def col_or_none(df, candidates):
        c = find_col(df, candidates)
        return c

    age_col = col_or_none(df, ["age"]) or "age"
    age1_col = col_or_none(df, ["age_1"]) or None
    gender_col = col_or_none(df, ["gender"]) or "gender"
    gender1_col = col_or_none(df, ["gender_1"]) or None
    drug_col = col_or_none(df, ["drug_name"]) or "drug_name"
    drug1_col = col_or_none(df, ["drug_name_1"]) or None
    weight_kg_col = col_or_none(df, ["weight_kg"]) or "weight_kg"
    weight_lbs_col = col_or_none(df, ["weight_lbs"]) or None
    bp_col = col_or_none(df, ["blood_pressure"]) or None
    systolic_col = col_or_none(df, ["systolic_bp"]) or "systolic_bp"
    diastolic_col = col_or_none(df, ["diastolic_bp"]) or "diastolic_bp"

    # Fill Age from age_1 where available
    if age1_col and age_col in df.columns and age1_col in df.columns:
        missing_age_before = int(df[age_col].isna().sum())
        df[age_col] = df[age_col].fillna(df[age1_col])
        filled_from_age1 = missing_age_before - int(df[age_col].isna().sum())
    else:
        filled_from_age1 = 0
    report_lines.append(f"Filled Age from age_1: {filled_from_age1}")

    # Fill Gender from gender_1 where available
    if gender1_col and gender_col in df.columns and gender1_col in df.columns:
        missing_gender_before = int(df[gender_col].isna().sum())
        df[gender_col] = df[gender_col].fillna(df[gender1_col])
        filled_from_gender1 = missing_gender_before - int(df[gender_col].isna().sum())
    else:
        filled_from_gender1 = 0
    report_lines.append(f"Filled Gender from gender_1: {filled_from_gender1}")

    # Fill Drug_Name from drug_name_1 where available
    if drug1_col and drug_col in df.columns and drug1_col in df.columns:
        missing_drug_before = int(df[drug_col].isna().sum())
        df[drug_col] = df[drug_col].fillna(df[drug1_col])
        filled_from_drug1 = missing_drug_before - int(df[drug_col].isna().sum())
    else:
        filled_from_drug1 = 0
    report_lines.append(f"Filled Drug_Name from drug_name_1: {filled_from_drug1}")

    # Fill weight_kg from Weight_lbs if missing
    if weight_lbs_col and weight_kg_col in df.columns and weight_lbs_col in df.columns:
        missing_weight_before = int(df[weight_kg_col].isna().sum())
        df[weight_kg_col] = df[weight_kg_col].fillna(df[weight_lbs_col] / 2.20462)
        filled_from_weightlbs = missing_weight_before - int(df[weight_kg_col].isna().sum())
    else:
        filled_from_weightlbs = 0
    report_lines.append(f"Filled weight_kg from Weight_lbs: {filled_from_weightlbs}")

    # Parse blood_pressure into systolic/diastolic and use to fill missing
    filled_systolic = 0
    filled_diastolic = 0
    if bp_col and bp_col in df.columns:
        bp_series = df[bp_col].astype(str).str.strip()
        m = bp_series.str.extract(r"(?P<s>\d{2,3})\s*/\s*(?P<d>\d{2,3})")
        if systolic_col in df.columns:
            missing_s_before = int(df[systolic_col].isna().sum())
            df.loc[df[systolic_col].isna() & m["s"].notna(), systolic_col] = pd.to_numeric(m["s"], errors="coerce")
            filled_systolic = missing_s_before - int(df[systolic_col].isna().sum())
        if diastolic_col in df.columns:
            missing_d_before = int(df[diastolic_col].isna().sum())
            df.loc[df[diastolic_col].isna() & m["d"].notna(), diastolic_col] = pd.to_numeric(m["d"], errors="coerce")
            filled_diastolic = missing_d_before - int(df[diastolic_col].isna().sum())
    report_lines.append(f"Filled systolic_bp from blood_pressure: {filled_systolic}")
    report_lines.append(f"Filled diastolic_bp from blood_pressure: {filled_diastolic}")

    # Now remove repeated patients (keep first occurrence). Use original Patient_ID column name if present.
    print("Removing repeated Patient_ID rows (keep first occurrence)...")
    if "Patient_ID" in df.columns:
        df = df.drop_duplicates(subset=["Patient_ID"], keep="first").copy()
    else:
        # fallback to previously detected pid_col
        df = df.drop_duplicates(subset=[pid_col], keep="first").copy()
    after_patient_drop_rows = len(df)
    report_lines.append(f"Rows after keeping first Patient_ID occurrences: {after_patient_drop_rows}")

    # (previously duplicated combine block removed)

    # Drop redundant/junk columns
    drop_cols = [
        "patient_id_1", "age_1", "gender_1", "drug_name_1", "Weight_lbs",
        "blood_pressure", "Notes", "Extra_Col_1", "Extra_Col_2", "unnamed_0"
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")
    report_lines.append(f"Dropped columns: {existing_drop}")

    # Rename remaining columns to exact snake_case list provided
    final_cols = [
        "patient_id", "age", "gender", "ethnicity", "weight_kg", "height_cm", "bmi",
        "systolic_bp", "diastolic_bp", "heart_rate", "temperature_f", "hemoglobin",
        "wbc_count", "alt_enzyme", "ast_enzyme", "creatinine", "egfr", "hba1c",
        "total_cholesterol", "drug_name", "dosage_mg", "duration_days", "route",
        "concurrent_drugs", "diagnosis", "smoking_status", "alcohol_use", "admission_date",
        "treatment_outcome", "adverse_event", "readmission_30d"
    ]

    # Create a simple lowercase key mapping for current columns to target names when possible
    col_map = {}
    current_cols = {c.lower(): c for c in df.columns}
    for target in final_cols:
        if target in df.columns:
            col_map[target] = target
            continue
        t = target.lower()
        aliases = {
            "patient_id": ["patient_id", "patient id", "patientid"],
            "age": ["age"],
            "gender": ["gender"],
            "ethnicity": ["ethnicity"],
            "weight_kg": ["weight_kg", "weightkg", "weight"],
            "height_cm": ["height_cm", "heightcm", "height"],
            "bmi": ["bmi"],
            "systolic_bp": ["systolic_bp", "systolicbp", "systolic"],
            "diastolic_bp": ["diastolic_bp", "diastolicbp", "diastolic"],
            "heart_rate": ["heart_rate", "heartrate", "pulse"],
            "temperature_f": ["temperature_f", "temperaturef", "temp"],
            "hemoglobin": ["hemoglobin", "hb"],
            "wbc_count": ["wbc_count", "wbc"],
            "alt_enzyme": ["alt_enzyme", "alt"],
            "ast_enzyme": ["ast_enzyme", "ast"],
            "creatinine": ["creatinine"],
            "egfr": ["egfr"],
            "hba1c": ["hba1c"],
            "total_cholesterol": ["total_cholesterol", "cholesterol"],
            "drug_name": ["drug_name", "drugname", "drug"],
            "dosage_mg": ["dosage_mg", "dosage"],
            "duration_days": ["duration_days", "duration"],
            "route": ["route"],
            "concurrent_drugs": ["concurrent_drugs", "concurrentdrugs"],
            "diagnosis": ["diagnosis"],
            "smoking_status": ["smoking_status", "smoking"],
            "alcohol_use": ["alcohol_use", "alcohol"],
            "admission_date": ["admission_date", "admissiondate", "admit_date"],
            "treatment_outcome": ["treatment_outcome", "treatmentoutcome"],
            "adverse_event": ["adverse_event", "adverse"],
            "readmission_30d": ["readmission_30d", "readmission30d"]
        }
        found = None
        for alias in aliases.get(t, [t]):
            if alias in current_cols:
                found = current_cols[alias]
                break
        if found:
            col_map[found] = target

    # Apply renaming
    df = df.rename(columns=col_map)

    # Verify all required final columns are present; raise error if any are missing
    missing_required = [c for c in final_cols if c not in df.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns after renaming: {missing_required}")
    # Reorder to the exact required final column order
    df = df[final_cols].copy()

    # Standardize missing text placeholders across object columns
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].apply(lambda x: pd.NA if to_na_like(x) else x)

    # Standardize categorical values: lowercase, strip spaces, map aliases
    print("Standardizing categorical values (gender, ethnicity, drug_name, route, diagnosis, smoking, alcohol)...")
    def norm_text(s):
        if pd.isna(s):
            return s
        return str(s).strip().lower()

    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(norm_text)
        gender_map = {"m": "male", "male": "male", "mal": "male", "f": "female", "female": "female", "femal": "female", "o": "other", "other": "other", "1": "male", "0": "female"}
        df["gender"] = df["gender"].map(lambda x: gender_map.get(x, x) if pd.notna(x) else x)

    if "ethnicity" in df.columns:
        df["ethnicity"] = df["ethnicity"].apply(norm_text)
        eth_map = {
            "white": "white", "caucasian": "white",
            "black": "black", "african american": "black", "african-american": "black", "aa": "black",
            "hispanic": "hispanic_or_latino", "latino": "hispanic_or_latino", "latina": "hispanic_or_latino", "hispanic/latino": "hispanic_or_latino"
        }
        df["ethnicity"] = df["ethnicity"].map(lambda x: eth_map.get(x, x) if pd.notna(x) else x)

    if "drug_name" in df.columns:
        df["drug_name"] = df["drug_name"].apply(norm_text)
        # normalize repeated internal spaces before mapping
        df["drug_name"] = df["drug_name"].str.replace(r"\s+", " ", regex=True)
        drug_map = {
            "warfarine": "warfarin",
            "metformine": "metformin", "met formin": "metformin",
            "omeprazol": "omeprazole",
            "gabapentine": "gabapentin",
            "amoxycillin": "amoxicillin", "amoxicilin": "amoxicillin",
            "lisinipril": "lisinopril", "lisino pril": "lisinopril",
            "ibuprofin": "ibuprofen", "ibuprophen": "ibuprofen",
            "atorvastatine": "atorvastatin", "atorvastain": "atorvastatin",
            "tramadole": "tramadol",
            "insulin glargine": "insulin glargine"
        }
        # Map common misspellings and treat literal 'none' as a sentinel meaning no drug
        drug_map["none"] = "no_drug"
        df["drug_name"] = df["drug_name"].map(lambda x: drug_map.get(x, x) if pd.notna(x) else x)

    if "route" in df.columns:
        df["route"] = df["route"].apply(norm_text)
        route_map = {"iv": "intravenous", "im": "intramuscular", "sc": "subcutaneous"}
        df["route"] = df["route"].map(lambda x: route_map.get(x, x) if pd.notna(x) else x)

    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].apply(norm_text)
        diag_map = {"htn": "hypertension", "dm2": "type 2 diabetes", "t2dm": "type 2 diabetes",
                    "afib": "atrial fibrillation", "ckd": "chronic kidney disease", "oa": "osteoarthritis",
                    "hld": "hyperlipidemia", "chf": "heart failure"}
        df["diagnosis"] = df["diagnosis"].map(lambda x: diag_map.get(x, x) if pd.notna(x) else x)

    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].apply(norm_text)
        smoke_map = {"yes": "current", "y": "current", "true": "current", "1": "current", "current": "current",
                     "no": "never", "n": "never", "false": "never", "0": "never", "never": "never",
                     "former": "former", "ex smoker": "former", "ex-smoker": "former", "ex": "former"}
        df["smoking_status"] = df["smoking_status"].map(lambda x: smoke_map.get(x, x) if pd.notna(x) else x)

    if "alcohol_use" in df.columns:
        df["alcohol_use"] = df["alcohol_use"].apply(norm_text)

    # Parse numeric fields and coerce to numeric types
    numeric_cols = [
        "age", "weight_kg", "height_cm", "bmi", "systolic_bp", "diastolic_bp",
        "heart_rate", "temperature_f", "hemoglobin", "wbc_count", "alt_enzyme",
        "ast_enzyme", "creatinine", "egfr", "hba1c", "total_cholesterol",
        "dosage_mg", "duration_days", "concurrent_drugs"
    ]
    for col in numeric_cols:
        if col in df.columns:
            if col == "dosage_mg":
                # extract numeric mg values from free text dosage strings
                df[col] = df[col].apply(extract_dosage_mg)
            elif col == "duration_days":
                # process duration textual labels before coercion
                df[col] = df[col].replace({"ongoing": 365, "chronic": 365, "lifetime": 365})
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # duration_days already processed above as part of numeric parsing

    # Convert heights between 4 and 8 to centimetres (likely feet values)
    if "height_cm" in df.columns:
        mask_feet = df["height_cm"].between(4, 8)
        df.loc[mask_feet, "height_cm"] = df.loc[mask_feet, "height_cm"] * 30.48

    # Replace infinities with NaN across numeric columns
    # Replace infinities only in numeric columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Replace physiologically impossible values with NaN using defined ranges
    ranges = {
        "age": (0, 120),
        "weight_kg": (2, 300),
        "height_cm": (50, 250),
        "bmi": (5, 100),
        "systolic_bp": (50, 250),
        "diastolic_bp": (30, 150),
        "heart_rate": (20, 250),
        "temperature_f": (90, 110),
        "hemoglobin": (3, 25),
        "wbc_count": (500, 100000),
        "alt_enzyme": (0, 1000),
        "ast_enzyme": (0, 1000),
        "creatinine": (0.1, 20),
        "egfr": (0, 200),
        "hba1c": (2, 25),
        "total_cholesterol": (50, 1000),
        "dosage_mg": (0, 5000),
        "duration_days": (0, 3650),
        "concurrent_drugs": (0, 50)
    }
    for col, (low, high) in ranges.items():
        if col in df.columns:
            df.loc[(df[col] < low) | (df[col] > high), col] = pd.NA

    # Recalculate BMI when missing or outside plausible range (5-100)
    if "bmi" in df.columns and "weight_kg" in df.columns and "height_cm" in df.columns:
        mask_bmi_bad = (df["bmi"].isna()) | (df["bmi"] < 5) | (df["bmi"] > 100)
        valid_height = df["height_cm"] > 0
        h_m = df.loc[valid_height, "height_cm"] / 100.0
        recomputed = pd.Series(pd.NA, index=df.index)
        recomputed.loc[valid_height] = df.loc[valid_height, "weight_kg"] / (h_m ** 2)
        df.loc[mask_bmi_bad & valid_height, "bmi"] = recomputed.loc[mask_bmi_bad & valid_height]

    # Parse admission_date
    if "admission_date" in df.columns:
        try:
            df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce", format="mixed")
        except Exception:
            df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce", infer_datetime_format=True)

    # Map treatment_outcome and other binary columns to 0/1
    def map_binary(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        if s in {"0", "no", "false", "ineffective"}:
            return 0
        if s in {"1", "yes", "true", "effective"}:
            return 1
        return pd.NA

    for col in ["treatment_outcome", "adverse_event", "readmission_30d"]:
        if col in df.columns:
            df[col] = df[col].apply(map_binary)

    # Report how many treatment_outcome values remain missing (they must be handled before supervised training)
    if "treatment_outcome" in df.columns:
        missing_outcomes = int(df["treatment_outcome"].isna().sum())
        report_lines.append(f"treatment_outcome missing after parsing: {missing_outcomes} (keep rows; remove before supervised training)")

    # Handle missing predictors: fill numeric with median and categorical with mode
    skip_impute = {"patient_id", "admission_date", "treatment_outcome", "adverse_event", "readmission_30d"}
    numeric_predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_impute]
    categorical_predictors = [c for c in df.select_dtypes(include=[object, "string"]).columns if c not in skip_impute]

    medians = {}
    for col in numeric_predictors:
        med = df[col].median(skipna=True)
        medians[col] = med
        df[col] = df[col].fillna(med)

    modes = {}
    for col in categorical_predictors:
        try:
            mode_val = df[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if not mode_val.empty else pd.NA
        except Exception:
            mode_val = pd.NA
        modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)

    report_lines.append(f"Numeric medians used for imputation: {medians}")
    report_lines.append(f"Categorical modes used: {modes}")

    # Cap outliers using IQR for selected clinical predictors
    iqr_cols = ["age", "bmi", "dosage_mg", "hemoglobin", "creatinine", "alt_enzyme", "ast_enzyme"]
    iqr_limits = {}
    capped_counts = {}
    for col in iqr_cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_low = int((df[col] < lower).sum())
        before_high = int((df[col] > upper).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        after_min = float(df[col].min(skipna=True)) if df[col].notna().any() else float("nan")
        after_max = float(df[col].max(skipna=True)) if df[col].notna().any() else float("nan")
        iqr_limits[col] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "lower": float(lower), "upper": float(upper),
                           "below": before_low, "above": before_high, "min_after": after_min, "max_after": after_max}
        capped_counts[col] = before_low + before_high

    report_lines.append(f"IQR limits and capped counts: {iqr_limits}")

    # Final checks and strict validations
    final_rows, final_cols_count = df.shape
    report_lines.append(f"Final shape: rows={final_rows}, columns={final_cols_count}")

    # Final verification uses the standardized column name `patient_id`
    if "patient_id" not in df.columns:
        raise SystemExit("Required column 'patient_id' missing after cleaning")

    patient_id_unique = df["patient_id"].is_unique
    report_lines.append(f"patient_id unique after cleaning: {patient_id_unique}")

    final_column_list = df.columns.tolist()
    report_lines.append(f"Final columns ({len(final_column_list)}): {final_column_list}")

    missing_after = df.isna().sum().to_dict()
    report_lines.append(f"Missing counts after cleaning: {missing_after}")

    # Enforce final requirements
    if len(df.columns) != 31:
        raise SystemExit(f"Final column count is {len(df.columns)} but expected exactly 31")
    if not patient_id_unique:
        raise SystemExit("patient_id values are not unique after cleaning")

    # Save the cleaned dataset
    print("Writing cleaned dataset to Parquet...")
    df.to_parquet(out, index=False)
    report_lines.append(f"Wrote cleaned Parquet to {out}")

    # Save report
    with open(report, "w", encoding="utf-8") as f:
        f.write("Step 2 - Data cleaning report\n")
        f.write("=============================\n\n")
        for line in report_lines:
            f.write(line + "\n")

    print("Done. Report written to", report)


if __name__ == "__main__":
    main()
