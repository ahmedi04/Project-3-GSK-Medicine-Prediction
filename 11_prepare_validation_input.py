#!/usr/bin/env python3
"""Prepare a corrected validation input dataset without imputation or IQR capping."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_FILE = Path("outputs/clinical_data_raw.parquet")
EXISTING_CLEANED_FILE = Path("outputs/data_cleaned.parquet")
EXISTING_ENGINEERED_FILE = Path("outputs/data_engineered.parquet")
OUTPUT_BASE_FILE = Path("outputs/scenario3_validation_input.parquet")
OUTPUT_SPLIT_FILE = Path("outputs/scenario3_split_assignments.parquet")
REPORT_FILE = Path("outputs/scenario3_preparation_report.txt")
TARGET = "treatment_outcome"


def find_col(df, names):
    """Find the first matching column name using lowercase comparison."""
    cols = {c.lower(): c for c in df.columns}
    for name in names:
        key = name.lower()
        if key in cols:
            return cols[key]
    return None


def to_na_like(x):
    """Return True if a value is a text placeholder for missingness."""
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s == "" or s in {"-", "--", "na", "n/a", "unknown", "<missing>"}


def extract_dosage_mg(val):
    """Extract dosage in mg from text while avoiding over-guessing."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    match = re.search(r"(?:^|\D)(\d+\.?\d*)\s*mg(?:\D|$)", s)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return np.nan
    stripped = s.strip()
    if re.fullmatch(r"\d+\.?\d*", stripped):
        try:
            return float(stripped)
        except Exception:
            return np.nan
    return np.nan


def map_binary(x):
    """Map known binary text labels to 0/1 and leave unknowns missing."""
    if pd.isna(x):
        return pd.NA

    # First: accept numeric 0/1 (including 0.0/1.0 and numeric strings).
    s_raw = str(x).strip()
    num = pd.to_numeric(s_raw, errors="coerce")
    if pd.notna(num):
        if np.isfinite(num) and num in (0, 1):
            return int(num)
        # Reject any other numeric values (do not round/truncate).
        return pd.NA

    # Second: accept known textual labels.
    s = s_raw.lower()
    if s in {"0", "no", "false", "ineffective"}:
        return 0
    if s in {"1", "yes", "true", "effective"}:
        return 1
    return pd.NA


def split_summary(name, series):
    """Summarize row count and binary class distribution."""
    total = len(series)
    positive = int((series == 1).sum())
    negative = int((series == 0).sum())
    pos_pct = (positive / total) * 100 if total else 0.0
    neg_pct = (negative / total) * 100 if total else 0.0
    return (
        f"{name}: rows={total:,}, negative(0)={negative:,} ({neg_pct:.3f}%), "
        f"positive(1)={positive:,} ({pos_pct:.3f}%)"
    )


def validate_patient_ids(df, dataset_name):
    """Require patient_id to be present, non-missing, and unique."""
    if "patient_id" not in df.columns:
        raise SystemExit(f"{dataset_name}: missing required column 'patient_id'.")
    if df["patient_id"].isna().any():
        raise SystemExit(f"{dataset_name}: patient_id contains missing values.")
    if not df["patient_id"].is_unique:
        raise SystemExit(f"{dataset_name}: patient_id values are not unique.")


def prepare_base_cleaned_from_raw(df_raw, report_lines):
    """Reuse deterministic cleaning logic while stopping before imputation/IQR clipping."""
    df = df_raw.copy()

    pid_col = find_col(df, ["Patient_ID", "patient_id", "patientid"])
    if pid_col is None:
        raise SystemExit("No Patient_ID column found in raw dataset.")

    print("Progress: Consolidating duplicate/helper columns before deduplication...")
    age_col = find_col(df, ["age"]) or "age"
    age1_col = find_col(df, ["age_1"])
    gender_col = find_col(df, ["gender"]) or "gender"
    gender1_col = find_col(df, ["gender_1"])
    drug_col = find_col(df, ["drug_name"]) or "drug_name"
    drug1_col = find_col(df, ["drug_name_1"])
    weight_kg_col = find_col(df, ["weight_kg"]) or "weight_kg"
    weight_lbs_col = find_col(df, ["weight_lbs"])
    bp_col = find_col(df, ["blood_pressure"])
    systolic_col = find_col(df, ["systolic_bp"]) or "systolic_bp"
    diastolic_col = find_col(df, ["diastolic_bp"]) or "diastolic_bp"

    if age1_col and age_col in df.columns and age1_col in df.columns:
        df[age_col] = df[age_col].fillna(df[age1_col])
    if gender1_col and gender_col in df.columns and gender1_col in df.columns:
        df[gender_col] = df[gender_col].fillna(df[gender1_col])
    if drug1_col and drug_col in df.columns and drug1_col in df.columns:
        df[drug_col] = df[drug_col].fillna(df[drug1_col])
    if weight_lbs_col and weight_kg_col in df.columns and weight_lbs_col in df.columns:
        df[weight_kg_col] = df[weight_kg_col].fillna(df[weight_lbs_col] / 2.20462)

    if bp_col and bp_col in df.columns:
        bp_series = df[bp_col].astype(str).str.strip()
        bp_parts = bp_series.str.extract(r"(?P<s>\d{2,3})\s*/\s*(?P<d>\d{2,3})")
        if systolic_col in df.columns:
            df.loc[df[systolic_col].isna() & bp_parts["s"].notna(), systolic_col] = pd.to_numeric(
                bp_parts["s"], errors="coerce"
            )
        if diastolic_col in df.columns:
            df.loc[df[diastolic_col].isna() & bp_parts["d"].notna(), diastolic_col] = pd.to_numeric(
                bp_parts["d"], errors="coerce"
            )

    print("Progress: Applying the existing Patient_ID deduplication policy...")
    if "Patient_ID" in df.columns:
        df = df.drop_duplicates(subset=["Patient_ID"], keep="first").copy()
    else:
        df = df.drop_duplicates(subset=[pid_col], keep="first").copy()

    drop_cols = [
        "patient_id_1", "age_1", "gender_1", "drug_name_1", "Weight_lbs",
        "blood_pressure", "Notes", "Extra_Col_1", "Extra_Col_2", "unnamed_0",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    final_cols = [
        "patient_id", "age", "gender", "ethnicity", "weight_kg", "height_cm", "bmi",
        "systolic_bp", "diastolic_bp", "heart_rate", "temperature_f", "hemoglobin",
        "wbc_count", "alt_enzyme", "ast_enzyme", "creatinine", "egfr", "hba1c",
        "total_cholesterol", "drug_name", "dosage_mg", "duration_days", "route",
        "concurrent_drugs", "diagnosis", "smoking_status", "alcohol_use", "admission_date",
        "treatment_outcome", "adverse_event", "readmission_30d",
    ]
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
        "readmission_30d": ["readmission_30d", "readmission30d"],
    }
    current_cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for target in final_cols:
        if target in df.columns:
            rename_map[target] = target
            continue
        for alias in aliases[target]:
            if alias in current_cols:
                rename_map[current_cols[alias]] = target
                break
    df = df.rename(columns=rename_map)

    missing_required = [c for c in final_cols if c not in df.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns after renaming: {missing_required}")
    df = df[final_cols].copy()

    print("Progress: Standardizing missing placeholders and category labels...")
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].apply(lambda x: pd.NA if to_na_like(x) else x)

    def norm_text(value):
        if pd.isna(value):
            return value
        return str(value).strip().lower()

    if "gender" in df.columns:
        gender_map = {
            "m": "male", "male": "male", "mal": "male",
            "f": "female", "female": "female", "femal": "female",
            "o": "other", "other": "other", "1": "male", "0": "female",
        }
        df["gender"] = df["gender"].apply(norm_text).map(lambda x: gender_map.get(x, x) if pd.notna(x) else x)
    if "ethnicity" in df.columns:
        eth_map = {
            "white": "white", "caucasian": "white",
            "black": "black", "african american": "black", "african-american": "black", "aa": "black",
            "hispanic": "hispanic_or_latino", "latino": "hispanic_or_latino",
            "latina": "hispanic_or_latino", "hispanic/latino": "hispanic_or_latino",
        }
        df["ethnicity"] = df["ethnicity"].apply(norm_text).map(lambda x: eth_map.get(x, x) if pd.notna(x) else x)
    if "drug_name" in df.columns:
        drug_map = {
            "warfarine": "warfarin", "metformine": "metformin", "met formin": "metformin",
            "omeprazol": "omeprazole", "gabapentine": "gabapentin",
            "amoxycillin": "amoxicillin", "amoxicilin": "amoxicillin",
            "lisinipril": "lisinopril", "lisino pril": "lisinopril",
            "ibuprofin": "ibuprofen", "ibuprophen": "ibuprofen",
            "atorvastatine": "atorvastatin", "atorvastain": "atorvastatin",
            "tramadole": "tramadol", "insulin glargine": "insulin glargine", "none": "no_drug",
        }
        df["drug_name"] = df["drug_name"].apply(norm_text)
        df["drug_name"] = df["drug_name"].str.replace(r"\s+", " ", regex=True)
        df["drug_name"] = df["drug_name"].map(lambda x: drug_map.get(x, x) if pd.notna(x) else x)
    if "route" in df.columns:
        route_map = {"iv": "intravenous", "im": "intramuscular", "sc": "subcutaneous"}
        df["route"] = df["route"].apply(norm_text).map(lambda x: route_map.get(x, x) if pd.notna(x) else x)
    if "diagnosis" in df.columns:
        diag_map = {
            "htn": "hypertension", "dm2": "type 2 diabetes", "t2dm": "type 2 diabetes",
            "afib": "atrial fibrillation", "ckd": "chronic kidney disease", "oa": "osteoarthritis",
            "hld": "hyperlipidemia", "chf": "heart failure",
        }
        df["diagnosis"] = df["diagnosis"].apply(norm_text).map(lambda x: diag_map.get(x, x) if pd.notna(x) else x)
    if "smoking_status" in df.columns:
        smoke_map = {
            "yes": "current", "y": "current", "true": "current", "1": "current", "current": "current",
            "no": "never", "n": "never", "false": "never", "0": "never", "never": "never",
            "former": "former", "ex smoker": "former", "ex-smoker": "former", "ex": "former",
        }
        df["smoking_status"] = df["smoking_status"].apply(norm_text).map(lambda x: smoke_map.get(x, x) if pd.notna(x) else x)
    if "alcohol_use" in df.columns:
        df["alcohol_use"] = df["alcohol_use"].apply(norm_text)

    print("Progress: Applying unit conversions and type parsing...")
    numeric_cols = [
        "age", "weight_kg", "height_cm", "bmi", "systolic_bp", "diastolic_bp",
        "heart_rate", "temperature_f", "hemoglobin", "wbc_count", "alt_enzyme",
        "ast_enzyme", "creatinine", "egfr", "hba1c", "total_cholesterol",
        "dosage_mg", "duration_days", "concurrent_drugs",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col == "dosage_mg":
            df[col] = df[col].apply(extract_dosage_mg)
        elif col == "duration_days":
            df[col] = df[col].replace({"ongoing": 365, "chronic": 365, "lifetime": 365})
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "height_cm" in df.columns:
        feet_mask = df["height_cm"].between(4, 8)
        df.loc[feet_mask, "height_cm"] = df.loc[feet_mask, "height_cm"] * 30.48
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    ranges = {
        "age": (0, 120), "weight_kg": (2, 300), "height_cm": (50, 250), "bmi": (5, 100),
        "systolic_bp": (50, 250), "diastolic_bp": (30, 150), "heart_rate": (20, 250),
        "temperature_f": (90, 110), "hemoglobin": (3, 25), "wbc_count": (500, 100000),
        "alt_enzyme": (0, 1000), "ast_enzyme": (0, 1000), "creatinine": (0.1, 20),
        "egfr": (0, 200), "hba1c": (2, 25), "total_cholesterol": (50, 1000),
        "dosage_mg": (0, 5000), "duration_days": (0, 3650), "concurrent_drugs": (0, 50),
    }
    for col, (low, high) in ranges.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = pd.NA

    if {"bmi", "weight_kg", "height_cm"}.issubset(df.columns):
        # Keep BMI processing in float64 with np.nan for missing values.
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").astype("float64")
        weight = pd.to_numeric(df["weight_kg"], errors="coerce").astype("float64")
        height = pd.to_numeric(df["height_cm"], errors="coerce").astype("float64")

        bmi_bad = df["bmi"].isna() | (df["bmi"] < 5) | (df["bmi"] > 100)
        valid_height = height > 0

        recomputed = pd.Series(np.nan, index=df.index, dtype="float64")
        recomputed.loc[valid_height] = weight.loc[valid_height] / ((height.loc[valid_height] / 100.0) ** 2)

        # Accept recalculated BMI only when finite and within 5..100.
        recomputed = recomputed.where(np.isfinite(recomputed) & recomputed.between(5, 100), np.nan)
        update_mask = bmi_bad & recomputed.notna()
        df.loc[update_mask, "bmi"] = recomputed.loc[update_mask]

    if "admission_date" in df.columns:
        try:
            df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce", format="mixed")
        except Exception:
            df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce", infer_datetime_format=True)

    print("Progress: Parsing binary outcome columns...")
    for col in ["treatment_outcome", "adverse_event"]:
        if col in df.columns:
            df[col] = df[col].apply(map_binary)
    if "readmission_30d" in df.columns:
        readm_num = pd.to_numeric(df["readmission_30d"], errors="coerce")
        readm_clean = readm_num.where(readm_num.isin([0, 1]), other=pd.NA)
        try:
            df["readmission_30d"] = readm_clean.astype("Int64")
        except Exception:
            df["readmission_30d"] = readm_clean.astype("Float64")

    if len(df.columns) != 31:
        raise SystemExit(f"Prepared dataset has {len(df.columns)} columns, expected 31.")
    validate_patient_ids(df, "Scenario3 prepared base")

    report_lines.append(f"Prepared base dataset shape: rows={len(df):,}, columns={len(df.columns)}")
    report_lines.append(f"Preserved missing {TARGET} rows: {int(df[TARGET].isna().sum()):,}")
    return df


def compare_against_existing_cleaned(df_new, df_existing, report_lines):
    """Validate patient IDs and treatment_outcome values against the existing cleaned dataset."""
    print("Progress: Comparing patient IDs and treatment_outcome with the existing cleaned dataset...")
    validate_patient_ids(df_new, "Scenario3 prepared base")
    validate_patient_ids(df_existing, "Existing cleaned dataset")

    discrepancies = []
    if list(df_existing.columns) != list(df_new.columns):
        discrepancies.append("Column order differs from existing cleaned dataset.")

    new_ids = set(df_new["patient_id"])
    old_ids = set(df_existing["patient_id"])
    only_new = len(new_ids - old_ids)
    only_old = len(old_ids - new_ids)
    report_lines.append(f"patient_id only in scenario3 input: {only_new}")
    report_lines.append(f"patient_id only in existing cleaned data: {only_old}")
    if only_new or only_old:
        discrepancies.append("patient_id sets differ between scenario3 input and existing cleaned dataset.")

    merged = df_new[["patient_id", "treatment_outcome"]].merge(
        df_existing[["patient_id", "treatment_outcome"]],
        on="patient_id",
        how="inner",
        suffixes=("_scenario3", "_existing"),
        validate="one_to_one",
    )
    scenario3_target = merged["treatment_outcome_scenario3"]
    existing_target = merged["treatment_outcome_existing"]
    both_missing = scenario3_target.isna() & existing_target.isna()
    equal_known = (scenario3_target == existing_target).fillna(False)
    target_match_mask = both_missing | equal_known
    mismatches = int((~target_match_mask).fillna(True).sum())
    report_lines.append(f"treatment_outcome mismatches for matched patient_id rows: {mismatches}")
    if mismatches > 0:
        discrepancies.append("treatment_outcome mismatches found between scenario3 input and existing cleaned dataset.")
        sample = merged.loc[(~target_match_mask).fillna(True), ["patient_id", "treatment_outcome_scenario3", "treatment_outcome_existing"]].head(10)
        report_lines.append(f"Sample target mismatches: {sample.to_dict(orient='records')}")
    return discrepancies


def build_split_assignments(df_base, df_engineered_ref, report_lines):
    """Create labelled split assignments using the exact Step 09 row order and recipe."""
    print("Progress: Building exact Step 09 split assignments from engineered labelled order...")

    validate_patient_ids(df_engineered_ref, "Existing engineered dataset")
    validate_patient_ids(df_base, "Scenario3 prepared base")

    ref_labelled = df_engineered_ref[df_engineered_ref[TARGET].notna()][["patient_id", TARGET]].copy()
    ref_y = pd.to_numeric(ref_labelled[TARGET], errors="coerce")
    if ref_y.isna().any():
        raise SystemExit("Existing engineered labelled rows contain invalid treatment_outcome values.")
    ref_y = ref_y.astype(int)

    # Validate ids/targets agree with new input for labelled rows.
    new_targets = df_base[["patient_id", TARGET]].copy()
    merged_ref_new = ref_labelled.merge(
        new_targets,
        on="patient_id",
        how="left",
        suffixes=("_ref", "_new"),
        validate="one_to_one",
    )
    missing_in_new = int(merged_ref_new[f"{TARGET}_new"].isna().sum())
    both_missing = merged_ref_new[f"{TARGET}_ref"].isna() & merged_ref_new[f"{TARGET}_new"].isna()
    equal_known = (merged_ref_new[f"{TARGET}_ref"] == merged_ref_new[f"{TARGET}_new"]).fillna(False)
    target_match = both_missing | equal_known
    target_mismatches = int((~target_match).fillna(True).sum())
    report_lines.append(f"Reference labelled rows missing in scenario3 input: {missing_in_new}")
    report_lines.append(f"Reference-vs-scenario3 labelled target mismatches: {target_mismatches}")
    if missing_in_new or target_mismatches:
        if target_mismatches > 0:
            sample_bad = merged_ref_new.loc[(~target_match).fillna(True), ["patient_id", f"{TARGET}_ref", f"{TARGET}_new"]].head(10)
            report_lines.append(f"Sample reference-vs-scenario3 mismatches: {sample_bad.to_dict(orient='records')}")
        return None, [
            "Scenario3 input does not fully agree with data_engineered labelled patient IDs/targets used by Step 09."
        ]

    temp_ids, test_ids, temp_y, test_y = train_test_split(
        ref_labelled[["patient_id"]], ref_y, test_size=0.20, stratify=ref_y, random_state=42
    )
    train_ids, val_ids, train_y, val_y = train_test_split(
        temp_ids, temp_y, test_size=0.25, stratify=temp_y, random_state=42
    )

    split_train = train_ids.copy()
    split_train["split"] = "train"
    split_val = val_ids.copy()
    split_val["split"] = "validation"
    split_test = test_ids.copy()
    split_test["split"] = "test"
    assignments = pd.concat([split_train, split_val, split_test], ignore_index=True)

    train_set = set(split_train["patient_id"])
    val_set = set(split_val["patient_id"])
    test_set = set(split_test["patient_id"])
    overlaps = {
        "train_validation": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "validation_test": len(val_set & test_set),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Split overlap detected: {overlaps}")

    print(split_summary("Train", train_y))
    print(split_summary("Validation", val_y))
    print(split_summary("Test", test_y))
    report_lines.append(split_summary("Train", train_y))
    report_lines.append(split_summary("Validation", val_y))
    report_lines.append(split_summary("Test", test_y))
    report_lines.append(f"Patient ID overlap counts: {overlaps}")

    # Final assignment integrity checks.
    labelled_base_ids = set(df_base.loc[df_base[TARGET].notna(), "patient_id"])
    assigned_ids = set(assignments["patient_id"])
    missing_assigned = len(labelled_base_ids - assigned_ids)
    extra_assigned = len(assigned_ids - labelled_base_ids)
    if missing_assigned or extra_assigned:
        report_lines.append(f"Assignment coverage issue: missing_labelled={missing_assigned}, extra_nonlabelled={extra_assigned}")
        return None, ["Split assignments do not cover labelled patients exactly once."]

    return assignments, []


def main():
    print("Progress: Loading the full raw clinical dataset...")
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")
    if not EXISTING_CLEANED_FILE.exists():
        raise FileNotFoundError(f"Existing cleaned dataset not found: {EXISTING_CLEANED_FILE}")
    if not EXISTING_ENGINEERED_FILE.exists():
        raise FileNotFoundError(f"Existing engineered dataset not found: {EXISTING_ENGINEERED_FILE}")

    raw_df = pd.read_parquet(RAW_FILE)
    existing_cleaned_df = pd.read_parquet(EXISTING_CLEANED_FILE)
    engineered_ref_df = pd.read_parquet(EXISTING_ENGINEERED_FILE, columns=["patient_id", TARGET])

    report_lines = []
    report_lines.append("Scenario 3 preparation report")
    report_lines.append("============================")
    report_lines.append(f"Raw source shape: rows={len(raw_df):,}, columns={len(raw_df.columns)}")
    report_lines.append(f"Existing cleaned shape: rows={len(existing_cleaned_df):,}, columns={len(existing_cleaned_df.columns)}")
    report_lines.append("This scenario reuses deterministic cleaning, but leaves predictor missing values missing.")
    report_lines.append("It does not apply dataset-wide medians, modes, IQR clipping, or feature engineering.")

    base_df = prepare_base_cleaned_from_raw(raw_df, report_lines)
    discrepancies = []
    discrepancies.extend(compare_against_existing_cleaned(base_df, existing_cleaned_df, report_lines))

    assignments, split_issues = build_split_assignments(base_df, engineered_ref_df, report_lines)
    discrepancies.extend(split_issues)

    # Complete validation first. If discrepancies exist, report and stop before saving either output parquet.
    if discrepancies:
        report_lines.append("")
        report_lines.append("Discrepancies detected:")
        for issue in discrepancies:
            report_lines.append(f"- {issue}")
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
        raise SystemExit(
            f"Validation failed with {len(discrepancies)} discrepancy item(s). "
            f"Report written to {REPORT_FILE}. Outputs were not saved."
        )

    print("Progress: All validations passed. Saving Scenario 3 base dataset and split assignments...")
    OUTPUT_BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_parquet(OUTPUT_BASE_FILE, index=False)
    assignments.to_parquet(OUTPUT_SPLIT_FILE, index=False)

    report_lines.append(f"Scenario 3 base file: {OUTPUT_BASE_FILE}")
    report_lines.append(f"Scenario 3 split assignment file: {OUTPUT_SPLIT_FILE}")
    report_lines.append(f"Scenario 3 base columns: {list(base_df.columns)}")
    report_lines.append(f"Scenario 3 split columns: {list(assignments.columns)}")

    print(f"Prepared dataset rows={len(base_df):,}, columns={len(base_df.columns)}")
    print(f"Split assignment rows={len(assignments):,}, columns={len(assignments.columns)}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Progress: Preparation report written to {REPORT_FILE}")
    print("Done. Review the script before running it.")


if __name__ == "__main__":
    main()
