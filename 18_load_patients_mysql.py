#!/usr/bin/env python3
"""Load Project 3 patient data into MySQL table `patients`.

Modes:
- default (no args): validate source + connection, report whether table exists.
- --load: create and populate `gsk_medicine_db.patients` from source parquet.

This script preserves all source rows/columns and does not perform sampling,
cleaning, deduplication, imputation, clipping, or feature engineering.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
SOURCE_FILE = Path("outputs/scenario3_validation_input.parquet")
REPORT_FILE = Path("outputs/mysql_patient_load_report.txt")

REQUIRED_KEYS = [
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
]

TARGET_DB = "gsk_medicine_db"
TARGET_TABLE = "patients"
EXPECTED_SHAPE = (1_050_000, 31)
EXPECTED_COLUMNS = [
    "patient_id",
    "age",
    "gender",
    "ethnicity",
    "weight_kg",
    "height_cm",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_f",
    "hemoglobin",
    "wbc_count",
    "alt_enzyme",
    "ast_enzyme",
    "creatinine",
    "egfr",
    "hba1c",
    "total_cholesterol",
    "drug_name",
    "dosage_mg",
    "duration_days",
    "route",
    "concurrent_drugs",
    "diagnosis",
    "smoking_status",
    "alcohol_use",
    "admission_date",
    "treatment_outcome",
    "adverse_event",
    "readmission_30d",
]
BINARY_COLS = ["treatment_outcome", "adverse_event", "readmission_30d"]
NUMERIC_MEASUREMENT_COLS = [
    "age",
    "weight_kg",
    "height_cm",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_f",
    "hemoglobin",
    "wbc_count",
    "alt_enzyme",
    "ast_enzyme",
    "creatinine",
    "egfr",
    "hba1c",
    "total_cholesterol",
    "dosage_mg",
    "duration_days",
    "concurrent_drugs",
]
NUMERIC_INSERT_COLS = ["patient_id", *NUMERIC_MEASUREMENT_COLS, *BINARY_COLS]
BATCH_SIZE = 5_000

# Explicit MySQL schema (based on source dtypes + observed text/date profile)
SCHEMA_COLUMNS_SQL = [
    "patient_id BIGINT NOT NULL",
    "age DOUBLE NULL",
    "gender VARCHAR(16) NULL",
    "ethnicity VARCHAR(64) NULL",
    "weight_kg DOUBLE NULL",
    "height_cm DOUBLE NULL",
    "bmi DOUBLE NULL",
    "systolic_bp DOUBLE NULL",
    "diastolic_bp DOUBLE NULL",
    "heart_rate DOUBLE NULL",
    "temperature_f DOUBLE NULL",
    "hemoglobin DOUBLE NULL",
    "wbc_count DOUBLE NULL",
    "alt_enzyme DOUBLE NULL",
    "ast_enzyme DOUBLE NULL",
    "creatinine DOUBLE NULL",
    "egfr DOUBLE NULL",
    "hba1c DOUBLE NULL",
    "total_cholesterol DOUBLE NULL",
    "drug_name VARCHAR(64) NULL",
    "dosage_mg DOUBLE NULL",
    "duration_days DOUBLE NULL",
    "route VARCHAR(32) NULL",
    "concurrent_drugs DOUBLE NULL",
    "diagnosis VARCHAR(128) NULL",
    "smoking_status VARCHAR(32) NULL",
    "alcohol_use VARCHAR(32) NULL",
    "admission_date DATE NULL",
    "treatment_outcome TINYINT NULL",
    "adverse_event TINYINT NULL",
    "readmission_30d TINYINT NULL",
    "PRIMARY KEY (patient_id)",
]

TEXT_LIMITS = {
    "gender": 16,
    "ethnicity": 64,
    "drug_name": 64,
    "route": 32,
    "diagnosis": 128,
    "smoking_status": 32,
    "alcohol_use": 32,
}


# -----------------------------------------------------------------------------
# Shared connection handling (aligned with 17_mysql_connection_check.py)
# -----------------------------------------------------------------------------


def load_mysql_settings() -> Dict[str, str]:
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"Missing .env at {ENV_PATH}. Create it from .env.example and set MySQL values."
        )

    load_dotenv(dotenv_path=ENV_PATH, override=False)

    settings: Dict[str, str] = {}
    missing: List[str] = []
    for key in REQUIRED_KEYS:
        value = os.getenv(key)
        if value is None:
            missing.append(key)
        else:
            settings[key] = value

    if missing:
        raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")

    try:
        port_value = int(settings["MYSQL_PORT"])
    except ValueError as exc:
        raise ValueError("MYSQL_PORT must be an integer.") from exc
    if not (1 <= port_value <= 65535):
        raise ValueError("MYSQL_PORT must be between 1 and 65535.")

    db_name = settings["MYSQL_DATABASE"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
        raise ValueError("MYSQL_DATABASE must contain only letters, numbers, and underscore.")
    if db_name != TARGET_DB:
        raise ValueError(
            f"MYSQL_DATABASE must be exactly '{TARGET_DB}' for this loader. Found '{db_name}'."
        )

    settings["MYSQL_PORT"] = str(port_value)
    return settings


def sanitize_error_message(message: str, password: str) -> str:
    if password:
        return message.replace(password, "***")
    return message


def build_server_url(settings: Dict[str, str]) -> URL:
    return URL.create(
        drivername="mysql+pymysql",
        username=settings["MYSQL_USER"],
        password=settings["MYSQL_PASSWORD"],
        host=settings["MYSQL_HOST"],
        port=int(settings["MYSQL_PORT"]),
        database=None,
    )


def build_database_url(settings: Dict[str, str]) -> URL:
    return URL.create(
        drivername="mysql+pymysql",
        username=settings["MYSQL_USER"],
        password=settings["MYSQL_PASSWORD"],
        host=settings["MYSQL_HOST"],
        port=int(settings["MYSQL_PORT"]),
        database=settings["MYSQL_DATABASE"],
    )


def connect_and_validate_server(settings: Dict[str, str]):
    server_engine = create_engine(build_server_url(settings), future=True, pool_pre_ping=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        server_engine.dispose()
        raise
    return server_engine


def create_database_if_missing(server_engine, db_name: str) -> None:
    statement = f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    with server_engine.begin() as conn:
        conn.execute(text(statement))


def table_exists(db_engine, database_name: str, table_name: str) -> bool:
    sql = text(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """
    )
    with db_engine.connect() as conn:
        count = conn.execute(sql, {"schema_name": database_name, "table_name": table_name}).scalar_one()
    return int(count) > 0


# -----------------------------------------------------------------------------
# Source validation and conversion
# -----------------------------------------------------------------------------


def read_source() -> pd.DataFrame:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source parquet: {SOURCE_FILE}")

    print("Progress: Reading source parquet into memory...")
    df = pd.read_parquet(SOURCE_FILE)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Source shape mismatch. Expected {EXPECTED_SHAPE}, found {df.shape}")

    if len(df.columns) != 31:
        raise ValueError(f"Expected 31 source columns, found {len(df.columns)}")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Source column names/order mismatch. "
            f"Expected {EXPECTED_COLUMNS}, found {list(df.columns)}"
        )

    return df


def validate_patient_id(df: pd.DataFrame) -> None:
    if "patient_id" not in df.columns:
        raise ValueError("Source is missing patient_id column.")
    if df["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values.")
    if not df["patient_id"].is_unique:
        raise ValueError("patient_id must be unique; duplicates detected.")

    pid_numeric = pd.to_numeric(df["patient_id"], errors="coerce")
    if pid_numeric.isna().any():
        raise ValueError("patient_id contains non-numeric values.")

    # Integer-valued check
    non_integer = (pid_numeric % 1 != 0)
    if bool(non_integer.any()):
        raise ValueError("patient_id contains non-integer numeric values.")


def validate_binary_nullable_columns(df: pd.DataFrame) -> None:
    for col in BINARY_COLS:
        if col not in df.columns:
            raise ValueError(f"Source is missing required column: {col}")
        original = df[col]
        s = pd.to_numeric(original, errors="coerce")

        # Reject any originally non-missing value that became NaN during conversion.
        became_nan = original.notna() & s.isna()
        if bool(became_nan.any()):
            sample = original[became_nan].head(5).tolist()
            raise ValueError(
                f"Column {col} has non-missing values that failed numeric conversion. Sample values: {sample}"
            )

        invalid_mask = (~s.isna()) & (~s.isin([0, 1]))
        if bool(invalid_mask.any()):
            sample = df.loc[invalid_mask, col].head(5).tolist()
            raise ValueError(
                f"Column {col} contains values outside {{0,1,NULL}}. Sample invalid values: {sample}"
            )


def validate_numeric_measurements(df: pd.DataFrame) -> None:
    """Reject invalid/non-finite numeric measurements before any table creation."""
    for col in NUMERIC_MEASUREMENT_COLS:
        if col not in df.columns:
            raise ValueError(f"Source is missing expected numeric column: {col}")
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")

        became_nan = original.notna() & converted.isna()
        if bool(became_nan.any()):
            sample = original[became_nan].head(5).tolist()
            raise ValueError(
                f"Column {col} has non-missing values that failed numeric conversion. Sample values: {sample}"
            )

        non_missing = ~converted.isna()
        non_finite = non_missing & ~np.isfinite(converted)
        if bool(non_finite.any()):
            sample = converted[non_finite].head(5).tolist()
            raise ValueError(f"Column {col} contains non-finite numeric values. Sample values: {sample}")


def validate_text_lengths(df: pd.DataFrame) -> None:
    for col, max_len in TEXT_LIMITS.items():
        if col not in df.columns:
            raise ValueError(f"Source is missing expected text column: {col}")

        s = df[col].astype("string")
        lengths = s.str.len()
        too_long = lengths > max_len
        if bool(too_long.fillna(False).any()):
            observed = int(lengths.max(skipna=True) or 0)
            raise ValueError(
                f"Column {col} exceeds VARCHAR({max_len}) limit. Observed max length: {observed}"
            )


def convert_admission_date(df: pd.DataFrame) -> pd.Series:
    if "admission_date" not in df.columns:
        raise ValueError("Source is missing admission_date column.")

    original = df["admission_date"]
    converted = pd.to_datetime(original, errors="coerce")

    # Fail if non-missing original values became NaT.
    failed = original.notna() & converted.isna()
    if bool(failed.any()):
        sample = original[failed].head(5).tolist()
        raise ValueError(f"Unexpected admission_date conversion failures. Sample values: {sample}")

    # Ensure no unexpected time component before DATE storage.
    non_na = converted.dropna()
    has_time = (
        (non_na.dt.hour != 0)
        | (non_na.dt.minute != 0)
        | (non_na.dt.second != 0)
        | (non_na.dt.microsecond != 0)
    )
    if bool(has_time.any()):
        raise ValueError("admission_date contains non-midnight time values; refusing DATE conversion.")

    return converted.dt.date


def prepare_dataframe_for_insert(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare an insertion frame while preserving source records and NULLs."""
    out = df.copy()

    # Keep all records and columns; only normalize dtypes for MySQL compatibility.
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="raise").astype("int64")

    # Ensure numeric measurement columns are numeric and finite where non-missing.
    for col in NUMERIC_MEASUREMENT_COLS:
        original = out[col]
        numeric = pd.to_numeric(original, errors="coerce")
        became_nan = original.notna() & numeric.isna()
        if bool(became_nan.any()):
            raise ValueError(f"Column {col} has non-missing values that failed numeric conversion during preparation.")
        non_missing = ~numeric.isna()
        non_finite = non_missing & ~np.isfinite(numeric)
        if bool(non_finite.any()):
            raise ValueError(f"Column {col} contains non-finite values during preparation.")
        out[col] = numeric.astype("float64")

    for col in BINARY_COLS:
        original = out[col]
        out[col] = pd.to_numeric(original, errors="coerce")
        became_nan = original.notna() & out[col].isna()
        if bool(became_nan.any()):
            raise ValueError(f"Column {col} has non-missing values that failed numeric conversion during preparation.")
        invalid = (~out[col].isna()) & (~out[col].isin([0, 1]))
        if bool(invalid.any()):
            raise ValueError(f"Column {col} has invalid values during conversion.")
        out[col] = out[col].astype("Int64")  # nullable integer; later converted to Python None/int

    out["admission_date"] = convert_admission_date(out)

    return out


def validate_preparation_integrity(source_df: pd.DataFrame, prepared_df: pd.DataFrame) -> None:
    """Ensure preparation preserved all columns, rows, IDs, and missingness profile."""
    if list(prepared_df.columns) != list(source_df.columns):
        raise ValueError("Preparation altered column names/order.")
    if len(prepared_df) != len(source_df):
        raise ValueError("Preparation altered row count.")

    source_pid = pd.to_numeric(source_df["patient_id"], errors="coerce").astype("int64")
    prepared_pid = pd.to_numeric(prepared_df["patient_id"], errors="coerce").astype("int64")
    if not source_pid.equals(prepared_pid):
        raise ValueError("Preparation altered patient_id values/order.")

    source_null_counts = source_df.isna().sum()
    prepared_null_counts = prepared_df.isna().sum()
    for col in source_df.columns:
        if int(source_null_counts[col]) != int(prepared_null_counts[col]):
            raise ValueError(
                f"Preparation changed missing-value count for {col}: "
                f"source={int(source_null_counts[col])}, prepared={int(prepared_null_counts[col])}"
            )


# -----------------------------------------------------------------------------
# DDL, insert, and verification
# -----------------------------------------------------------------------------


def create_patients_table(db_engine) -> None:
    ddl = f"CREATE TABLE `{TARGET_TABLE}` (\n  " + ",\n  ".join(SCHEMA_COLUMNS_SQL) + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    with db_engine.begin() as conn:
        conn.execute(text(ddl))


def batch_records(df: pd.DataFrame, batch_size: int):
    cols = list(df.columns)
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = df.iloc[start:end].astype(object)
        # Preserve NULLs for SQL by converting pandas missing values to Python None.
        chunk = chunk.where(pd.notna(chunk), None)

        # Verify missing values become Python None and numerics passed to MySQL are finite.
        for col in chunk.columns:
            values = chunk[col].tolist()
            for val in values:
                if col in NUMERIC_INSERT_COLS:
                    if val is None:
                        continue
                    try:
                        fval = float(val)
                    except Exception as exc:
                        raise ValueError(f"Column {col} has non-numeric value in batch serialization: {val!r}") from exc
                    if not np.isfinite(fval):
                        raise ValueError(f"Column {col} has non-finite numeric value in batch serialization: {val!r}")
                if col == "admission_date" and val is not None and not isinstance(val, date):
                    raise ValueError(f"admission_date must serialize as date or None. Found {type(val).__name__}.")
                if val is not None and pd.isna(val):
                    raise ValueError(f"Column {col} contains missing sentinel not converted to None in batch serialization.")

        records = chunk.to_dict(orient="records")
        yield start, end, cols, records


def insert_all_rows(db_engine, df: pd.DataFrame) -> None:
    cols = list(df.columns)
    column_sql = ", ".join(f"`{c}`" for c in cols)
    value_sql = ", ".join(f":{c}" for c in cols)
    stmt = text(f"INSERT INTO `{TARGET_TABLE}` ({column_sql}) VALUES ({value_sql})")

    total = len(df)
    inserted = 0

    with db_engine.begin() as conn:
        for start, end, _, records in batch_records(df, BATCH_SIZE):
            conn.execute(stmt, records)
            inserted += len(records)
            print(f"Progress: Inserted rows {start + 1:,} to {end:,} of {total:,}...")

    if inserted != total:
        raise RuntimeError(f"Inserted row count mismatch. Expected {total:,}, inserted {inserted:,}")


def fetch_db_null_counts(db_engine, columns: List[str]) -> Dict[str, int]:
    expressions = [f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in columns]
    sql = text(f"SELECT {', '.join(expressions)} FROM `{TARGET_TABLE}`")
    with db_engine.connect() as conn:
        row = conn.execute(sql).mappings().one()
    return {c: int(row[c]) for c in columns}


def fetch_binary_distribution(db_engine, col: str) -> Dict[str, int]:
    sql = text(
        f"""
        SELECT
            SUM(CASE WHEN `{col}` = 0 THEN 1 ELSE 0 END) AS c0,
            SUM(CASE WHEN `{col}` = 1 THEN 1 ELSE 0 END) AS c1,
            SUM(CASE WHEN `{col}` IS NULL THEN 1 ELSE 0 END) AS cnull
        FROM `{TARGET_TABLE}`
        """
    )
    with db_engine.connect() as conn:
        row = conn.execute(sql).mappings().one()
    return {"0": int(row["c0"]), "1": int(row["c1"]), "NULL": int(row["cnull"])}


def verify_loaded_table(db_engine, source_df: pd.DataFrame) -> Dict[str, object]:
    with db_engine.connect() as conn:
        row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM `{TARGET_TABLE}`")).scalar_one())
        distinct_pid = int(conn.execute(text(f"SELECT COUNT(DISTINCT patient_id) FROM `{TARGET_TABLE}`")).scalar_one())

    source_row_count = len(source_df)
    source_distinct_pid = int(source_df["patient_id"].nunique(dropna=True))

    source_null_counts = {c: int(source_df[c].isna().sum()) for c in source_df.columns}
    db_null_counts = fetch_db_null_counts(db_engine, list(source_df.columns))

    source_binary = {}
    db_binary = {}
    for col in BINARY_COLS:
        original = source_df[col]
        s = pd.to_numeric(original, errors="coerce")
        became_nan = original.notna() & s.isna()
        if bool(became_nan.any()):
            raise ValueError(f"Source verification failed: non-missing values became NaN in {col} conversion.")
        source_binary[col] = {
            "0": int((s == 0).sum()),
            "1": int((s == 1).sum()),
            "NULL": int(s.isna().sum()),
        }
        db_binary[col] = fetch_binary_distribution(db_engine, col)

    return {
        "source_row_count": source_row_count,
        "db_row_count": row_count,
        "source_distinct_patient_id": source_distinct_pid,
        "db_distinct_patient_id": distinct_pid,
        "source_null_counts": source_null_counts,
        "db_null_counts": db_null_counts,
        "source_binary": source_binary,
        "db_binary": db_binary,
    }


def write_report(report_data: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("MySQL Patient Load Verification Report")
    lines.append("=====================================")
    lines.append("")
    lines.append(f"Source file: {SOURCE_FILE}")
    lines.append(f"Target database: {TARGET_DB}")
    lines.append(f"Target table: {TARGET_TABLE}")
    lines.append("")

    lines.append("Row count checks")
    lines.append("----------------")
    lines.append(f"Source total rows: {report_data['source_row_count']:,}")
    lines.append(f"Database total rows: {report_data['db_row_count']:,}")
    lines.append(f"Source distinct patient_id: {report_data['source_distinct_patient_id']:,}")
    lines.append(f"Database distinct patient_id: {report_data['db_distinct_patient_id']:,}")
    lines.append("")

    lines.append("Missing-value counts by column (source vs database)")
    lines.append("---------------------------------------------------")
    source_null_counts = report_data["source_null_counts"]
    db_null_counts = report_data["db_null_counts"]
    for col in source_null_counts:
        lines.append(f"{col}: source_NULL={source_null_counts[col]:,}, db_NULL={db_null_counts[col]:,}")
    lines.append("")

    lines.append("Outcome column distributions (0, 1, NULL)")
    lines.append("-----------------------------------------")
    source_binary = report_data["source_binary"]
    db_binary = report_data["db_binary"]
    for col in BINARY_COLS:
        s = source_binary[col]
        d = db_binary[col]
        lines.append(
            f"{col}: source(0={s['0']:,}, 1={s['1']:,}, NULL={s['NULL']:,}) | "
            f"db(0={d['0']:,}, 1={d['1']:,}, NULL={d['NULL']:,})"
        )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Program modes
# -----------------------------------------------------------------------------


def run_default() -> None:
    print("Progress: Running validation mode (no table creation or inserts)...")

    settings = load_mysql_settings()
    password = settings["MYSQL_PASSWORD"]

    df = read_source()
    validate_patient_id(df)
    validate_numeric_measurements(df)
    validate_binary_nullable_columns(df)
    validate_text_lengths(df)
    _ = convert_admission_date(df)
    print("Progress: Source validation checks passed.")

    print("Progress: Validating server and database connections...")
    server_engine = create_engine(build_server_url(settings), future=True, pool_pre_ping=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Server connection failed. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc
    finally:
        server_engine.dispose()

    db_engine = create_engine(build_database_url(settings), future=True, pool_pre_ping=True)
    try:
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        exists = table_exists(db_engine, settings["MYSQL_DATABASE"], TARGET_TABLE)
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Database connection failed. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc
    finally:
        db_engine.dispose()

    print(f"Progress: Table `{TARGET_TABLE}` exists: {exists}")
    print("Done: Validation mode completed. No table created and no data inserted.")


def run_load() -> None:
    print("Progress: Starting full load mode...")

    settings = load_mysql_settings()
    password = settings["MYSQL_PASSWORD"]

    # Server connection + ensure target DB exists.
    print("Progress: Connecting to MySQL server...")
    server_engine = create_engine(build_server_url(settings), future=True, pool_pre_ping=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        create_database_if_missing(server_engine, settings["MYSQL_DATABASE"])
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Server-level setup failed. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc
    finally:
        server_engine.dispose()

    # Read + validate + prepare source.
    df = read_source()
    validate_patient_id(df)
    validate_numeric_measurements(df)
    validate_binary_nullable_columns(df)
    validate_text_lengths(df)
    prepared = prepare_dataframe_for_insert(df)
    validate_preparation_integrity(df, prepared)
    print("Progress: Source validation and conversion checks passed.")

    db_engine = create_engine(build_database_url(settings), future=True, pool_pre_ping=True)
    try:
        if table_exists(db_engine, settings["MYSQL_DATABASE"], TARGET_TABLE):
            print(
                f"ERROR: Table `{TARGET_TABLE}` already exists in `{settings['MYSQL_DATABASE']}`. "
                "Stopping without replace/truncate/append."
            )
            raise SystemExit(1)

        print("Progress: Creating patients table...")
        create_patients_table(db_engine)

        print("Progress: Inserting all rows in batches...")
        insert_all_rows(db_engine, prepared)

        print("Progress: Running post-load verification...")
        verification = verify_loaded_table(db_engine, df)

        # Strict verification checks
        if verification["source_row_count"] != verification["db_row_count"]:
            raise RuntimeError("Post-load row count mismatch.")
        if verification["source_distinct_patient_id"] != verification["db_distinct_patient_id"]:
            raise RuntimeError("Post-load distinct patient_id count mismatch.")

        for col in prepared.columns:
            if verification["source_null_counts"][col] != verification["db_null_counts"][col]:
                raise RuntimeError(f"Post-load NULL count mismatch for column {col}.")

        for col in BINARY_COLS:
            src = verification["source_binary"][col]
            dst = verification["db_binary"][col]
            if src != dst:
                raise RuntimeError(f"Post-load binary distribution mismatch for column {col}.")

        write_report(verification)
        print(f"Done: Load complete and verified. Report saved to {REPORT_FILE}")

    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Database operation failed. {exc.__class__.__name__}: {safe_msg}")
        print(
            "Failure state note: the insert transaction is rolled back, "
            "but if table creation already succeeded, the empty `patients` table may remain. "
            "The script does not auto-delete, replace, or append."
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"ERROR: Load failed: {exc}")
        print(
            "Failure state note: the insert transaction is rolled back, "
            "but if table creation already succeeded, the empty `patients` table may remain. "
            "The script does not auto-delete, replace, or append."
        )
        raise SystemExit(1) from exc
    finally:
        db_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Project 3 patients parquet into MySQL")
    parser.add_argument("--load", action="store_true", help="Create and populate gsk_medicine_db.patients")
    args = parser.parse_args()

    if args.load:
        run_load()
    else:
        run_default()


if __name__ == "__main__":
    main()
