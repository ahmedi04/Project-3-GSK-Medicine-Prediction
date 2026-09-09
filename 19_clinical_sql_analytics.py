#!/usr/bin/env python3
"""
Step 11: Clinical SQL Analytics Runner

- Reads authoritative SQL blocks from 19_clinical_sql_queries.sql
- Optionally executes queries against gsk_medicine_db.patients
- Prints results to console
- Saves per-query CSV files + summary report to outputs/clinical_sql/

Default behavior is check-only (parsing/validation). Use --run to execute.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import os


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
SQL_FILE = PROJECT_ROOT / "19_clinical_sql_queries.sql"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinical_sql"
REPORT_FILE = OUTPUT_DIR / "clinical_sql_report.txt"

EXPECTED_QUERY_IDS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


@dataclass
class QueryBlock:
    query_id: str
    title: str
    sql: str


def load_environment() -> None:
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)


def require_env(name: str, strip_value: bool = True) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip() if strip_value else value


def get_database_config() -> dict:
    config = {
        "host": require_env("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": require_env("MYSQL_USER"),
        "password": require_env("MYSQL_PASSWORD", strip_value=False),
        "database": require_env("MYSQL_DATABASE"),
    }

    if config["database"] != "gsk_medicine_db":
        raise ValueError(
            "MYSQL_DATABASE must be 'gsk_medicine_db' for Step 11 analytics."
        )

    return config


def parse_sql_file(sql_path: Path) -> Tuple[str, List[QueryBlock]]:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    raw = sql_path.read_text(encoding="utf-8")

    use_match = re.search(r"^\s*USE\s+([A-Za-z0-9_]+)\s*;", raw, flags=re.IGNORECASE | re.MULTILINE)
    if not use_match:
        raise ValueError("SQL file must include a USE <database>; statement.")

    use_database = use_match.group(1)

    blocks: List[QueryBlock] = []
    lines = raw.splitlines()
    i = 0

    while i < len(lines):
        id_match = re.match(r"^\s*--\s*QUERY_ID:\s*(Q\d+)\s*$", lines[i], flags=re.IGNORECASE)
        if not id_match:
            i += 1
            continue

        query_id = id_match.group(1).upper().strip()
        i += 1

        if i >= len(lines):
            raise ValueError(f"Missing QUERY_TITLE for {query_id}")

        title_match = re.match(
            r"^\s*--\s*QUERY_TITLE:\s*(.*?)\s*$",
            lines[i],
            flags=re.IGNORECASE,
        )
        if not title_match:
            raise ValueError(f"Missing QUERY_TITLE for {query_id}")

        title = title_match.group(1).strip()
        i += 1

        sql_lines: List[str] = []
        while i < len(lines):
            current = lines[i]

            if re.match(r"^\s*--\s*QUERY_ID:\s*(Q\d+)\s*$", current, flags=re.IGNORECASE):
                break

            sql_lines.append(current)
            if current.strip().endswith(";"):
                i += 1
                break

            i += 1

        sql = "\n".join(sql_lines).strip()
        if not sql:
            raise ValueError(f"Missing SQL body for {query_id}")
        if not sql.endswith(";"):
            raise ValueError(f"SQL body for {query_id} must end with ';'")

        blocks.append(QueryBlock(query_id=query_id, title=title, sql=sql))

    if not blocks:
        raise ValueError("No query blocks found. Ensure QUERY_ID/QUERY_TITLE markers are present.")

    return use_database, blocks


def validate_query_blocks(blocks: List[QueryBlock]) -> None:
    found_ids = [b.query_id for b in blocks]

    if found_ids != EXPECTED_QUERY_IDS:
        raise ValueError(
            f"Expected query IDs/order {EXPECTED_QUERY_IDS}, found {found_ids}"
        )


def build_engine(config: dict):
    url = URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
        database=config["database"],
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def slugify_title(text_value: str) -> str:
    s = text_value.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def print_dataframe_preview(query_id: str, title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(f"{query_id}: {title}")
    print("=" * 90)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    if df.empty:
        print("(No rows returned)")
        return

    print(df.to_string(index=False, max_rows=50))


def execute_queries_and_export(
    engine,
    use_database: str,
    blocks: List[QueryBlock],
    output_dir: Path,
) -> List[Tuple[QueryBlock, pd.DataFrame, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[QueryBlock, pd.DataFrame, Path]] = []
    with engine.connect() as conn:
        conn.execute(text(f"USE {use_database}"))

        for block in blocks:
            df = pd.read_sql_query(text(block.sql), conn)
            file_name = f"{block.query_id.lower()}_{slugify_title(block.title)}.csv"
            csv_path = output_dir / file_name
            df.to_csv(csv_path, index=False)

            print_dataframe_preview(block.query_id, block.title, df)
            print(f"Saved CSV: {csv_path}")

            results.append((block, df, csv_path))

    return results


def write_text_report(
    report_path: Path,
    use_database: str,
    results: List[Tuple[QueryBlock, pd.DataFrame, Path]],
) -> None:
    lines = []
    lines.append("Clinical SQL Analytics Report")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {use_database}")
    lines.append("")
    lines.append("Interpretation note: Results are observational associations, not causal claims.")
    lines.append("")

    for block, df, csv_path in results:
        lines.append(f"{block.query_id} - {block.title}")
        lines.append(f"Rows returned: {len(df)}")
        lines.append(f"CSV: {csv_path}")
        lines.append("Results table:")
        if df.empty:
            lines.append("(No rows returned)")
        else:
            lines.append(df.to_string(index=False))
        lines.append("-" * 80)

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 11 clinical SQL analytics from authoritative SQL file."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute SQL queries and export outputs. Without this flag, only parse/validate SQL.",
    )
    args = parser.parse_args()

    try:
        load_environment()

        use_database, query_blocks = parse_sql_file(SQL_FILE)
        validate_query_blocks(query_blocks)

        db_name_env = require_env("MYSQL_DATABASE") if ENV_FILE.exists() else os.getenv("MYSQL_DATABASE")
        if db_name_env and db_name_env.strip() != use_database:
            raise ValueError(
                f"SQL USE database ('{use_database}') must match MYSQL_DATABASE ('{db_name_env.strip()}')."
            )

        if use_database != "gsk_medicine_db":
            raise ValueError("SQL file must target USE gsk_medicine_db;")

        print(f"SQL validation passed: {len(query_blocks)} query blocks found ({', '.join(EXPECTED_QUERY_IDS)}).")
        print(f"Authoritative SQL source: {SQL_FILE}")
        print(f"USE database: {use_database}")

        if not args.run:
            print("Check-only mode complete. No queries were executed.")
            return 0

        config = get_database_config()
        engine = build_engine(config)

        results = execute_queries_and_export(
            engine=engine,
            use_database=use_database,
            blocks=query_blocks,
            output_dir=OUTPUT_DIR,
        )
        write_text_report(REPORT_FILE, use_database, results)

        print("\n" + "=" * 90)
        print(f"Saved report: {REPORT_FILE}")
        print("Completed clinical SQL analytics successfully.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
