#!/usr/bin/env python3
"""
One-time conversion from the original CSV file to Parquet.

DuckDB performs the conversion efficiently without loading the complete
CSV into a Pandas DataFrame. All later project scripts will use Parquet.
"""

from pathlib import Path
import duckdb


CSV_PATH = Path("outputs/clinical_data_raw.csv")
PARQUET_PATH = Path("outputs/clinical_data_raw.parquet")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Original CSV was not found: {CSV_PATH}"
        )

    PARQUET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only an older generated Parquet copy, if one exists.
    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()

    connection = duckdb.connect()

    print("Converting CSV to Parquet...")

    connection.execute(
        """
        COPY (
            SELECT *
            FROM read_csv_auto(
                'outputs/clinical_data_raw.csv',
                header = true,
                sample_size = -1
            )
        )
        TO 'outputs/clinical_data_raw.parquet'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    # Verify the converted Parquet file.
    row_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM read_parquet(
            'outputs/clinical_data_raw.parquet'
        )
        """
    ).fetchone()[0]

    column_details = connection.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(
            'outputs/clinical_data_raw.parquet'
        )
        """
    ).fetchall()

    column_names = [
        row[0]
        for row in column_details
    ]

    file_size_mb = (
        PARQUET_PATH.stat().st_size
        / (1024 * 1024)
    )

    connection.close()

    print("Parquet conversion complete!")
    print(f"Rows: {row_count}")
    print(f"Columns: {len(column_names)}")
    print(f"Column names: {column_names}")
    print(
        f"Parquet file size: "
        f"{file_size_mb:.2f} MB"
    )
    print(f"Saved to: {PARQUET_PATH}")


if __name__ == "__main__":
    main()