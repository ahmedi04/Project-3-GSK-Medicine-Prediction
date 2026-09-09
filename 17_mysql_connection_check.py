#!/usr/bin/env python3
"""Project 3 MySQL connection check using SQLAlchemy + PyMySQL + python-dotenv.

This script:
1) Loads MySQL settings from .env (explicit path relative to this script)
2) Connects to MySQL server without selecting a database
3) Creates the configured database if it does not exist
4) Connects to that database and prints basic connection metadata

It never prints the password and avoids exposing credentials in error output.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
import os


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"

REQUIRED_KEYS = [
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
]


def load_mysql_settings() -> dict[str, str]:
    """Load MySQL settings from the project .env file via explicit path."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"Missing .env at {ENV_PATH}. Create it from .env.example and set MySQL values."
        )

    load_dotenv(dotenv_path=ENV_PATH, override=False)

    settings: dict[str, str] = {}
    missing: list[str] = []

    for key in REQUIRED_KEYS:
        value = os.getenv(key)
        if value is None:
            missing.append(key)
            continue
        settings[key] = value

    if missing:
        raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")

    # Validate port safely
    try:
        port_value = int(settings["MYSQL_PORT"])
    except ValueError as exc:
        raise ValueError("MYSQL_PORT must be an integer.") from exc

    if not (1 <= port_value <= 65535):
        raise ValueError("MYSQL_PORT must be between 1 and 65535.")

    settings["MYSQL_PORT"] = str(port_value)

    # Validate database identifier used in CREATE DATABASE statement.
    db_name = settings["MYSQL_DATABASE"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
        raise ValueError("MYSQL_DATABASE must contain only letters, numbers, and underscore.")

    return settings


def sanitize_error_message(message: str, password: str) -> str:
    """Remove password text from errors if present."""
    if password:
        return message.replace(password, "***")
    return message


def build_server_url(settings: dict[str, str]) -> URL:
    """Build URL for server-level connection (without selecting a database)."""
    return URL.create(
        drivername="mysql+pymysql",
        username=settings["MYSQL_USER"],
        password=settings["MYSQL_PASSWORD"],
        host=settings["MYSQL_HOST"],
        port=int(settings["MYSQL_PORT"]),
        database=None,
    )


def build_database_url(settings: dict[str, str]) -> URL:
    """Build URL for database-level connection."""
    return URL.create(
        drivername="mysql+pymysql",
        username=settings["MYSQL_USER"],
        password=settings["MYSQL_PASSWORD"],
        host=settings["MYSQL_HOST"],
        port=int(settings["MYSQL_PORT"]),
        database=settings["MYSQL_DATABASE"],
    )


def create_database_if_missing(server_engine, db_name: str) -> None:
    """Create database if missing using server-level connection."""
    statement = f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    with server_engine.begin() as conn:
        conn.execute(text(statement))


def fetch_connection_metadata(db_engine) -> dict[str, str]:
    """Fetch server and session metadata from selected database connection."""
    with db_engine.connect() as conn:
        server_version = conn.execute(text("SELECT VERSION()")) .scalar_one()
        hostname = conn.execute(text("SELECT @@hostname")) .scalar_one()
        port = conn.execute(text("SELECT @@port")) .scalar_one()
        current_user = conn.execute(text("SELECT CURRENT_USER()")) .scalar_one()
        selected_database = conn.execute(text("SELECT DATABASE()")) .scalar_one()

    return {
        "server_version": str(server_version),
        "hostname": str(hostname),
        "port": str(port),
        "current_user": str(current_user),
        "selected_database": str(selected_database),
    }


def main() -> None:
    print("Progress: Loading MySQL settings from project .env...")

    try:
        settings = load_mysql_settings()
    except Exception as exc:
        print(f"ERROR: Could not load MySQL settings. {exc}")
        raise SystemExit(1) from exc

    password = settings["MYSQL_PASSWORD"]

    # 1) Server-level connection without selecting database
    print("Progress: Connecting to MySQL server (no database selected)...")
    server_url = build_server_url(settings)
    server_engine = create_engine(server_url, future=True, pool_pre_ping=True)

    try:
        with server_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Server connection failed. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc

    # 2) Create database if needed
    print(f"Progress: Ensuring database exists: {settings['MYSQL_DATABASE']}")
    try:
        create_database_if_missing(server_engine, settings["MYSQL_DATABASE"])
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Could not create database. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc
    finally:
        server_engine.dispose()

    # 3) Database-level connection
    print("Progress: Connecting to selected database...")
    db_url = build_database_url(settings)
    db_engine = create_engine(db_url, future=True, pool_pre_ping=True)

    try:
        metadata = fetch_connection_metadata(db_engine)
    except SQLAlchemyError as exc:
        safe_msg = sanitize_error_message(str(exc), password)
        print(f"ERROR: Database connection failed. {exc.__class__.__name__}: {safe_msg}")
        raise SystemExit(1) from exc
    finally:
        db_engine.dispose()

    print("\nMySQL connection confirmed:")
    print(f"- Server version: {metadata['server_version']}")
    print(f"- Hostname: {metadata['hostname']}")
    print(f"- Port: {metadata['port']}")
    print(f"- Current user: {metadata['current_user']}")
    print(f"- Selected database: {metadata['selected_database']}")

    print("\nDone: Connection check completed. No retail_db changes and no patient data loaded.")


if __name__ == "__main__":
    main()
