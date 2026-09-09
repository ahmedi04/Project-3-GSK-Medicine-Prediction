#!/usr/bin/env python3
"""Educational treatment-outcome prediction demo API (Step 12).

This Flask API reuses the saved Scenario 3 neural-network model and preprocessing
artifacts exactly as produced in Step 13. It does not retrain or refit anything.

Endpoints:
- GET /health
- POST /predict  with JSON body: {"patient_id": 26319}

Operational notes:
- Uses .env MySQL settings via SQLAlchemy URL.create
- Requires MYSQL_DATABASE == gsk_medicine_db
- Does NOT create tables on import, startup, or /health
- prediction_results table setup is explicit via:
    python app.py --setup-table
  or by running the SQL file in Workbench.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
MODEL_PATH = PROJECT_ROOT / "models" / "scenario3_neural_network.keras"
BUNDLE_PATH = PROJECT_ROOT / "models" / "scenario3_nn_preprocessing.joblib"
SETUP_SQL_PATH = PROJECT_ROOT / "20_prediction_results_setup.sql"

TARGET_DATABASE = "gsk_medicine_db"
MODEL_NAME = "scenario3_neural_network"
MODEL_VERSION = "step13_saved_artifact_v1"
CUTOFF = 0.50

EXCLUDED_FROM_MODEL_INPUT = {
    "patient_id",
    "treatment_outcome",
    "admission_date",
    "adverse_event",
    "readmission_30d",
}

REQUIRED_ENV_KEYS = [
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
]


class ApiError(Exception):
    status_code = 500

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class BadRequestError(ApiError):
    status_code = 400


class NotFoundError(ApiError):
    status_code = 404


class ServiceUnavailableError(ApiError):
    status_code = 503


@dataclass
class PredictionOutput:
    prediction_id: int
    patient_id: int
    model_name: str
    model_version: str
    probability_effective: float
    cutoff: float
    predicted_class: int
    predicted_outcome: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "patient_id": self.patient_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "probability_effective": self.probability_effective,
            "cutoff": self.cutoff,
            "predicted_class": self.predicted_class,
            "predicted_outcome": self.predicted_outcome,
        }


class PredictionService:
    def __init__(self) -> None:
        self.settings: Optional[Dict[str, str]] = None
        self.engine: Optional[Engine] = None

        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.numeric_positions: Optional[np.ndarray] = None
        self.encoded_feature_count: Optional[int] = None
        self.predictor_columns: Optional[List[str]] = None

        self.db_ready = False
        self.model_ready = False
        self.startup_warnings: List[str] = []

    def initialize(self) -> None:
        self._load_database_settings()
        self._build_engine()
        self._load_model_artifacts()

    def health_payload(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "demo": "Educational treatment-outcome prediction demo",
            "database": TARGET_DATABASE,
            "db_ready": self.db_ready,
            "model_ready": self.model_ready,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "warnings": self.startup_warnings,
        }

    def check_database_health(self) -> bool:
        if not self.db_ready or self.engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def get_health_response(self) -> tuple[Dict[str, Any], int]:
        model_ok = bool(
            self.model_ready
            and self.model is not None
            and self.preprocessor is not None
            and self.scaler is not None
            and self.numeric_positions is not None
        )
        db_ok = self.check_database_health()
        is_ok = model_ok and db_ok

        payload = {
            "status": "ok" if is_ok else "unavailable",
            "demo": "Educational treatment-outcome prediction demo",
            "database": TARGET_DATABASE,
            "db_ready": db_ok,
            "model_ready": model_ok,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "warnings": self.startup_warnings,
        }
        return payload, (200 if is_ok else 503)

    def _load_database_settings(self) -> None:
        if not ENV_PATH.exists():
            self.startup_warnings.append("Project .env file is missing.")
            return

        load_dotenv(dotenv_path=ENV_PATH, override=False)

        settings: Dict[str, str] = {}
        missing_keys: List[str] = []

        for key in REQUIRED_ENV_KEYS:
            value = os.getenv(key)
            if value is None:
                missing_keys.append(key)
                continue
            settings[key] = value

        if missing_keys:
            self.startup_warnings.append("Database configuration is incomplete.")
            return

        # Keep password exactly as-is (no .strip()), but validate existence above.
        password = settings["MYSQL_PASSWORD"]
        if password == "":
            self.startup_warnings.append("Database password is empty.")
            return

        try:
            port = int(settings["MYSQL_PORT"].strip())
            if not (1 <= port <= 65535):
                raise ValueError
        except Exception:
            self.startup_warnings.append("MYSQL_PORT is invalid.")
            return

        db_name = settings["MYSQL_DATABASE"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
            self.startup_warnings.append("MYSQL_DATABASE contains invalid characters.")
            return
        if db_name != TARGET_DATABASE:
            self.startup_warnings.append(
                f"MYSQL_DATABASE must be '{TARGET_DATABASE}' for this API."
            )
            return

        settings["MYSQL_PORT"] = str(port)
        settings["MYSQL_DATABASE"] = db_name
        self.settings = settings

    def _build_engine(self) -> None:
        if not self.settings:
            self.db_ready = False
            return

        try:
            db_url = URL.create(
                drivername="mysql+pymysql",
                username=self.settings["MYSQL_USER"],
                password=self.settings["MYSQL_PASSWORD"],
                host=self.settings["MYSQL_HOST"],
                port=int(self.settings["MYSQL_PORT"]),
                database=self.settings["MYSQL_DATABASE"],
            )
            self.engine = create_engine(db_url, pool_pre_ping=True, future=True)
            self.db_ready = True
        except Exception:
            self.engine = None
            self.db_ready = False
            self.startup_warnings.append("Database engine initialization failed.")

    def _load_model_artifacts(self) -> None:
        try:
            importlib.import_module("scenario3_preprocessing_fixed")
            # Defensive aliasing for any historic module path assumptions.
            sys.modules["scenario3_preprocessing"] = sys.modules["scenario3_preprocessing_fixed"]

            if not MODEL_PATH.exists() or not BUNDLE_PATH.exists():
                raise FileNotFoundError("Model or preprocessing artifact is missing.")

            bundle = joblib.load(BUNDLE_PATH)
            model = tf.keras.models.load_model(MODEL_PATH)

            required_keys = {
                "preprocessor",
                "scaler",
                "numeric_positions",
                "encoded_feature_count",
            }
            missing = required_keys - set(bundle.keys())
            if missing:
                raise ValueError(f"Missing preprocessing bundle keys: {sorted(missing)}")

            preprocessor = bundle["preprocessor"]
            predictor_columns = list(getattr(preprocessor, "predictor_columns_", []) or [])
            if len(predictor_columns) != 26:
                from scenario3_preprocessing_fixed import EXPECTED_BASE_PREDICTOR_COLUMNS

                predictor_columns = list(EXPECTED_BASE_PREDICTOR_COLUMNS)

            if len(predictor_columns) != 26:
                raise ValueError("Expected exactly 26 base predictors.")
            if any(column in EXCLUDED_FROM_MODEL_INPUT for column in predictor_columns):
                raise ValueError("Predictor list includes excluded non-model columns.")

            numeric_positions = np.asarray(bundle["numeric_positions"], dtype=np.int32)
            if numeric_positions.ndim != 1:
                raise ValueError("numeric_positions must be a 1D array.")

            self.model = model
            self.preprocessor = preprocessor
            self.scaler = bundle["scaler"]
            self.numeric_positions = numeric_positions
            self.encoded_feature_count = int(bundle["encoded_feature_count"])
            self.predictor_columns = predictor_columns
            self.model_ready = True
        except Exception:
            self.model_ready = False
            self.startup_warnings.append("Model artifacts failed to load.")

    def _require_predict_ready(self) -> None:
        if not self.model_ready or self.model is None or self.preprocessor is None or self.scaler is None:
            raise ServiceUnavailableError("Model is unavailable.")
        if not self.db_ready or self.engine is None:
            raise ServiceUnavailableError("Database is unavailable.")

    def _fetch_patient_predictors(self, patient_id: int) -> pd.DataFrame:
        assert self.engine is not None
        assert self.predictor_columns is not None

        safe_columns = ", ".join(f"`{col}`" for col in self.predictor_columns)
        sql = text(
            f"SELECT {safe_columns} "
            "FROM patients "
            "WHERE patient_id = :patient_id "
            "LIMIT 1"
        )

        try:
            with self.engine.connect() as conn:
                row = conn.execute(sql, {"patient_id": patient_id}).mappings().first()
        except SQLAlchemyError:
            raise ServiceUnavailableError("Database query failed.")

        if row is None:
            raise NotFoundError("Patient not found.")

        row_dict = dict(row)
        return pd.DataFrame([row_dict], columns=self.predictor_columns)

    def _prepare_model_input(self, predictors_df: pd.DataFrame) -> np.ndarray:
        assert self.preprocessor is not None
        assert self.scaler is not None
        assert self.numeric_positions is not None

        encoded = self.preprocessor.transform(predictors_df)
        if sp.issparse(encoded):
            dense = encoded.toarray().astype(np.float32, copy=False)
        else:
            dense = np.asarray(encoded, dtype=np.float32)

        dense[:, self.numeric_positions] = self.scaler.transform(
            dense[:, self.numeric_positions]
        ).astype(np.float32)

        if self.encoded_feature_count is not None and dense.shape[1] != self.encoded_feature_count:
            raise ServiceUnavailableError("Encoded feature count mismatch.")

        if not np.isfinite(dense).all():
            raise ServiceUnavailableError("Model input contains non-finite values.")

        return dense

    def _insert_prediction(
        self,
        patient_id: int,
        probability_effective: float,
        predicted_class: int,
        predicted_outcome: str,
    ) -> int:
        assert self.engine is not None

        insert_sql = text(
            """
            INSERT INTO prediction_results (
                patient_id,
                model_name,
                model_version,
                probability_effective,
                cutoff,
                predicted_class,
                predicted_outcome,
                predicted_at
            ) VALUES (
                :patient_id,
                :model_name,
                :model_version,
                :probability_effective,
                :cutoff,
                :predicted_class,
                :predicted_outcome,
                CURRENT_TIMESTAMP
            )
            """
        )

        try:
            with self.engine.begin() as conn:
                result = conn.execute(
                    insert_sql,
                    {
                        "patient_id": patient_id,
                        "model_name": MODEL_NAME,
                        "model_version": MODEL_VERSION,
                        "probability_effective": probability_effective,
                        "cutoff": CUTOFF,
                        "predicted_class": predicted_class,
                        "predicted_outcome": predicted_outcome,
                    },
                )
                prediction_id = result.lastrowid
        except SQLAlchemyError:
            raise ServiceUnavailableError(
                "Could not save prediction. Ensure prediction_results table exists and database is available."
            )

        if prediction_id is None:
            raise ServiceUnavailableError("Prediction save failed.")

        return int(prediction_id)

    def predict_for_patient(self, patient_id: int) -> PredictionOutput:
        self._require_predict_ready()
        assert self.model is not None

        predictors_df = self._fetch_patient_predictors(patient_id)
        model_input = self._prepare_model_input(predictors_df)

        raw_output = self.model.predict(model_input, verbose=0)
        probabilities = np.asarray(raw_output, dtype=np.float64).reshape(-1)
        if probabilities.size != 1:
            raise ServiceUnavailableError("Prediction service unavailable.")

        probability = float(probabilities[0])
        if (not np.isfinite(probability)) or probability < 0.0 or probability > 1.0:
            raise ServiceUnavailableError("Prediction service unavailable.")

        predicted_class = int(probability >= CUTOFF)
        predicted_outcome = "Effective" if predicted_class == 1 else "Ineffective"

        prediction_id = self._insert_prediction(
            patient_id=patient_id,
            probability_effective=probability,
            predicted_class=predicted_class,
            predicted_outcome=predicted_outcome,
        )

        return PredictionOutput(
            prediction_id=prediction_id,
            patient_id=patient_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            probability_effective=probability,
            cutoff=CUTOFF,
            predicted_class=predicted_class,
            predicted_outcome=predicted_outcome,
        )


def parse_patient_id(payload: Dict[str, Any]) -> int:
    if "patient_id" not in payload:
        raise BadRequestError("Missing required field: patient_id.")

    value = payload["patient_id"]
    if isinstance(value, bool):
        raise BadRequestError("patient_id must be an integer.")

    if isinstance(value, int):
        patient_id = value
    elif isinstance(value, str) and value.strip().isdigit():
        patient_id = int(value.strip())
    else:
        raise BadRequestError("patient_id must be an integer.")

    if patient_id <= 0:
        raise BadRequestError("patient_id must be a positive integer.")

    return patient_id


def read_sql_statements(sql_path: Path) -> List[str]:
    if not sql_path.exists():
        raise FileNotFoundError(f"Missing SQL file: {sql_path}")

    lines = sql_path.read_text(encoding="utf-8").splitlines()
    statements: List[str] = []
    buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []

    if buffer:
        trailing = "\n".join(buffer).strip()
        if trailing:
            statements.append(trailing)

    return statements


def setup_prediction_results_table(service: PredictionService) -> None:
    if not service.db_ready or service.engine is None:
        raise ServiceUnavailableError("Database is unavailable for setup.")

    statements = read_sql_statements(SETUP_SQL_PATH)
    if not statements:
        raise ValueError("No SQL statements found in setup SQL file.")

    try:
        with service.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    except SQLAlchemyError:
        raise ServiceUnavailableError("Table setup SQL execution failed.")


def create_app(service: Optional[PredictionService] = None) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    if service is None:
        service = PredictionService()
        service.initialize()

    app.config["PREDICTION_SERVICE"] = service

    @app.get("/health")
    def health() -> Any:
        svc: PredictionService = app.config["PREDICTION_SERVICE"]
        payload, code = svc.get_health_response()
        return jsonify(payload), code

    @app.post("/predict")
    def predict() -> Any:
        svc: PredictionService = app.config["PREDICTION_SERVICE"]

        try:
            if not request.is_json:
                raise BadRequestError("Request body must be JSON.")

            payload = request.get_json(silent=True)
            if payload is None or not isinstance(payload, dict):
                raise BadRequestError("Invalid JSON body.")

            patient_id = parse_patient_id(payload)
            result = svc.predict_for_patient(patient_id)
            return jsonify(result.to_dict()), 200

        except ApiError as exc:
            return jsonify({"error": exc.message}), exc.status_code
        except Exception:
            return jsonify({"error": "Internal service error."}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 3 Flask prediction API")
    parser.add_argument(
        "--setup-table",
        action="store_true",
        help="Create/check prediction_results table from SQL file (explicit setup only).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for Flask app. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port for Flask app. Default: 5001",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = PredictionService()
    service.initialize()

    if args.setup_table:
        try:
            setup_prediction_results_table(service)
            print("prediction_results setup/check SQL executed successfully.")
            return 0
        except ApiError as exc:
            print(f"ERROR: {exc.message}")
            return 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    app = create_app(service=service)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
