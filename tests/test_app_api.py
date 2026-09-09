import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app import (
    BadRequestError,
    NotFoundError,
    PredictionService,
    PredictionOutput,
    ServiceUnavailableError,
    create_app,
)


class FakeService:
    def __init__(self, mode="ok"):
        self.mode = mode

    def get_health_response(self):
        ok = self.mode == "ok"
        payload = {
            "status": "ok" if ok else "unavailable",
            "demo": "Educational treatment-outcome prediction demo",
            "database": "gsk_medicine_db",
            "db_ready": self.mode != "db_unavailable",
            "model_ready": self.mode != "model_unavailable",
            "model_name": "scenario3_neural_network",
            "model_version": "step13_saved_artifact_v1",
            "warnings": [],
        }
        return payload, (200 if ok else 503)

    def predict_for_patient(self, patient_id: int):
        if self.mode == "not_found":
            raise NotFoundError("Patient not found.")
        if self.mode == "db_unavailable":
            raise ServiceUnavailableError("Database is unavailable.")
        if self.mode == "model_unavailable":
            raise ServiceUnavailableError("Model is unavailable.")
        if self.mode == "bad_request":
            raise BadRequestError("Bad request.")

        return PredictionOutput(
            prediction_id=101,
            patient_id=patient_id,
            model_name="scenario3_neural_network",
            model_version="step13_saved_artifact_v1",
            probability_effective=0.742,
            cutoff=0.50,
            predicted_class=1,
            predicted_outcome="Effective",
        )


class FlaskApiTests(unittest.TestCase):
    def test_health(self):
        app = create_app(service=FakeService(mode="ok"))
        client = app.test_client()

        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("Educational treatment-outcome prediction demo", data["demo"])

    def test_health_db_check_failed(self):
        app = create_app(service=FakeService(mode="db_unavailable"))
        client = app.test_client()

        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertEqual(data["status"], "unavailable")

    def test_predict_requires_json(self):
        app = create_app(service=FakeService(mode="ok"))
        client = app.test_client()

        response = client.post("/predict", data="not-json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.get_json()["error"])

    def test_predict_missing_patient_id(self):
        app = create_app(service=FakeService(mode="ok"))
        client = app.test_client()

        response = client.post("/predict", json={"x": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("patient_id", response.get_json()["error"])

    def test_predict_invalid_patient_id_type(self):
        app = create_app(service=FakeService(mode="ok"))
        client = app.test_client()

        response = client.post("/predict", json={"patient_id": "abc"})
        self.assertEqual(response.status_code, 400)

    def test_predict_unknown_patient(self):
        app = create_app(service=FakeService(mode="not_found"))
        client = app.test_client()

        response = client.post("/predict", json={"patient_id": 26319})
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.get_json()["error"].lower())

    def test_predict_db_unavailable(self):
        app = create_app(service=FakeService(mode="db_unavailable"))
        client = app.test_client()

        response = client.post("/predict", json={"patient_id": 26319})
        self.assertEqual(response.status_code, 503)

    def test_predict_model_unavailable(self):
        app = create_app(service=FakeService(mode="model_unavailable"))
        client = app.test_client()

        response = client.post("/predict", json={"patient_id": 26319})
        self.assertEqual(response.status_code, 503)

    def test_predict_success(self):
        app = create_app(service=FakeService(mode="ok"))
        client = app.test_client()

        response = client.post("/predict", json={"patient_id": 26319})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["patient_id"], 26319)
        self.assertEqual(data["predicted_class"], 1)
        self.assertEqual(data["predicted_outcome"], "Effective")
        self.assertIn("prediction_id", data)


class PredictionServiceProbabilityValidationTests(unittest.TestCase):
    def _build_service_with_probability(self, probability_output):
        service = PredictionService()
        service.model_ready = True
        service.db_ready = True

        service.model = MagicMock()
        service.model.predict = MagicMock(return_value=probability_output)

        service.preprocessor = MagicMock()
        service.preprocessor.transform = MagicMock(return_value=np.array([[0.2, 0.8]], dtype=np.float32))

        service.scaler = MagicMock()
        service.scaler.transform = MagicMock(return_value=np.array([[0.2]], dtype=np.float32))

        service.numeric_positions = np.array([0], dtype=np.int32)
        service.encoded_feature_count = 2
        service.predictor_columns = ["age", "gender"]

        service._fetch_patient_predictors = MagicMock(
            return_value=pd.DataFrame([{"age": 45.0, "gender": "M"}])
        )
        service._insert_prediction = MagicMock(return_value=123)
        return service

    def test_invalid_probability_nan_blocks_insert(self):
        service = self._build_service_with_probability(np.array([[np.nan]], dtype=np.float32))

        with self.assertRaises(ServiceUnavailableError):
            service.predict_for_patient(26319)

        service._insert_prediction.assert_not_called()

    def test_invalid_probability_infinity_blocks_insert(self):
        service = self._build_service_with_probability(np.array([[np.inf]], dtype=np.float32))

        with self.assertRaises(ServiceUnavailableError):
            service.predict_for_patient(26319)

        service._insert_prediction.assert_not_called()

    def test_invalid_probability_out_of_range_blocks_insert(self):
        service = self._build_service_with_probability(np.array([[1.2]], dtype=np.float32))

        with self.assertRaises(ServiceUnavailableError):
            service.predict_for_patient(26319)

        service._insert_prediction.assert_not_called()

    def test_invalid_probability_shape_blocks_insert(self):
        service = self._build_service_with_probability(np.array([[0.3], [0.7]], dtype=np.float32))

        with self.assertRaises(ServiceUnavailableError):
            service.predict_for_patient(26319)

        service._insert_prediction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
