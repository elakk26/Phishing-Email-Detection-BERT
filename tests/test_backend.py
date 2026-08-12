"""
Unit and integration test suite for FastAPI backend endpoints.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure project root directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.backend.main import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Using TestClient with context manager triggers FastAPI lifespan startup
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_01_health_endpoint(self):
        """Test GET /api/v1/health returns healthy status and model_loaded = True."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_02_predict_legitimate_email(self):
        """Test POST /api/v1/predict with legitimate email content."""
        payload = {
            "subject": "Sprint Review & Planning Meeting Notes",
            "body": "Hi Team, Thanks for the great discussion today. Please find the attached action items and slides for next week's sprint."
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["prediction"], "LEGITIMATE")
        self.assertGreaterEqual(data["confidence"], 0.70)
        self.assertIn("LEGITIMATE", data["probabilities"])
        self.assertIn("PHISHING", data["probabilities"])
        self.assertGreaterEqual(data["probabilities"]["LEGITIMATE"], 0.70)

    def test_03_predict_phishing_email(self):
        """Test POST /api/v1/predict with phishing email content."""
        payload = {
            "subject": "ACTION REQUIRED: Account Deactivation Notice",
            "body": "Dear Customer, We noticed suspicious login activity. Your online access will be terminated in 24 hours unless you verify your password immediately: http://fake-login-verify-account.com"
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["prediction"], "PHISHING")
        self.assertGreaterEqual(data["confidence"], 0.70)
        self.assertIn("LEGITIMATE", data["probabilities"])
        self.assertIn("PHISHING", data["probabilities"])
        self.assertGreaterEqual(data["probabilities"]["PHISHING"], 0.70)

    def test_04_empty_input_validation(self):
        """Test POST /api/v1/predict with empty subject and body returns HTTP 400."""
        payload = {
            "subject": "   ",
            "body": "  "
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)

    def test_05_model_unavailable_service_error(self):
        """Test POST /api/v1/predict when predictor model is uninitialized returns HTTP 503."""
        original_predictor = getattr(app.state, "predictor", None)
        try:
            app.state.predictor = None
            payload = {
                "subject": "Test Subject",
                "body": "Test Body"
            }
            response = self.client.post("/api/v1/predict", json=payload)
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertIn("detail", data)
        finally:
            app.state.predictor = original_predictor


if __name__ == "__main__":
    unittest.main()
