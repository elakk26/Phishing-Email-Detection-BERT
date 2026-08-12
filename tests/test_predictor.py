"""
Unit test suite for PhishingPredictor inference engine.
"""

import os
import sys
import unittest

# Ensure project root directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor import PhishingPredictor


class TestPhishingPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PhishingPredictor()

    def test_legitimate_email(self):
        subject = "Team Weekly Standup Meeting Notes"
        body = "Hi all, thanks for attending today's standup. Here are the meeting notes and action items for next week."
        result = self.predictor.predict(subject=subject, body=body)

        self.assertEqual(result["prediction"], "LEGITIMATE")
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertGreater(result["confidence"], 0.70)
        self.assertGreater(result["probabilities"]["LEGITIMATE"], 0.70)

    def test_phishing_email(self):
        subject = "CRITICAL ALERT: Verify Your Account Credentials Now"
        body = "Dear user, your email account password will expire in 2 hours. Click here to confirm your password and avoid termination immediately: http://bit.ly/fake-phish-link"
        result = self.predictor.predict(subject=subject, body=body)

        self.assertEqual(result["prediction"], "PHISHING")
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertGreater(result["confidence"], 0.70)
        self.assertGreater(result["probabilities"]["PHISHING"], 0.70)

    def test_empty_input(self):
        result = self.predictor.predict(subject="", body="")
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)


if __name__ == "__main__":
    unittest.main()
