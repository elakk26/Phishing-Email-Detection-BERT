"""
BERT-based Phishing Email Predictor Engine.
Loads trained model from model/final_model and runs CPU inference on input email text.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Ensure project root directory is in sys.path for direct execution and IDE linting
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config


class PhishingPredictor:
    """
    Inference class for predicting whether an email is PHISHING or LEGITIMATE.
    """
    def __init__(self, model_dir: str = str(config.OUTPUT_MODEL_DIR)):
        self.model_dir = model_dir
        self.device = torch.device("cpu")

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found at '{self.model_dir}'. "
                "Please ensure the fine-tuned BERT model is saved in model/final_model/."
            )

        print(f"[*] Loading PhishingPredictor model from: '{self.model_dir}'")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("[+] PhishingPredictor loaded successfully on CPU!")

    def predict(self, subject: str = "", body: str = "") -> dict:
        """
        Runs inference on given email subject and body.

        Args:
            subject (str): Email subject line.
            body (str): Email body content.

        Returns:
            dict: Structured prediction dictionary containing:
                  - prediction: "PHISHING" or "LEGITIMATE"
                  - confidence: float score (0.0 to 1.0)
                  - probabilities: dict of class probabilities
        """
        # Combine subject and body into a single formatted text
        subject_str = subject.strip() if subject else ""
        body_str = body.strip() if body else ""

        if subject_str and body_str:
            combined_text = f"subject: {subject_str}\n\n{body_str}"
        elif subject_str:
            combined_text = f"subject: {subject_str}"
        else:
            combined_text = body_str

        # Preprocess text (lowercasing as done during training)
        cleaned_text = combined_text.lower()

        if not cleaned_text.strip():
            # Edge case for completely empty input
            return {
                "prediction": "LEGITIMATE",
                "confidence": 0.5000,
                "probabilities": {
                    "LEGITIMATE": 0.5000,
                    "PHISHING": 0.5000
                }
            }

        # Tokenize with training max_length constraint (128)
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=config.MAX_LENGTH,
            truncation=True,
            padding=True
        )

        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Run CPU forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        legit_prob = round(float(probs[0]), 4)
        phish_prob = round(float(probs[1]), 4)

        if phish_prob >= legit_prob:
            predicted_label = "PHISHING"
            confidence = phish_prob
        else:
            predicted_label = "LEGITIMATE"
            confidence = legit_prob

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "LEGITIMATE": legit_prob,
                "PHISHING": phish_prob
            }
        }


def run_demo():
    """
    Demo test runner evaluating one legitimate and one obvious phishing email.
    """
    predictor = PhishingPredictor()

    sample_legitimate = {
        "subject": "Project Status & Q3 Deliverables Update",
        "body": "Hi Team, Please find attached the updated project timeline and Q3 deliverables schedule for our upcoming sprint review meeting on Thursday at 10 AM. Let me know if you have any feedback."
    }

    sample_phishing = {
        "subject": "URGENT: Your Account Has Been Suspended - Verify Password Now!",
        "body": "Dear Customer, We detected unauthorized login attempts on your online bank account. Your account has been temporarily disabled. Please click the link below immediately to re-verify your password and SSN credentials or your account will be closed permanently: http://secure-banking-login-update-verify.com"
    }

    print("\n" + "=" * 65)
    print("  PHISHING PREDICTOR DEMO TEST")
    print("=" * 65)

    print("\n--- Test 1: Clearly Legitimate Email ---")
    print(f"Subject: {sample_legitimate['subject']}")
    print(f"Body   : {sample_legitimate['body']}")
    res_legit = predictor.predict(sample_legitimate['subject'], sample_legitimate['body'])
    print("Result :", res_legit)

    print("\n--- Test 2: Obvious Phishing Email ---")
    print(f"Subject: {sample_phishing['subject']}")
    print(f"Body   : {sample_phishing['body']}")
    res_phish = predictor.predict(sample_phishing['subject'], sample_phishing['body'])
    print("Result :", res_phish)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
