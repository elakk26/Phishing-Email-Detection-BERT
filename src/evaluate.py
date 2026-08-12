"""
Standalone Evaluation Script for Phishing Email Detection BERT Model.
Loads fine-tuned weights from model/final_model and evaluates the test split without retraining.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Ensure project root directory is in sys.path for direct execution and IDE linting
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local configuration and helper from src
from src import config
from src.train import EmailDataset, load_and_clean_data, split_dataset, compute_metrics


def evaluate_model():
    print("=" * 60)
    print("  EVALUATING TRAINED BERT MODEL ON TEST SET")
    print("=" * 60)

    # 1. Load Data and split using the exact configuration seed (42)
    df = load_and_clean_data(config.DATA_FILE)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = split_dataset(df)

    # 2. Check if saved model exists
    if not os.path.exists(config.OUTPUT_MODEL_DIR):
        raise FileNotFoundError(f"Trained model directory not found at '{config.OUTPUT_MODEL_DIR}'.")

    # 3. Load Saved Tokenizer and Model
    print(f"\n[*] Loading Trained Tokenizer & Model from: '{config.OUTPUT_MODEL_DIR}'")
    tokenizer = BertTokenizer.from_pretrained(config.OUTPUT_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(config.OUTPUT_MODEL_DIR)

    # 4. Tokenize Test Set
    print(f"[*] Tokenizing {len(test_texts)} Test Samples (max_length={config.MAX_LENGTH})...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config.MAX_LENGTH)
    test_dataset = EmailDataset(test_encodings, test_labels)

    # 5. Initialize Trainer in Evaluation Mode
    eval_args = TrainingArguments(
        output_dir=str(config.RESULTS_DIR / "eval_temp"),
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        seed=config.RANDOM_SEED
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 6. Run Prediction on Test Set
    print("[*] Running Test Evaluation...")
    test_results = trainer.predict(test_dataset)
    y_pred = test_results.predictions.argmax(axis=1)
    y_true = np.array(test_labels)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # 7. Print Detailed Evaluation Summary
    print("\n" + "=" * 60)
    print("  TEST EVALUATION METRICS")
    print("=" * 60)
    print(f"  - Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Test Precision : {precision:.4f}")
    print(f"  - Test Recall    : {recall:.4f}")
    print(f"  - Test F1-Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Legitimate (TN) : {cm[0][0]}  | False Phishing (FP) : {cm[0][1]}")
    print(f"  False Legitimate (FN): {cm[1][0]}  | True Phishing (TP)  : {cm[1][1]}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_model()
