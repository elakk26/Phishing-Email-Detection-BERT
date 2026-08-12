"""
Modular Training Script for Phishing Email Detection using BERT.
Preserves original notebook methodology with CPU-optimized hyperparameter configuration.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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

# Import local configuration
from src import config


class EmailDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for tokenized email text inputs and binary labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_and_clean_data(data_path, sample_size=config.SAMPLE_SIZE, seed=config.RANDOM_SEED):
    """
    Loads dataset, handles column mapping, cleans text, and extracts a balanced sample.
    """
    print(f"[*] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{data_path}'. "
            "Please ensure phishing_email.csv is placed in the data/ directory."
        )

    df = pd.read_csv(data_path)

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        if col.lower() in ['text', 'email text', 'body', 'email_text']:
            col_mapping[col] = 'text'
        elif col.lower() in ['label', 'email type', 'type', 'email_type', 'target']:
            col_mapping[col] = 'label'

    df.rename(columns=col_mapping, inplace=True)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset columns could not be mapped to 'text' and 'label'. Found: {df.columns.tolist()}")

    # Drop null values
    initial_count = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    print(f"[*] Dropped {initial_count - len(df)} null rows. Remaining: {len(df)}")

    # Standardize label values to integer (0 = Legitimate, 1 = Phishing)
    def map_label(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if 'safe' in s or 'legit' in s or s == '0' or s == '0.0':
            return 0
        elif 'phish' in s or s == '1' or s == '1.0':
            return 1
        try:
            val_int = int(float(val))
            if val_int in (0, 1):
                return val_int
        except Exception:
            pass
        return np.nan

    df['label'] = df['label'].apply(map_label)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # Lowercase text (preserving URLs, numbers, special characters)
    df['text'] = df['text'].astype(str).str.lower()

    # Create balanced subset of target sample size
    if len(df) > sample_size:
        n_per_class = sample_size // 2
        df_legit = df[df['label'] == 0].sample(n=min(n_per_class, len(df[df['label'] == 0])), random_state=seed)
        df_phish = df[df['label'] == 1].sample(n=min(n_per_class, len(df[df['label'] == 1])), random_state=seed)
        df_sampled = pd.concat([df_legit, df_phish]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df_sampled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[*] Sampled balanced dataset shape: {df_sampled.shape}")
    print(f"[*] Class distribution: Legitimate (0): {(df_sampled['label']==0).sum()}, Phishing (1): {(df_sampled['label']==1).sum()}")

    return df_sampled


def split_dataset(df, seed=config.RANDOM_SEED):
    """
    Performs stratified 70% Train / 10% Val / 20% Test split.
    """
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Step 1: Split out 20% test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=seed,
        stratify=labels
    )

    # Step 2: Split remaining 80% into 70% train and 10% val (0.125 of 80% = 10% total)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=config.VAL_SPLIT_SIZE,
        random_state=seed,
        stratify=train_val_labels
    )

    print(f"[*] Dataset Splits -> Train: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}")
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def compute_metrics(pred):
    """
    Computes Accuracy, Precision, Recall, and F1-score for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train():
    """
    Executes the fine-tuning pipeline for BERT sequence classification.
    """
    print("=" * 60)
    print("  PHISHING EMAIL DETECTION - BERT FINE-TUNING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df = load_and_clean_data(config.DATA_FILE)

    # 2. Split Data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = split_dataset(df)

    # 3. Tokenize Inputs
    print(f"[*] Initializing Tokenizer: '{config.PRETRAINED_MODEL_NAME}'")
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)

    print(f"[*] Tokenizing datasets (max_length={config.MAX_LENGTH})...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config.MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=config.MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config.MAX_LENGTH)

    # 4. PyTorch Datasets
    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)
    test_dataset = EmailDataset(test_encodings, test_labels)

    # 5. Initialize Pretrained BERT Model
    print(f"[*] Loading Model Architecture: '{config.PRETRAINED_MODEL_NAME}' (num_labels=2)")
    model = BertForSequenceClassification.from_pretrained(
        config.PRETRAINED_MODEL_NAME,
        num_labels=2
    )

    # 6. Define Training Arguments
    os.makedirs(config.OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.RESULTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        optim="adamw_torch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.RANDOM_SEED
    )

    # 7. Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 8. Run Training
    print("[*] Starting Model Fine-Tuning...")
    trainer.train()
    print("[+] Model Fine-Tuning Completed Successfully!")

    # 9. Save Best Model and Tokenizer
    print(f"[*] Saving Fine-Tuned Model & Tokenizer to: '{config.OUTPUT_MODEL_DIR}'")
    model.save_pretrained(config.OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(config.OUTPUT_MODEL_DIR)
    print("[+] Model and Tokenizer Saved Successfully!")

    # 10. Final Evaluation on Test Set
    print("\n" + "=" * 60)
    print("  FINAL MODEL EVALUATION ON TEST SET (1,000 Samples)")
    print("=" * 60)

    test_results = trainer.predict(test_dataset)
    y_pred = test_results.predictions.argmax(axis=1)
    y_true = np.array(test_labels)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"\nTest Set Metrics:")
    print(f"  - Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Precision : {precision:.4f}")
    print(f"  - Recall    : {recall:.4f}")
    print(f"  - F1-Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],\n   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()
