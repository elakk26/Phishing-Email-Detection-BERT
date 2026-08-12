"""
Central configuration settings for Phishing Email Detection BERT pipeline.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Data Files
DATA_FILE = DATA_DIR / "phishing_email.csv"
OUTPUT_MODEL_DIR = MODEL_DIR / "final_model"

# Hyperparameters (Optimized for CPU Laptop Training)
PRETRAINED_MODEL_NAME = "bert-base-uncased"
SAMPLE_SIZE = 5000         # Balanced dataset subset
MAX_LENGTH = 128           # Truncation & padding length for fast CPU attention
NUM_TRAIN_EPOCHS = 3       # Efficient 3 epochs fine-tuning
TRAIN_BATCH_SIZE = 16      # CPU batch size
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5       # Standard AdamW learning rate for BERT
WEIGHT_DECAY = 0.01        # L2 regularization
RANDOM_SEED = 42           # Reproducibility seed

# Train/Val/Test Split Ratios
TEST_SPLIT_SIZE = 0.20     # 20% reserved strictly for final test evaluation (1,000 samples)
VAL_SPLIT_SIZE = 0.125     # 10% of total (12.5% of remaining 80%) = 500 samples
# Training Set = 70% (3,500 samples)

# Label Mappings
LABEL_MAP = {
    0: "LEGITIMATE",
    1: "PHISHING"
}
