"""
config.py — Central configuration for the Modelling Formulation project.
American Express Campus Challenge 2026 | Coding Patterns Hybrid ML Pipeline

All paths, constants, and experiment settings are defined here.
Nothing is hardcoded elsewhere.
"""

import os
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\verma\Desktop\AmEx"

DATA_PATH = os.path.join(BASE_DIR, "6a3eb196bc7a3_campus_challenge_r1_data.csv")
TEMPLATE_PATH = os.path.join(BASE_DIR, "6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx")
FORMULA_PATH = os.path.join(
    BASE_DIR,
    "amex_campus_challenge_r1-main",
    "amex_campus_challenge_r1-main",
    "pipeline.py"
)

PROJECT_DIR = os.path.join(BASE_DIR, "modelling_formulation")

# Every experiment gets a unique timestamped folder — never overwrite
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments", f"exp_{EXPERIMENT_TIMESTAMP}")

# ─────────────────────────────────────────────
# DATASET CONSTANTS
# ─────────────────────────────────────────────
ID_COL = "id"
FEATURE_COLS = [f"f{i}" for i in range(1, 24)]   # f1 through f23
N_CUSTOMERS = 500_000
TOP_K = 100_000                                    # Top 20% → pseudo-label = 1

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# DATASET A — Formula imputation (Coding Patterns pipeline.py exact policy)
# All features: coerce to numeric, fillna(0)
# This matches: z = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)
# ─────────────────────────────────────────────
FORMULA_IMPUTE_ZERO = FEATURE_COLS    # all features filled with 0

# ─────────────────────────────────────────────
# DATASET B — ML Dataset: raw NaN preserved
# LightGBM and XGBoost handle NaN natively
# Missingness itself is a signal — do not impute
# ─────────────────────────────────────────────
ML_PRESERVE_NAN = True

# ─────────────────────────────────────────────
# PROFILING THRESHOLDS
# ─────────────────────────────────────────────
HIGH_MISSING_THRESHOLD = 0.50     # Features > 50% missing → flag RED
MED_MISSING_THRESHOLD  = 0.20     # Features > 20% missing → flag YELLOW
HIGH_CORR_THRESHOLD    = 0.90     # Feature pairs with |corr| > 0.90 → flag
NEAR_CONSTANT_THRESHOLD = 0.005   # Features with variance < 0.5% of max → flag
Z_SCORE_OUTLIER_THRESHOLD = 3.0   # Z-score threshold for outlier detection
IQR_OUTLIER_MULTIPLIER = 1.5      # Standard IQR multiplier

# ─────────────────────────────────────────────
# ENSEMBLE WEIGHTS (confirmed in Checkpoint 1)
# 80% formula + 10% LightGBM + 10% XGBoost
# Subject to revision after model training
# ─────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "formula": 0.80,
    "lgbm":    0.10,
    "xgb":     0.10,
}

# ─────────────────────────────────────────────
# OUTPUT FILENAMES
# ─────────────────────────────────────────────
FORMULA_DATASET_FILE     = "formula_dataset.parquet"
ML_DATASET_FILE          = "ml_dataset.parquet"
PROFILING_REPORT_FILE    = "profiling_report.txt"
MISSING_REPORT_FILE      = "missing_report.csv"
OUTLIER_REPORT_FILE      = "outlier_report.csv"
CORRELATION_REPORT_FILE  = "correlation_report.csv"
VALIDATION_REPORT_FILE   = "validation_report.txt"
EXPERIMENT_LOG_FILE      = "experiment_log.json"
METADATA_FILE            = "dataset_metadata.json"
