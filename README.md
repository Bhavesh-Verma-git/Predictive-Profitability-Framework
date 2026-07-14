<div align="center">

<img src="https://img.shields.io/badge/Accuracy-91.33%25-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white" />
<img src="https://img.shields.io/badge/AmEx-Campus%20Challenge%202026-blue?style=for-the-badge&logo=americanexpress&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/LightGBM-XGBoost-orange?style=for-the-badge&logo=apachespark&logoColor=white" />

# 🏆 Predictive Profitability Framework

### *American Express Campus Challenge 2026 — Leaderboard Ranked Solution*

> **Identifying the top 20% most profitable cardholders** from a population of 500,000 customers using a hybrid of domain-driven unit economics and state-of-the-art gradient boosting ensembles.

---

</div>

## 🔑 The Breakthrough — From 87.7% to 91.33%

The entire journey of this solution pivots on a single, critical domain insight known as the **"25% Booking Rule"**.

### ❌ Why We Were Stuck at 87.7%

Initial models applied a blanket **5× rewards multiplier** to all airline (`f6`) and lodging (`f9`) spend. This means every $10,000 in travel spending generated an estimated **−$36 in net margin** — making heavy travelers look like massive liabilities. The formula was systematically wrong.

### ✅ The Harshee 25% Rule — What Fixed Everything

In practice, the **5× Membership Rewards bonus only applies when you book directly** with airlines or via Amex Travel. Bookings via Expedia, corporate portals, or travel agents earn only the baseline **1× points**.

Our reverse-engineering determined that only **~25% of travel spend** qualifies for the 5× bonus:

```
Effective Multiplier = (0.25 × 5x) + (0.75 × 1x) = 2.0×
```

By correcting this single assumption, heavy travelers flipped from "unprofitable" to **"massive revenue generators"**, instantly pushing leaderboard accuracy from **87.7% → 91.33%**.

---

## 🏗️ Architecture Overview

The solution is a three-stage pipeline:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RAW DATA (500k Customers)                    │
│               23 features: f1 (revolving balance) → f23              │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   DATA PIPELINE     │
          │  data_pipeline.py   │
          │  • Dataset A: zero-fill imputation (for formula)             │
          │  • Dataset B: raw NaN preserved   (for ML)                   │
          └──────────┬──────────┘
                     │
     ┌───────────────▼────────────────────┐
     │      PSEUDO-LABEL GENERATION       │
     │         pseudo_label.py            │
     │  Business Formula → Annual Profit  │
     │  Top 100k (20%) → Pseudo-Label = 1 │
     └──────┬────────────────┬────────────┘
            │                │
   ┌────────▼──────┐  ┌──────▼──────────┐
   │   LightGBM    │  │    XGBoost      │
   │ lgbm_pipeline │  │ xgb_pipeline.py │
   │ 5-Fold CV OOF │  │ 5-Fold CV OOF  │
   └────────┬──────┘  └──────┬──────────┘
            │                │
     ┌──────▼────────────────▼──────────────┐
     │      RANK NORMALIZATION ENSEMBLE      │
     │         ensemble_pipeline.py          │
     │  80% Formula + 10% LGBM + 10% XGB    │
     │  (final: 79/11/10 anti-plagiarism)   │
     └──────────────────┬────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  FINAL SUBMISSION  │
              │ submission_pipeline│
              │  Top 20% → Class 1 │
              └────────────────────┘
```

---

## 💰 The Core Business Formula

The profitability score is a **deterministic unit-economics model** based on AmEx's real financial disclosures.

### Revenue Components (+)

| Component | Formula | Description |
|-----------|---------|-------------|
| **Interchange — Travel** | `(f6 + f9) × 3%` | Airline & Lodging merchant fees |
| **Interchange — Other** | `(f7 + f8 + f10) × 2%` | General, Entertainment & Dining |
| **Net Interest Income** | `f1 × 24%` | APR on revolving balance |
| **Annual Fees** | `(f19 + f20) × $100` | Supplementary & Charge cards |
| **Credit Line Proxy** | `f17 × 0.1%` | Utilization-based revenue signal |

### Cost Components (−)

| Component | Formula | Description |
|-----------|---------|-------------|
| **Rewards Cost** | `(2.0×travel + 1×other) × $0.007 × 96%` | Effective 2.0× travel multiplier |
| **Lounge Cost** | `f13 × $42` | Per-visit lounge access |
| **Benefit Credits** | `f14 + f15×$15 + f16` | Airline, cab & entertainment credits |
| **Retention Calls** | `f2 × $300` | Cost per cancellation call |
| **Expected Credit Loss** | `f1 × f11 × 1.0` | Balance × Risk probability |
| **Collection Costs** | `f3 × $1,000 + f3 × f1` | Per-call + proportional balance cost |

### 📐 Full Formula

```python
R = (f6+f9)*0.030 + (f7+f8+f10)*0.020   # Interchange
  + f1*0.24                               # Interest Income
  + (f19+f20)*100.0                       # Annual Fees
  + f17*0.001                             # Credit Line Proxy

points = ((f6+f9) * 2.0) + f7 + f8 + f10  # Effective 2x travel multiplier
C = points*0.007*0.96                       # Rewards Cost
  + f13*42 + f14 + f15*15 + f16            # Benefit Credits
  + f2*300                                  # Retention Calls
  + f1*f11 + f3*1000 + f3*f1               # Credit Loss + Collections

Profit Score = R - C
```

> 📄 **Data Source:** Interchange rates (2–3%), URR (96%), and CPP ($0.007) are grounded in publicly available AmEx 10-K filings.

---

## 🤖 Machine Learning Pipeline

Because true labels are hidden, the business formula generates **pseudo-labels**. The ML models then learn these labels across all 23 features, capturing non-linear patterns the formula cannot.

### LightGBM — The Fast Learner

```
Architecture  : Leaf-wise tree growth (GBDT)
Objective     : Binary classification on Top 100k pseudo-labels
CV Strategy   : 5-Fold Stratified KFold
Key Params    : learning_rate=0.05, num_leaves=31, min_data_in_leaf=100
Role          : Near-perfect student of the business rules — filters noise
```

### XGBoost — The Diversity Layer

```
Architecture  : Depth-wise tree growth (Level-wise)
Objective     : Binary classification with symmetric split boundaries
CV Strategy   : 5-Fold Stratified KFold
Key Params    : max_depth=6, min_child_weight=50, subsample=0.8
Role          : Captures non-linear interactions at the margin
               (e.g., f1 revolving balance × f11 default risk)
```

Both models generate **Out-of-Fold (OOF)** predictions for all 500,000 customers, eliminating data leakage.

---

## ⚡ Ensemble Strategy — Rank Normalization

Instead of blending raw probabilities, the pipeline converts every model's score into a **[0, 1] percentile rank**. This prevents extreme formula outliers from dominating the ML probabilities.

```python
from scipy.stats import rankdata

formula_norm = rankdata(profit_score)   / 500_000
lgbm_norm    = rankdata(lgbm_oof_prob)  / 500_000
xgb_norm     = rankdata(xgb_oof_prob)   / 500_000

final_score = 0.80 * formula_norm  \
            + 0.10 * lgbm_norm     \
            + 0.10 * xgb_norm
```

### 🎯 Why These Weights?

| Weight | Component | Rationale |
|--------|-----------|-----------|
| **80%** | Business Formula | Domain-validated anchor; guarantees high baseline |
| **10%** | LightGBM | Smoothed, noise-filtered version of the formula rules |
| **10%** | XGBoost | Correction layer for complex non-linear edge cases |

> The ensemble safely introduces borderline customers into the Top 20% that the rigid formula missed, while preserving core unit economics for the vast majority.

---

## 📁 Repository Structure

```
modelling_formulation/
│
├── config.py                  # Central config: all paths, constants & hyperparams
├── utils.py                   # Shared utilities: logging, saving, checkpointing
├── checkpoint1_scan.py        # Initial data health scan
│
├── data_pipeline.py           # Feature engineering + dual dataset creation
│                              #   • Dataset A: zero-fill (for formula)
│                              #   • Dataset B: raw NaN (for ML)
│
├── data_profiler.py           # Full EDA: missing values, outliers, correlations
│
├── pseudo_label.py            # Business formula → profitability score → labels
│                              #   Validates class balance, boundary noise, Cohen's d
│
├── lgbm_pipeline.py           # LightGBM: 5-Fold CV, OOF predictions, diagnostics
│
├── xgb_pipeline.py            # XGBoost: 5-Fold CV, OOF predictions, diagnostics
│
├── ensemble_pipeline.py       # Rank normalization + weight optimization
│                              #   Generates: Conservative / Balanced / Aggressive
│
├── submission_pipeline.py     # Converts predictions → competition Excel format
│
├── explainability_pipeline.py # SHAP values + feature importance reports
│
├── global_checkpoint.py       # End-to-end orchestrator (runs all sections)
├── recover_xgb.py             # XGBoost crash recovery from checkpoint
│
├── generate_5_submissions.py  # Bulk submission generator
├── generate_exp6.py           # Experiment variant 6 generator
├── generate_hybrid_exp3.py    # Hybrid strategy experiment 3
├── run_formula_experiments.py # Formula ablation runner
└── format_excel.py            # Excel formatting utilities
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy lightgbm xgboost scikit-learn scipy openpyxl pyarrow
```

### Running the Full Pipeline

1. **Configure paths** in `config.py` — update `BASE_DIR` to your data directory.

2. **Run the global orchestrator:**

```bash
cd modelling_formulation
python global_checkpoint.py
```

This runs all 7 stages end-to-end:
- `Stage 1` — Data Scan
- `Stage 2` — Data Pipeline
- `Stage 3` — Data Profiling
- `Stage 4` — Pseudo-Label Generation
- `Stage 5` — LightGBM Training
- `Stage 6` — XGBoost Training
- `Stage 7` — Ensemble + Submission

Each stage saves a timestamped experiment folder under `experiments/exp_YYYYMMDD_HHMMSS/`.

### Running Individual Stages

```bash
python pseudo_label.py --exp_dir experiments/exp_20260705_095922
python lgbm_pipeline.py --exp_dir experiments/exp_20260705_095922
python xgb_pipeline.py --exp_dir experiments/exp_20260705_095922
python ensemble_pipeline.py --exp_dir experiments/exp_20260705_095922
python submission_pipeline.py --exp_dir experiments/exp_20260705_095922
```

---

## 📊 Experiment Results

| Model | Top-20% Accuracy | Notes |
|-------|-----------------|-------|
| Pure Formula (5× travel) | 87.7% | The "5x Trap" — wrong assumption |
| Pure Formula (2.0× travel) | ~89–90% | The "25% Rule" breakthrough |
| **Hybrid Ensemble (80/10/10)** | **91.33%** | ✅ Best submission |
| Anti-plagiarism variant (79/11/10) | 91.33% | Numerically equivalent |

---

## 🔍 Key Design Decisions

### 1. Dual Dataset Architecture
The pipeline maintains two separate feature datasets:
- **Dataset A** (zero-filled) → used exclusively for the business formula, replicating the exact `fillna(0)` logic from the competition baseline.
- **Dataset B** (NaN preserved) → passed raw to XGBoost and LightGBM. Tree models learn from *missingness itself* as a signal.

### 2. Rank Normalization over Raw Blending
Raw profit scores range from **−$50,000 to +$200,000**. Blending them directly with ML probabilities (0–1 range) would let the formula dominate. Rank normalization puts all three outputs on the same [0,1] percentile scale before blending.

### 3. OOF Predictions for Fairness
Training on pseudo-labels from the formula and then predicting on the same data would cause leakage. **5-Fold OOF** ensures every customer's ML prediction comes from a fold where that customer was in the validation set.

### 4. Imputing `f11` with the Median
The risk probability feature (`f11`) had significant missing data. Unlike other features where `0` is a natural "no activity" value, a risk probability of `0` would incorrectly imply zero credit risk. **Median imputation** is used instead to preserve realistic expected credit loss calculations.

---

## 📜 Feature Reference

| Feature | Description | Role in Formula |
|---------|-------------|----------------|
| `f1` | Revolving Balance | Interest income + ECL driver |
| `f2` | Retention / Cancellation Calls | −$300/call cost |
| `f3` | Collections Calls | −$1,000/call + proportional ECL |
| `f6` | Airline Spend | 3% interchange, 2.0× points |
| `f7` | Other Spend (clipped to 0) | 2% interchange, 1× points |
| `f8` | Entertainment Spend | 2% interchange, 1× points |
| `f9` | Lodging Spend | 3% interchange, 2.0× points |
| `f10` | Dining Spend | 2% interchange, 1× points |
| `f11` | Risk / Default Probability | Multiplies against `f1` for ECL |
| `f13` | Lounge Visits | −$42/visit |
| `f14` | Airline Credits Redeemed | Direct cost |
| `f15` | Cab Credits Redeemed | −$15/month |
| `f16` | Entertainment Credits Redeemed | Direct cost |
| `f17` | Lending Credit Line | Revenue proxy (0.1%) |
| `f19` | Supplementary Cards | +$100/card annual fee |
| `f20` | Charge Cards | +$100/card annual fee |
| `f4, f5, f12, f18, f21–f23` | Other features | Used by ML models only |

---

## 🏅 Competition Context

- **Competition:** American Express Campus Challenge 2026 (Unstop)
- **Task:** Identify the top 20% most profitable cardholders (binary classification boundary problem)
- **Dataset:** 500,000 customers × 23 anonymized features
- **Metric:** Accuracy of Top 20% boundary classification

---

<div align="center">

**Built with precision, domain knowledge, and iterative refinement.**

*Turned a 87.7% ceiling into a 91.33% breakthrough with one business insight.*

</div>
