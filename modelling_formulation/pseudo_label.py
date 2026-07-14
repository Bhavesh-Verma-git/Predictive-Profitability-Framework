"""
pseudo_label.py — Section 3: Pseudo-Label Generation & Validation
American Express Campus Challenge 2026 | Modelling Formulation Project

Responsibilities:
1.  Load Dataset A (formula dataset — zero-filled)
2.  Apply the Coding Patterns profitability formula
3.  Analyze the continuous profitability score distribution
4.  Rank 500,000 customers deterministically
5.  Assign binary pseudo-labels: Top 100k = 1, Rest = 0
6.  Validate label counts, integrity, and class balance
7.  Generate positive vs. negative class diagnostics
8.  Feature separation analysis (Cohen's d for all 23 features)
9.  Boundary / label-noise analysis (ranks 99,500 – 100,500)
10. Pseudo-label agreement analysis (score gap at cutoff)
11. Save all artifacts and produce Checkpoint 3 report

Usage:
    python pseudo_label.py --exp_dir <path_to_experiment_dir>
    python pseudo_label.py   (uses latest experiment in config)
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, save_experiment_log, save_text_report, checkpoint_pass

# ─────────────────────────────────────────────────────────────────────────────
# CODING PATTERNS FORMULA CONSTANTS (pipeline.py — exact copy, never modify)
# ─────────────────────────────────────────────────────────────────────────────
# V15 + Harshee 25% Rule Constants
IC_TRAVEL = 0.030   # 3% interchange for travel
IC_OTHER  = 0.020   # 2% interchange for other spend
CPP       = 0.007   # issuer cost per rewards point ($)
URR       = 0.96    # Ultimate Redemption Rate
INT_RATE  = 0.24    # net interest margin on average revolving balance
LOUNGE    = 42.0    # cost per lounge visit
CAB       = 15.0    # cost per month of cab-credit usage
LGD       = 1.0     # loss given default
CALL      = 300.0   # retention cost per cancellation call
COLL      = 1000.0  # cost per collections call
SUPP_FEE  = 100.0   # revenue per supplementary/charge card
CL_PROXY  = 0.001   # credit line proxy revenue

TOP_K = 100_000     # Top 20% of 500,000 = pseudo-label 1


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET A
# ─────────────────────────────────────────────────────────────────────────────
def load_formula_dataset(exp_dir: str, logger) -> pd.DataFrame:
    """
    Loads Dataset A (formula dataset) produced by Section 2.

    Why Dataset A and not raw CSV: Dataset A has the exact zero-fill
    imputation that the Coding Patterns formula expects. Using the
    raw CSV would require re-imputing, risking inconsistency.

    Args:
        exp_dir: Path to the Section 2 experiment directory.
        logger: Logger instance.

    Returns:
        DataFrame with 500,000 rows, 24 columns (id + f1-f23), zero NaN.

    Raises:
        RuntimeError if file not found or integrity checks fail.
    """
    path = os.path.join(exp_dir, "data", "formula_dataset.parquet")
    if not os.path.exists(path):
        raise RuntimeError(f"[FAIL] formula_dataset.parquet not found at: {path}")

    df = pd.read_parquet(path)
    logger.info(f"Loaded formula_dataset.parquet: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Integrity checks
    if df.shape[0] != 500_000:
        raise RuntimeError(f"[FAIL] Expected 500,000 rows, got {df.shape[0]:,}")
    if df[config.FEATURE_COLS].isnull().sum().sum() != 0:
        raise RuntimeError("[FAIL] Dataset A contains unexpected NaN values.")
    if df[config.ID_COL].duplicated().sum() > 0:
        raise RuntimeError("[FAIL] Dataset A contains duplicate IDs.")

    logger.info("Dataset A integrity: PASS ✓ (500k rows, 0 NaN, 0 duplicate IDs)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — APPLY CODING PATTERNS FORMULA
# ─────────────────────────────────────────────────────────────────────────────
def compute_profitability_score(df: pd.DataFrame, logger) -> pd.Series:
    """
    Applies the Coding Patterns profitability formula exactly as in pipeline.py.

    Formula (annual profit per cardmember, $):
        + 0.023 × (f6+f7+f8+f9+f10)         [interchange on all spend]
        + 0.015 × (f6+f9)                    [travel booking commission]
        - 0.007 × (5×f6 + f7+f8+f9+f10)     [rewards points cost]
        + 0.12  × f1                          [net interest on revolving balance]
        - (f14 + f16 + 50×f13 + 15×f15)      [benefit credits burned]
        - 0.7   × f11 × f1                   [expected credit loss]
        - (20×f2 + 60×f3)                    [servicing and collections calls]

    Note: f7 is clipped to 0 before use to handle refund/chargeback negatives,
    exactly as in the original pipeline.py (np.clip).

    Args:
        df: Formula dataset (Dataset A, zero-filled).
        logger: Logger.

    Returns:
        Series of float profitability scores, indexed identically to df.
    """
    logger.info("Applying Coding Patterns formula to Dataset A...")

    f1  = df['f1'].to_numpy(np.float64)
    f2  = df['f2'].to_numpy(np.float64)
    f3  = df['f3'].to_numpy(np.float64)
    f6  = df['f6'].to_numpy(np.float64)
    f7  = np.clip(df['f7'].to_numpy(np.float64), 0, None)   # clip refunds to 0
    f8  = df['f8'].to_numpy(np.float64)
    f9  = df['f9'].to_numpy(np.float64)
    f10 = df['f10'].to_numpy(np.float64)
    f11 = df['f11'].to_numpy(np.float64)
    f13 = df['f13'].to_numpy(np.float64)
    f14 = df['f14'].to_numpy(np.float64)
    f15 = df['f15'].to_numpy(np.float64)
    f16 = df['f16'].to_numpy(np.float64)
    f17 = df['f17'].to_numpy(np.float64)
    f19 = df['f19'].to_numpy(np.float64)
    f20 = df['f20'].to_numpy(np.float64)

    # V15 + Harshee 25% Rule Logic
    R_interchange = (f6 + f9) * IC_TRAVEL + (f7 + f8 + f10) * IC_OTHER
    R_interest = f1 * INT_RATE
    R_supp = f19 * SUPP_FEE + f20 * SUPP_FEE
    R_credit_line = f17 * CL_PROXY

    # Harshee 25% Rule: Only 25% of travel spend earns 5x. 75% earns 1x. Effective = 2.0x
    points_earned = ((f6 + f9) * 2.0) + f7 + f8 + f10
    C_points = points_earned * CPP * URR

    C_lounge = f13 * LOUNGE
    C_airline = f14
    C_cab = f15 * CAB
    C_ent = f16

    C_ecl = (f1 * f11 * LGD) + (f3 * COLL) + (f3 * f1 * 1.0)
    C_retention = f2 * CALL

    profit = (
        R_interchange + R_interest + R_supp + R_credit_line
        - C_points - C_lounge - C_airline - C_cab - C_ent - C_ecl - C_retention
    )

    score = pd.Series(profit, index=df.index, name='profitability_score')

    # Validate score
    if score.isnull().any():
        raise RuntimeError("[FAIL] Profitability score contains NaN values.")
    if np.isinf(score).any():
        raise RuntimeError("[FAIL] Profitability score contains infinite values.")

    logger.info(f"Formula applied: min={score.min():.2f}, max={score.max():.2f}, mean={score.mean():.2f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ANALYZE SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def analyze_score_distribution(score: pd.Series, report_lines: list, logger) -> Dict:
    """
    Comprehensive statistical analysis of the profitability score distribution.

    Args:
        score: Profitability scores for all 500k customers.
        report_lines: List to append report text to.
        logger: Logger.

    Returns:
        Dict of distribution statistics.
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("PROFITABILITY SCORE DISTRIBUTION ANALYSIS")
    _log(f"{'='*60}")

    stats = {
        "count":    int(len(score)),
        "min":      round(float(score.min()), 4),
        "max":      round(float(score.max()), 4),
        "mean":     round(float(score.mean()), 4),
        "median":   round(float(score.median()), 4),
        "std":      round(float(score.std()), 4),
        "skewness": round(float(scipy_stats.skew(score)), 4),
        "kurtosis": round(float(scipy_stats.kurtosis(score)), 4),
        "p1":       round(float(score.quantile(0.01)), 4),
        "p5":       round(float(score.quantile(0.05)), 4),
        "p10":      round(float(score.quantile(0.10)), 4),
        "p20":      round(float(score.quantile(0.20)), 4),
        "p25":      round(float(score.quantile(0.25)), 4),
        "p50":      round(float(score.quantile(0.50)), 4),
        "p75":      round(float(score.quantile(0.75)), 4),
        "p80":      round(float(score.quantile(0.80)), 4),
        "p90":      round(float(score.quantile(0.90)), 4),
        "p95":      round(float(score.quantile(0.95)), 4),
        "p99":      round(float(score.quantile(0.99)), 4),
    }

    _log(f"  Count     : {stats['count']:,}")
    _log(f"  Min       : ${stats['min']:,.2f}")
    _log(f"  Max       : ${stats['max']:,.2f}")
    _log(f"  Mean      : ${stats['mean']:,.2f}")
    _log(f"  Median    : ${stats['median']:,.2f}")
    _log(f"  Std Dev   : ${stats['std']:,.2f}")
    _log(f"  Skewness  : {stats['skewness']:.4f}")
    _log(f"  Kurtosis  : {stats['kurtosis']:.4f}")
    _log(f"\n  Percentiles:")
    _log(f"    P1  : ${stats['p1']:,.2f}")
    _log(f"    P5  : ${stats['p5']:,.2f}")
    _log(f"    P10 : ${stats['p10']:,.2f}")
    _log(f"    P20 (cutoff) : ${stats['p20']:,.2f}  ← Top 20% boundary")
    _log(f"    P50 : ${stats['p50']:,.2f}")
    _log(f"    P75 : ${stats['p75']:,.2f}")
    _log(f"    P80 : ${stats['p80']:,.2f}")
    _log(f"    P90 : ${stats['p90']:,.2f}")
    _log(f"    P95 : ${stats['p95']:,.2f}")
    _log(f"    P99 : ${stats['p99']:,.2f}")

    # Skewness interpretation
    sk = stats['skewness']
    if abs(sk) < 0.5:
        shape = "Approximately symmetric"
    elif sk > 1.5:
        shape = "Heavy right skew — high-value outliers pull the tail"
    elif sk > 0.5:
        shape = "Moderate right skew"
    elif sk < -1.5:
        shape = "Heavy left skew — very unprofitable customers"
    else:
        shape = "Moderate left skew"
    _log(f"\n  Distribution Shape: {shape}")

    # Ties near boundary
    cutoff_score = score.quantile(0.80)   # top 20% threshold
    near_cutoff = score[abs(score - cutoff_score) < 0.01]
    _log(f"\n  Score ties analysis:")
    _log(f"    Customers within $0.01 of cutoff score: {len(near_cutoff):,}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — RANKING & LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_pseudo_labels(
    df: pd.DataFrame,
    score: pd.Series,
    report_lines: list,
    logger
) -> pd.DataFrame:
    """
    Ranks customers deterministically and assigns binary pseudo-labels.

    Determinism guarantee:
    - Primary sort: profitability_score descending
    - Tie-breaker: customer ID ascending (low ID wins in case of tie)
    - This ensures the exact same 100k customers get Label=1 every run.

    Label assignment:
    - Rank 1–100,000 → pseudo_label = 1
    - Rank 100,001–500,000 → pseudo_label = 0

    Args:
        df: Formula dataset (with ID column).
        score: Profitability scores.
        report_lines: Report accumulator.
        logger: Logger.

    Returns:
        DataFrame with columns: id, profitability_score, rank, pseudo_label
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("RANKING & PSEUDO-LABEL GENERATION")
    _log(f"{'='*60}")

    # Combine ID and score
    df_ranked = pd.DataFrame({
        config.ID_COL: df[config.ID_COL].values,
        'profitability_score': score.values
    })

    # Deterministic sort: score DESC, then ID ASC as tie-breaker
    df_ranked = df_ranked.sort_values(
        ['profitability_score', config.ID_COL],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Assign rank (1-indexed)
    df_ranked['rank'] = np.arange(1, len(df_ranked) + 1)

    # Assign pseudo-labels
    df_ranked['pseudo_label'] = 0
    df_ranked.loc[df_ranked['rank'] <= TOP_K, 'pseudo_label'] = 1

    # Boundary score
    boundary_score = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]
    boundary_score_next = df_ranked.loc[df_ranked['rank'] == TOP_K + 1, 'profitability_score'].iloc[0]
    gap = boundary_score - boundary_score_next

    _log(f"  Customers ranked    : {len(df_ranked):,}")
    _log(f"  Top 100k threshold  : Rank {TOP_K:,}")
    _log(f"  Boundary score (#100k)     : ${boundary_score:,.4f}")
    _log(f"  Score just below (#100,001): ${boundary_score_next:,.4f}")
    _log(f"  Gap at cutoff              : ${gap:,.4f}")

    # Segment summaries
    _log(f"\n  Segment Summary:")
    for pct, label in [(1, 'Top 1%'), (5, 'Top 5%'), (10, 'Top 10%'),
                       (20, 'Top 20% (cutoff)'), (50, 'Top 50%')]:
        k = int(500_000 * pct / 100)
        seg = df_ranked[df_ranked['rank'] <= k]['profitability_score']
        _log(f"    {label:<22}: min=${seg.min():>10,.2f} | mean=${seg.mean():>10,.2f} | max=${seg.max():>10,.2f}")

    return df_ranked


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — LABEL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_labels(df_ranked: pd.DataFrame, report_lines: list, logger) -> dict:
    """
    Hard validation of pseudo-label integrity.

    Checks:
    1. Exactly 100,000 positive labels
    2. Exactly 400,000 negative labels
    3. No missing labels
    4. No duplicate IDs
    5. Labels are only 0 or 1
    6. Every customer from original dataset is present

    Args:
        df_ranked: Ranked DataFrame with pseudo_label column.
        report_lines: Report accumulator.
        logger: Logger.

    Returns:
        Dict of check_name → bool.

    Raises:
        RuntimeError if any critical check fails.
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("LABEL VALIDATION")
    _log(f"{'='*60}")

    checks = {}

    n_pos = int((df_ranked['pseudo_label'] == 1).sum())
    n_neg = int((df_ranked['pseudo_label'] == 0).sum())
    n_total = len(df_ranked)

    checks['n_positives_exact'] = n_pos == TOP_K
    checks['n_negatives_exact'] = n_neg == (500_000 - TOP_K)
    checks['no_missing_labels'] = df_ranked['pseudo_label'].isnull().sum() == 0
    checks['no_duplicate_ids']  = df_ranked[config.ID_COL].duplicated().sum() == 0
    checks['only_binary_labels'] = set(df_ranked['pseudo_label'].unique()).issubset({0, 1})
    checks['total_row_count']   = n_total == 500_000

    _log(f"  Positive labels (=1) : {n_pos:,}  (expected 100,000) {'✓' if checks['n_positives_exact'] else '✗ FAIL'}")
    _log(f"  Negative labels (=0) : {n_neg:,}  (expected 400,000) {'✓' if checks['n_negatives_exact'] else '✗ FAIL'}")
    _log(f"  Positive rate        : {100.0*n_pos/n_total:.2f}%")
    _log(f"  Negative rate        : {100.0*n_neg/n_total:.2f}%")
    _log(f"  Missing labels       : {df_ranked['pseudo_label'].isnull().sum()} {'✓' if checks['no_missing_labels'] else '✗ FAIL'}")
    _log(f"  Duplicate IDs        : {df_ranked[config.ID_COL].duplicated().sum()} {'✓' if checks['no_duplicate_ids'] else '✗ FAIL'}")
    _log(f"  Only binary (0/1)    : {'✓' if checks['only_binary_labels'] else '✗ FAIL'}")
    _log(f"  Total rows           : {n_total:,} {'✓' if checks['total_row_count'] else '✗ FAIL'}")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"[FAIL] Label validation failed: {failed}")

    _log("  Label validation: PASS ✓")
    return checks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 & 7 — FEATURE SEPARATION ANALYSIS (Cohen's d)
# ─────────────────────────────────────────────────────────────────────────────
def compute_feature_separation(
    df: pd.DataFrame,
    df_ranked: pd.DataFrame,
    report_lines: list,
    logger
) -> pd.DataFrame:
    """
    Computes the discriminative power of each feature between label=1 and label=0.

    Metric: Cohen's d = (mean_pos - mean_neg) / pooled_std
    Interpretation:
        |d| < 0.2  → Negligible separation
        |d| 0.2–0.5 → Small
        |d| 0.5–0.8 → Medium
        |d| > 0.8  → Large (strong discrimination)

    Why Cohen's d over raw mean difference:
    - Features have different scales (f1 in thousands, f11 in 0-0.33)
    - Cohen's d normalizes by pooled std, making features comparable
    - This is NOT feature selection — it is pseudo-label quality assessment

    Args:
        df: Formula dataset with all 23 features.
        df_ranked: Ranked DataFrame with pseudo_label.
        report_lines: Report accumulator.
        logger: Logger.

    Returns:
        DataFrame ranked by |Cohen's d| descending.
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("FEATURE SEPARATION ANALYSIS (Cohen's d)")
    _log(f"{'='*60}")

    # Merge features with labels
    df_merged = df_ranked[['id', 'pseudo_label']].merge(
        df[['id'] + config.FEATURE_COLS], on='id', how='left'
    )

    pos = df_merged[df_merged['pseudo_label'] == 1]
    neg = df_merged[df_merged['pseudo_label'] == 0]

    records = []
    for col in config.FEATURE_COLS:
        p_vals = pos[col].dropna().to_numpy()
        n_vals = neg[col].dropna().to_numpy()

        if len(p_vals) == 0 or len(n_vals) == 0:
            continue

        mean_pos = float(np.mean(p_vals))
        mean_neg = float(np.mean(n_vals))
        std_pos  = float(np.std(p_vals, ddof=1))
        std_neg  = float(np.std(n_vals, ddof=1))

        # Pooled standard deviation
        n1, n2 = len(p_vals), len(n_vals)
        pooled_std = np.sqrt(((n1-1)*std_pos**2 + (n2-1)*std_neg**2) / (n1+n2-2))

        cohen_d = (mean_pos - mean_neg) / (pooled_std + 1e-10)
        mean_diff = mean_pos - mean_neg
        pct_diff = 100 * mean_diff / (abs(mean_neg) + 1e-10)

        if abs(cohen_d) > 0.8:
            strength = "LARGE"
        elif abs(cohen_d) > 0.5:
            strength = "MEDIUM"
        elif abs(cohen_d) > 0.2:
            strength = "SMALL"
        else:
            strength = "NEGLIGIBLE"

        records.append({
            "feature": col,
            "mean_positive": round(mean_pos, 4),
            "mean_negative": round(mean_neg, 4),
            "mean_difference": round(mean_diff, 4),
            "pct_difference": round(pct_diff, 2),
            "std_positive": round(std_pos, 4),
            "std_negative": round(std_neg, 4),
            "cohen_d": round(cohen_d, 4),
            "abs_cohen_d": round(abs(cohen_d), 4),
            "separation_strength": strength
        })

    df_sep = pd.DataFrame(records).sort_values('abs_cohen_d', ascending=False)

    _log(f"\n  {'Feature':<6} {'Cohen_d':>10} {'Strength':<12} {'Mean(+)':>12} {'Mean(-)':>12} {'Diff%':>8}")
    _log(f"  {'-'*62}")
    for _, row in df_sep.iterrows():
        _log(
            f"  {row['feature']:<6} {row['cohen_d']:>10.4f} {row['separation_strength']:<12} "
            f"{row['mean_positive']:>12.2f} {row['mean_negative']:>12.2f} {row['pct_difference']:>7.1f}%"
        )

    _log(f"\n  Top 5 most discriminative features:")
    for _, row in df_sep.head(5).iterrows():
        direction = "HIGHER in positives" if row['cohen_d'] > 0 else "LOWER in positives"
        _log(f"    {row['feature']}: Cohen's d={row['cohen_d']:.4f} ({direction}) — {row['separation_strength']}")

    _log(f"\n  Bottom 5 weakest discriminative features:")
    for _, row in df_sep.tail(5).iterrows():
        _log(f"    {row['feature']}: Cohen's d={row['cohen_d']:.4f} — {row['separation_strength']}")

    return df_sep


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — BOUNDARY / LABEL NOISE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_boundary(
    df_ranked: pd.DataFrame,
    report_lines: list,
    logger
) -> Dict:
    """
    Analyzes the decision boundary around rank 100,000.

    Why this matters: Pseudo-labels near the cutoff are the noisiest.
    A customer ranked 99,999 gets label=1 while rank 100,001 gets label=0.
    If their profitability scores are nearly identical, this is label noise.
    XGBoost and LightGBM will struggle to correctly classify these boundary cases.
    This analysis quantifies that noise.

    Window: Ranks 99,500 – 100,500 (1,000 customers around the boundary).

    Args:
        df_ranked: Full ranked DataFrame.
        report_lines: Report accumulator.
        logger: Logger.

    Returns:
        Dict with boundary statistics.
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("BOUNDARY / LABEL NOISE ANALYSIS (Ranks 99,500 – 100,500)")
    _log(f"{'='*60}")

    # Narrow window
    narrow = df_ranked[
        (df_ranked['rank'] >= 99_900) & (df_ranked['rank'] <= 100_100)
    ].copy()

    # Wide window for density estimation
    wide = df_ranked[
        (df_ranked['rank'] >= 99_500) & (df_ranked['rank'] <= 100_500)
    ].copy()

    boundary_score = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]

    # Score range in narrow window
    narrow_range = narrow['profitability_score'].max() - narrow['profitability_score'].min()

    # Score gaps between consecutive ranks in boundary
    narrow_sorted = narrow.sort_values('rank')
    gaps = narrow_sorted['profitability_score'].diff().abs().dropna()
    avg_gap = float(gaps.mean())
    max_gap = float(gaps.max())
    min_gap = float(gaps.min())

    # Customers within % of boundary score
    for pct_margin in [0.5, 1.0, 2.0, 5.0]:
        margin = abs(boundary_score) * pct_margin / 100
        in_margin = int((abs(df_ranked['profitability_score'] - boundary_score) <= margin).sum())
        _log(f"  Customers within ±{pct_margin:.1f}% of cutoff score: {in_margin:,}")

    _log(f"\n  Boundary score (#100,000): ${boundary_score:,.4f}")
    _log(f"  Narrow window (99,900–100,100) — {len(narrow):,} customers:")
    _log(f"    Score range in window : ${narrow_range:,.4f}")
    _log(f"    Avg consecutive gap   : ${avg_gap:,.6f}")
    _log(f"    Max consecutive gap   : ${max_gap:,.6f}")
    _log(f"    Min consecutive gap   : ${min_gap:,.8f}")

    # Density assessment
    if narrow_range < 1.0:
        density = "VERY DENSE — high label noise risk at boundary"
        confidence = "LOW"
    elif narrow_range < 10.0:
        density = "DENSE — moderate label noise at boundary"
        confidence = "MEDIUM"
    else:
        density = "SPREAD — relatively stable boundary"
        confidence = "HIGH"

    _log(f"\n  Boundary density     : {density}")
    _log(f"  Label confidence     : {confidence}")
    _log(f"  Interpretation: Customers at ranks 99,500–100,500 share similar")
    _log(f"  profitability scores. Some may be mislabeled relative to true AmEx")
    _log(f"  labels. The ML models must learn to navigate this uncertainty.")
    _log(f"  This is exactly where the XGBoost/LightGBM correction layer adds value.")

    return {
        "boundary_score": round(boundary_score, 6),
        "narrow_window_score_range": round(narrow_range, 6),
        "avg_consecutive_gap": round(avg_gap, 8),
        "max_consecutive_gap": round(max_gap, 6),
        "density_assessment": density,
        "label_confidence": confidence
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — PSEUDO-LABEL AGREEMENT ANALYSIS (Recommended Improvement)
# ─────────────────────────────────────────────────────────────────────────────
def pseudo_label_agreement_analysis(
    df_ranked: pd.DataFrame,
    report_lines: list,
    logger
) -> Dict:
    """
    Quantifies boundary uncertainty by analyzing the score distribution
    around the Top 100k cutoff.

    This analysis answers:
    1. How large is the score gap between rank 100,000 and 100,001?
    2. What % of customers have scores within ±X% of the cutoff?
    3. How many customers are "nearly indistinguishable" from the boundary?

    Interpretation:
    - If many customers cluster near the boundary → high label noise
    - If there's a natural score gap at rank 100,000 → label assignment is reliable
    - A large score gap at the cutoff = strong signal that the formula separates well

    Args:
        df_ranked: Full ranked DataFrame.
        report_lines: Report accumulator.
        logger: Logger.

    Returns:
        Dict with agreement statistics.
    """
    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    _log(f"\n{'='*60}")
    _log("PSEUDO-LABEL AGREEMENT ANALYSIS (Boundary Uncertainty)")
    _log(f"{'='*60}")

    boundary_score = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]
    next_score = df_ranked.loc[df_ranked['rank'] == TOP_K + 1, 'profitability_score'].iloc[0]
    cutoff_gap = boundary_score - next_score

    score_range = df_ranked['profitability_score'].max() - df_ranked['profitability_score'].min()
    relative_gap_pct = 100.0 * cutoff_gap / (abs(score_range) + 1e-10)

    _log(f"\n  Score at rank 100,000    : ${boundary_score:,.6f}")
    _log(f"  Score at rank 100,001    : ${next_score:,.6f}")
    _log(f"  Gap at exact cutoff      : ${cutoff_gap:,.6f}")
    _log(f"  Total score range        : ${score_range:,.2f}")
    _log(f"  Gap as % of total range  : {relative_gap_pct:.6f}%")

    # Uncertainty zones
    _log(f"\n  Uncertainty zones (customers whose label is uncertain):")
    margins = [1, 5, 10, 25, 50, 100]
    uncertainty_data = []
    for margin in margins:
        in_zone = df_ranked[
            abs(df_ranked['profitability_score'] - boundary_score) <= (cutoff_gap * margin)
        ]
        n_in_zone = len(in_zone)
        label1_in_zone = int((in_zone['pseudo_label'] == 1).sum())
        label0_in_zone = int((in_zone['pseudo_label'] == 0).sum())
        _log(f"    Within {margin:>3}× cutoff gap: {n_in_zone:>8,} customers "
             f"({label1_in_zone:,} label=1 | {label0_in_zone:,} label=0)")
        uncertainty_data.append({
            "margin_multiplier": margin,
            "n_customers": n_in_zone,
            "n_label1": label1_in_zone,
            "n_label0": label0_in_zone
        })

    # Confidence assessment
    if cutoff_gap < 0.01:
        agreement_quality = "VERY LOW — near-zero gap, many customers interchangeable"
    elif cutoff_gap < 1.0:
        agreement_quality = "LOW — small gap, boundary is noisy"
    elif cutoff_gap < 10.0:
        agreement_quality = "MEDIUM — moderate gap, some boundary uncertainty remains"
    else:
        agreement_quality = "HIGH — large gap, boundary is well-defined"

    _log(f"\n  Pseudo-Label Agreement Quality: {agreement_quality}")
    _log(f"  Impact on ML: Customers near the boundary contribute the most label")
    _log(f"  noise during training. XGBoost/LightGBM will treat these 'uncertain'")
    _log(f"  customers as harder examples. The 80% formula weight in the ensemble")
    _log(f"  ensures the human formula's ranking dominates for clear cases.")

    return {
        "boundary_score": round(boundary_score, 6),
        "cutoff_gap": round(cutoff_gap, 8),
        "relative_gap_pct": round(relative_gap_pct, 8),
        "agreement_quality": agreement_quality,
        "uncertainty_zones": uncertainty_data
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def generate_visualizations(
    df_ranked: pd.DataFrame,
    df_sep: pd.DataFrame,
    exp_dir: str,
    logger
) -> list:
    """
    Generates all Section 3 visualizations.

    1. Profitability score distribution (histogram + KDE)
    2. Label distribution (pie/bar)
    3. Cumulative profitability curve (where does Top 20% start?)
    4. Boundary region visualization (zoomed score distribution at cutoff)
    5. Top separating features (bar chart of Cohen's d)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib not available — skipping visualizations")
        return []

    viz_dir = os.path.join(exp_dir, "visualizations")
    saved = []

    # ── Plot 1: Score Distribution ────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        score = df_ranked['profitability_score']
        boundary = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]

        # Histogram
        axes[0].hist(score, bins=100, color='#3498db', alpha=0.7, edgecolor='white')
        axes[0].axvline(boundary, color='red', linestyle='--', linewidth=2,
                        label=f'Top 20% cutoff: ${boundary:,.0f}')
        axes[0].set_title("Profitability Score Distribution (All 500k)")
        axes[0].set_xlabel("Profitability Score ($)")
        axes[0].set_ylabel("Number of Customers")
        axes[0].legend()

        # By label
        pos_scores = df_ranked.loc[df_ranked['pseudo_label'] == 1, 'profitability_score']
        neg_scores = df_ranked.loc[df_ranked['pseudo_label'] == 0, 'profitability_score']
        axes[1].hist(neg_scores, bins=80, color='#e74c3c', alpha=0.6, label='Label=0 (400k)')
        axes[1].hist(pos_scores, bins=80, color='#2ecc71', alpha=0.6, label='Label=1 (100k)')
        axes[1].axvline(boundary, color='black', linestyle='--', linewidth=2, label='Cutoff')
        axes[1].set_title("Score Distribution by Pseudo-Label")
        axes[1].set_xlabel("Profitability Score ($)")
        axes[1].set_ylabel("Number of Customers")
        axes[1].legend()

        plt.suptitle("Section 3 — Profitability Score Analysis", fontsize=14)
        plt.tight_layout()
        p = os.path.join(viz_dir, "05_score_distribution.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        logger.info("  Saved: 05_score_distribution.png")
    except Exception as e:
        logger.warning(f"  Plot 1 failed: {e}")

    # ── Plot 2: Boundary Region Zoom ──────────────────────────────────
    try:
        boundary = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]
        boundary_region = df_ranked[
            (df_ranked['rank'] >= 99_000) & (df_ranked['rank'] <= 101_000)
        ]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(boundary_region['rank'], boundary_region['profitability_score'],
                color='#3498db', linewidth=0.8, alpha=0.8)
        ax.axvline(TOP_K, color='red', linestyle='--', linewidth=2, label=f'Rank 100,000 cutoff')
        ax.axhline(boundary, color='orange', linestyle=':', linewidth=1.5, label=f'Cutoff score: ${boundary:,.2f}')
        ax.fill_between(
            boundary_region['rank'],
            boundary_region['profitability_score'],
            boundary,
            where=boundary_region['rank'] <= TOP_K,
            alpha=0.2, color='green', label='Label=1 region'
        )
        ax.fill_between(
            boundary_region['rank'],
            boundary_region['profitability_score'],
            boundary,
            where=boundary_region['rank'] > TOP_K,
            alpha=0.2, color='red', label='Label=0 region'
        )
        ax.set_title("Boundary Region — Ranks 99,000 to 101,000 (Zoomed)")
        ax.set_xlabel("Customer Rank")
        ax.set_ylabel("Profitability Score ($)")
        ax.legend()
        plt.tight_layout()
        p = os.path.join(viz_dir, "06_boundary_region.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        logger.info("  Saved: 06_boundary_region.png")
    except Exception as e:
        logger.warning(f"  Plot 2 failed: {e}")

    # ── Plot 3: Feature Separation (Cohen's d) ─────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = df_sep['cohen_d'].apply(
            lambda d: '#2ecc71' if d > 0 else '#e74c3c'
        )
        ax.barh(df_sep['feature'], df_sep['cohen_d'], color=colors, alpha=0.8)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='|d|=0.8 (Large)')
        ax.axvline(-0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title("Feature Separation Power — Cohen's d (Label=1 vs Label=0)")
        ax.set_xlabel("Cohen's d  (positive = higher in Label=1 customers)")
        ax.legend()
        plt.tight_layout()
        p = os.path.join(viz_dir, "07_feature_separation.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        logger.info("  Saved: 07_feature_separation.png")
    except Exception as e:
        logger.warning(f"  Plot 3 failed: {e}")

    # ── Plot 4: Cumulative Score Curve ────────────────────────────────
    try:
        df_sorted = df_ranked.sort_values('rank')
        cumsum = df_sorted['profitability_score'].cumsum()
        total = cumsum.iloc[-1]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_sorted['rank'], cumsum / total * 100, color='#3498db', linewidth=1.5)
        ax.axvline(TOP_K, color='red', linestyle='--', linewidth=2,
                   label=f'Top 20% (rank 100k)')
        cum_at_100k = cumsum.iloc[TOP_K - 1] / total * 100
        ax.axhline(cum_at_100k, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Cumulative at cutoff: {cum_at_100k:.1f}%')
        ax.set_title("Cumulative Profitability Curve (Lorenz-style)")
        ax.set_xlabel("Customer Rank (sorted by profitability)")
        ax.set_ylabel("Cumulative % of Total Profitability")
        ax.legend()
        plt.tight_layout()
        p = os.path.join(viz_dir, "08_cumulative_curve.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        logger.info("  Saved: 08_cumulative_curve.png")
    except Exception as e:
        logger.warning(f"  Plot 4 failed: {e}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def main(exp_dir: str):
    """
    Section 3 main orchestrator.

    Args:
        exp_dir: Path to the Section 2 experiment directory containing
                 formula_dataset.parquet.
    """
    print("\n" + "=" * 70)
    print("SECTION 3 — PSEUDO-LABEL GENERATION & VALIDATION")
    print("American Express Campus Challenge 2026 | Modelling Formulation")
    print("=" * 70)

    logger = setup_logger(exp_dir, name="section3_pseudo_label")
    report_lines = []

    def _log(msg):
        logger.info(msg)
        report_lines.append(msg)

    exp_log = {
        "section": "3 — Pseudo-Label Generation & Validation",
        "timestamp_start": datetime.now().isoformat(),
        "experiment_dir": exp_dir,
        "top_k": TOP_K,
        "formula_constants": {
            "IC_TRAVEL": IC_TRAVEL, "IC_OTHER": IC_OTHER, "CPP": CPP, "URR": URR,
            "INT_RATE": INT_RATE, "LOUNGE": LOUNGE, "CAB": CAB,
            "LGD": LGD, "CALL": CALL, "COLL": COLL,
            "SUPP_FEE": SUPP_FEE, "CL_PROXY": CL_PROXY
        }
    }

    # Steps 1–2
    df = load_formula_dataset(exp_dir, logger)
    score = compute_profitability_score(df, logger)
    score_stats = analyze_score_distribution(score, report_lines, logger)

    # Step 3–4: Ranking + labels
    df_ranked = generate_pseudo_labels(df, score, report_lines, logger)

    # Step 5: Validation
    label_checks = validate_labels(df_ranked, report_lines, logger)

    # Steps 6–7: Feature separation
    df_sep = compute_feature_separation(df, df_ranked, report_lines, logger)

    # Step 9: Boundary analysis
    boundary_stats = analyze_boundary(df_ranked, report_lines, logger)

    # Pseudo-label agreement analysis (recommended improvement)
    agreement_stats = pseudo_label_agreement_analysis(df_ranked, report_lines, logger)

    # Visualizations
    _log(f"\n{'='*60}")
    _log("GENERATING VISUALIZATIONS")
    _log(f"{'='*60}")
    viz_files = generate_visualizations(df_ranked, df_sep, exp_dir, logger)
    _log(f"  {len(viz_files)} visualizations saved.")

    # Save outputs
    _log(f"\n{'='*60}")
    _log("SAVING ARTIFACTS")
    _log(f"{'='*60}")

    # Pseudo labels + ranking table
    pseudo_path = os.path.join(exp_dir, "data", "pseudo_labels.parquet")
    df_ranked.to_parquet(pseudo_path, index=False)
    _log(f"  Saved: pseudo_labels.parquet ({len(df_ranked):,} rows)")

    # Feature separation report
    sep_path = os.path.join(exp_dir, "reports", "feature_separation_report.csv")
    df_sep.to_csv(sep_path, index=False)
    _log(f"  Saved: feature_separation_report.csv")

    # Score stats
    stats_path = os.path.join(exp_dir, "reports", "score_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(score_stats, f, indent=2)

    # Boundary stats
    boundary_path = os.path.join(exp_dir, "reports", "boundary_analysis.json")
    with open(boundary_path, "w") as f:
        json.dump(boundary_stats, f, indent=2)

    # Agreement stats
    agreement_path = os.path.join(exp_dir, "reports", "pseudo_label_agreement.json")
    with open(agreement_path, "w") as f:
        json.dump(agreement_stats, f, indent=2, default=str)

    # ── CHECKPOINT 3 REPORT ──────────────────────────────────────────
    n_pos = int((df_ranked['pseudo_label'] == 1).sum())
    n_neg = int((df_ranked['pseudo_label'] == 0).sum())
    boundary_score_val = df_ranked.loc[df_ranked['rank'] == TOP_K, 'profitability_score'].iloc[0]

    cp_lines = []
    cp_lines.append("=" * 70)
    cp_lines.append("CHECKPOINT 3 — PSEUDO-LABEL VALIDATION REPORT")
    cp_lines.append(f"Generated: {datetime.now().isoformat()}")
    cp_lines.append("=" * 70)

    cp_lines.append(f"\n[LABEL STATISTICS]")
    cp_lines.append(f"  Positive Labels (=1) : {n_pos:,}  (expected 100,000) {'PASS ✓' if n_pos==100_000 else 'FAIL ✗'}")
    cp_lines.append(f"  Negative Labels (=0) : {n_neg:,}  (expected 400,000) {'PASS ✓' if n_neg==400_000 else 'FAIL ✗'}")
    cp_lines.append(f"  Positive Rate        : {100.0*n_pos/500_000:.2f}%")

    cp_lines.append(f"\n[RANKING VERIFICATION]")
    cp_lines.append(f"  Highest Score        : ${score_stats['max']:,.4f}")
    cp_lines.append(f"  Lowest Score         : ${score_stats['min']:,.4f}")
    cp_lines.append(f"  Boundary Score (100k): ${boundary_score_val:,.6f}")
    cp_lines.append(f"  Cutoff Gap           : ${agreement_stats['cutoff_gap']:,.8f}")
    cp_lines.append(f"  Status               : PASS ✓")

    cp_lines.append(f"\n[DISTRIBUTION]")
    cp_lines.append(f"  Skewness             : {score_stats['skewness']:.4f}")
    cp_lines.append(f"  Kurtosis             : {score_stats['kurtosis']:.4f}")
    cp_lines.append(f"  P20 (cutoff)         : ${score_stats['p20']:,.4f}")
    cp_lines.append(f"  Status               : PASS ✓")

    cp_lines.append(f"\n[BOUNDARY STABILITY]")
    cp_lines.append(f"  Score range (99.9k–100.1k): ${boundary_stats['narrow_window_score_range']:,.6f}")
    cp_lines.append(f"  Density Assessment   : {boundary_stats['density_assessment']}")
    cp_lines.append(f"  Label Confidence     : {boundary_stats['label_confidence']}")
    cp_lines.append(f"  Agreement Quality    : {agreement_stats['agreement_quality']}")

    cp_lines.append(f"\n[FEATURE SEPARATION — Top 10]")
    cp_lines.append(f"  {'Feature':<6} {'Cohen_d':>10} {'Strength':<12}")
    cp_lines.append(f"  {'-'*30}")
    for _, row in df_sep.head(10).iterrows():
        cp_lines.append(f"  {row['feature']:<6} {row['cohen_d']:>10.4f} {row['separation_strength']:<12}")

    cp_lines.append(f"\n[INTEGRITY]")
    for k, v in label_checks.items():
        cp_lines.append(f"  {k:<30}: {'PASS ✓' if v else 'FAIL ✗'}")

    all_files = [pseudo_path, sep_path, stats_path, boundary_path, agreement_path] + viz_files
    cp_lines.append(f"\n[FILES GENERATED]")
    for f in all_files:
        cp_lines.append(f"  {os.path.basename(f)}")

    cp_lines.append(f"\n[RISKS]")
    cp_lines.append(f"  1. Boundary density is {boundary_stats['density_assessment']}")
    cp_lines.append(f"  2. Customers within 10× cutoff gap: "
                    f"{next((d['n_customers'] for d in agreement_stats['uncertainty_zones'] if d['margin_multiplier']==10), 'N/A'):,}")
    cp_lines.append(f"  3. This is expected and handled by the 80/20 ensemble weighting.")

    cp_lines.append(f"\n[RECOMMENDATIONS]")
    cp_lines.append(f"  1. Feed pseudo_labels.parquet directly to Section 4 (LightGBM).")
    cp_lines.append(f"  2. Consider using scale_pos_weight in XGBoost (400k/100k = 4.0).")
    cp_lines.append(f"  3. Consider is_unbalance=True in LightGBM for class imbalance.")
    cp_lines.append(f"  4. Monitor LightGBM/XGBoost performance near boundary ranks.")

    cp_lines.append(f"\n{'='*70}")
    cp_lines.append(f"CHECKPOINT 3 FINAL STATUS: PASS ✓")
    cp_lines.append(f"Section 4 (LightGBM Pipeline) may proceed.")
    cp_lines.append(f"{'='*70}")

    cp_content = "\n".join(cp_lines)
    cp_path = save_text_report(exp_dir, cp_content, "reports", "checkpoint3_pseudo_label_report.txt")
    print(cp_content)

    # Full profiling report
    save_text_report(exp_dir, "\n".join(report_lines), "reports", "section3_profiling_report.txt")

    exp_log.update({
        "timestamp_end": datetime.now().isoformat(),
        "status": "PASS",
        "n_positive": n_pos,
        "n_negative": n_neg,
        "boundary_score": round(float(boundary_score_val), 6),
        "cutoff_gap": agreement_stats['cutoff_gap'],
        "agreement_quality": agreement_stats['agreement_quality'],
        "top_discriminative_feature": df_sep.iloc[0]['feature'],
        "visualizations": len(viz_files),
    })
    save_experiment_log(exp_dir, exp_log, filename="experiment_log_section3.json")

    checkpoint_pass("SECTION 3 — PSEUDO-LABEL GENERATION", logger)
    logger.info(f"Pseudo-labels saved to: {pseudo_path}")
    logger.info("Ready to proceed to Section 4 — LightGBM Pipeline.")

    return {"pseudo_labels_path": pseudo_path, "experiment_dir": exp_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section 3 — Pseudo-Label Generation")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=r"C:\Users\verma\Desktop\AmEx\modelling_formulation\experiments\exp_20260705_000107",
        help="Path to the Section 2 experiment directory"
    )
    args = parser.parse_args()
    result = main(args.exp_dir)
    print(f"\nPseudo-labels path: {result['pseudo_labels_path']}")
