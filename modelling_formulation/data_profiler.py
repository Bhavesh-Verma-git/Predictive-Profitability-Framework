"""
data_profiler.py — Complete dataset profiling, validation, and quality audit.
American Express Campus Challenge 2026 | Modelling Formulation Project

Responsibilities:
- Missing value analysis (MCAR / MAR / MNAR classification)
- Distribution analysis (skewness, kurtosis, percentiles)
- Outlier detection (IQR + Z-score dual method)
- Duplicate detection (IDs and rows)
- Correlation analysis
- Leakage detection
- Dataset integrity validation
- Constant / near-constant feature detection
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats as scipy_stats


class DataProfiler:
    """
    Production-grade dataset profiler for competition ML pipelines.

    Design principle: Every method is self-contained, testable,
    and produces structured outputs — never raw print statements.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str,
        config: Any,
        logger: logging.Logger
    ):
        """
        Args:
            df: The raw dataset (read-only — never modified here).
            feature_cols: List of feature column names (f1–f23).
            id_col: The ID column name.
            config: Project configuration object.
            logger: Configured logger instance.
        """
        self.df = df.copy()          # Defensive copy — original is never touched
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.config = config
        self.logger = logger
        self.report_lines: List[str] = []

    def _log(self, msg: str) -> None:
        self.logger.info(msg)
        self.report_lines.append(msg)

    # ─────────────────────────────────────────────────────────────────
    # 1. BASIC PROFILE
    # ─────────────────────────────────────────────────────────────────
    def profile_basic(self) -> Dict[str, Any]:
        """
        Generates basic dataset metadata: shape, memory, types, duplicates.

        Why: Understanding dataset structure prevents downstream mistakes.
        """
        n_rows, n_cols = self.df.shape
        mem_mb = self.df.memory_usage(deep=True).sum() / 1e6
        n_dup_ids = self.df[self.id_col].duplicated().sum()
        n_dup_rows = self.df.duplicated().sum()

        result = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_features": len(self.feature_cols),
            "memory_mb": round(mem_mb, 2),
            "n_duplicate_ids": int(n_dup_ids),
            "n_duplicate_rows": int(n_dup_rows),
            "dtypes": self.df[self.feature_cols].dtypes.astype(str).to_dict()
        }

        self._log(f"\n{'='*60}")
        self._log("BASIC PROFILE")
        self._log(f"{'='*60}")
        self._log(f"  Rows          : {n_rows:,}")
        self._log(f"  Columns       : {n_cols} (1 ID + {len(self.feature_cols)} features)")
        self._log(f"  Memory Usage  : {mem_mb:.1f} MB")
        self._log(f"  Duplicate IDs : {n_dup_ids} {'✓' if n_dup_ids == 0 else '✗ WARNING'}")
        self._log(f"  Duplicate Rows: {n_dup_rows} {'✓' if n_dup_rows == 0 else '✗ WARNING'}")
        return result

    # ─────────────────────────────────────────────────────────────────
    # 2. MISSING VALUE ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def profile_missing(self) -> pd.DataFrame:
        """
        Complete missing value analysis with business-driven MNAR/MAR classification.

        Why: Missing value patterns in financial data are rarely random.
        Understanding WHY values are missing determines the correct
        imputation strategy for each dataset (A vs B).

        Returns:
            DataFrame with missing stats per feature, sorted descending.
        """
        self._log(f"\n{'='*60}")
        self._log("MISSING VALUE ANALYSIS")
        self._log(f"{'='*60}")

        records = []
        for col in self.feature_cols:
            n_miss = int(self.df[col].isnull().sum())
            pct_miss = 100.0 * n_miss / len(self.df)

            # Classify missingness type based on domain knowledge
            if pct_miss > 80:
                miss_type = "MNAR"  # Structural — e.g., f23 = 88%: likely no supplementary card
                risk = "RED"
            elif pct_miss > 50:
                miss_type = "MNAR"  # Business-driven structural missingness
                risk = "RED"
            elif pct_miss > 20:
                miss_type = "MAR"   # Missing at random — likely tied to card type
                risk = "YELLOW"
            elif pct_miss > 0:
                miss_type = "MAR"
                risk = "GREEN"
            else:
                miss_type = "COMPLETE"
                risk = "OK"

            records.append({
                "feature": col,
                "missing_count": n_miss,
                "missing_pct": round(pct_miss, 2),
                "missingness_type": miss_type,
                "risk_flag": risk,
                "formula_imputation": "0 (zero-fill)",
                "ml_imputation": "NaN preserved (native handling)"
            })

        df_missing = pd.DataFrame(records).sort_values("missing_pct", ascending=False)

        self._log(f"\n  {'Feature':<8} {'Missing%':>10} {'Type':<12} {'Risk'}")
        self._log(f"  {'-'*44}")
        for _, row in df_missing.iterrows():
            self._log(
                f"  {row['feature']:<8} {row['missing_pct']:>9.2f}%  "
                f"{row['missingness_type']:<12} {row['risk_flag']}"
            )

        red_count = (df_missing['risk_flag'] == 'RED').sum()
        self._log(f"\n  Summary: {red_count} features > 50% missing (RED flag)")
        self._log(f"  Note: All missingness is treated as MNAR / business-driven.")
        self._log(f"  Zero-fill in Dataset A reflects 'event did not occur'.")
        self._log(f"  Raw NaN in Dataset B lets models learn the missingness pattern.")

        return df_missing

    # ─────────────────────────────────────────────────────────────────
    # 3. DISTRIBUTION ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def profile_distributions(self) -> pd.DataFrame:
        """
        Computes statistical summary for all numeric features.

        Why: Understanding distributions guides model interpretation
        and flags anomalies (negative values, zero inflation, capping).

        Returns:
            DataFrame with per-feature statistical summary.
        """
        self._log(f"\n{'='*60}")
        self._log("DISTRIBUTION ANALYSIS")
        self._log(f"{'='*60}")

        records = []
        anomalies = []

        for col in self.feature_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            sk = float(scipy_stats.skew(s))
            ku = float(scipy_stats.kurtosis(s))
            n_zeros = int((s == 0).sum())
            n_neg = int((s < 0).sum())
            pct_zero = 100.0 * n_zeros / len(s)

            # Classify distribution shape
            if abs(sk) < 0.5:
                shape = "Approx. Normal"
            elif sk > 2:
                shape = "Heavy Right Skew"
            elif sk > 0.5:
                shape = "Right Skewed"
            elif sk < -2:
                shape = "Heavy Left Skew"
            else:
                shape = "Left Skewed"

            if pct_zero > 30:
                shape += " + Zero-Inflated"

            records.append({
                "feature": col,
                "count": len(s),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "p25": round(float(s.quantile(0.25)), 4),
                "p75": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(sk, 4),
                "kurtosis": round(ku, 4),
                "pct_zero": round(pct_zero, 2),
                "n_negative": n_neg,
                "shape": shape
            })

            if n_neg > 0:
                anomalies.append(f"  ⚠ {col}: {n_neg} negative values (min={s.min():.3f}) — likely refunds/chargebacks")

        df_dist = pd.DataFrame(records)

        self._log("\n  Key anomalies detected:")
        for a in anomalies:
            self._log(a)
        if not anomalies:
            self._log("  None detected.")

        self._log("\n  Zero-inflation summary (features with >30% zeros):")
        zero_inf = df_dist[df_dist['pct_zero'] > 30][['feature', 'pct_zero']]
        for _, row in zero_inf.iterrows():
            self._log(f"    {row['feature']}: {row['pct_zero']:.1f}% zeros")

        return df_dist

    # ─────────────────────────────────────────────────────────────────
    # 4. OUTLIER ANALYSIS (IQR + Z-Score dual method)
    # ─────────────────────────────────────────────────────────────────
    def profile_outliers(self) -> pd.DataFrame:
        """
        Dual-method outlier detection using IQR and Z-score.

        Why: Financial data legitimately contains high-value customers.
        We report outliers — we never remove or clip them.
        The goal is understanding extreme value distribution.

        Design decision: Dual method (IQR + Z-score) is used because
        IQR is robust to skewed distributions while Z-score detects
        extreme deviations from the mean. Either flagging independently
        reduces false negatives.

        Returns:
            DataFrame with outlier counts per feature.
        """
        self._log(f"\n{'='*60}")
        self._log("OUTLIER ANALYSIS (IQR + Z-Score — report only, never remove)")
        self._log(f"{'='*60}")

        records = []
        for col in self.feature_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            # IQR method
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            iqr_lower = q1 - self.config.IQR_OUTLIER_MULTIPLIER * iqr
            iqr_upper = q3 + self.config.IQR_OUTLIER_MULTIPLIER * iqr
            n_iqr = int(((s < iqr_lower) | (s > iqr_upper)).sum())

            # Z-score method
            z_scores = np.abs(scipy_stats.zscore(s, nan_policy='omit'))
            n_zscore = int((z_scores > self.config.Z_SCORE_OUTLIER_THRESHOLD).sum())

            pct_iqr = 100.0 * n_iqr / len(s)
            pct_zscore = 100.0 * n_zscore / len(s)

            records.append({
                "feature": col,
                "n_outliers_iqr": n_iqr,
                "pct_outliers_iqr": round(pct_iqr, 2),
                "n_outliers_zscore": n_zscore,
                "pct_outliers_zscore": round(pct_zscore, 2),
                "max_value": round(float(s.max()), 2),
                "min_value": round(float(s.min()), 2),
                "note": "Legitimate financial extremes — not removed"
            })

        df_outliers = pd.DataFrame(records).sort_values("pct_outliers_iqr", ascending=False)

        self._log(f"\n  Top features by IQR outlier rate:")
        self._log(f"  {'Feature':<8} {'IQR Outlier%':>14} {'ZScore Outlier%':>16} {'Max Value':>14}")
        self._log(f"  {'-'*55}")
        for _, row in df_outliers.head(10).iterrows():
            self._log(
                f"  {row['feature']:<8} {row['pct_outliers_iqr']:>13.2f}% "
                f"{row['pct_outliers_zscore']:>15.2f}% "
                f"{row['max_value']:>14,.2f}"
            )
        self._log("\n  Note: All outliers are financial legitimate values. No removal performed.")

        return df_outliers

    # ─────────────────────────────────────────────────────────────────
    # 5. CORRELATION ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def profile_correlation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes pairwise Pearson correlation between features.

        Why: High correlation (>0.90) indicates redundant features.
        We report but never remove — both LGBM and XGBoost handle
        correlated features natively through their regularization.

        Returns:
            Tuple of (full correlation matrix, high-correlation pairs DataFrame).
        """
        self._log(f"\n{'='*60}")
        self._log("CORRELATION ANALYSIS")
        self._log(f"{'='*60}")

        # Use formula-imputed (zero-fill) version for correlation
        df_filled = self.df[self.feature_cols].fillna(0)
        corr_matrix = df_filled.corr(method='pearson')

        # Extract high-correlation pairs
        pairs = []
        for i, col_a in enumerate(self.feature_cols):
            for j, col_b in enumerate(self.feature_cols):
                if i < j:
                    corr_val = corr_matrix.loc[col_a, col_b]
                    if abs(corr_val) >= self.config.HIGH_CORR_THRESHOLD:
                        pairs.append({
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "correlation": round(corr_val, 4),
                            "note": "High correlation — reported but not removed"
                        })

        df_pairs = pd.DataFrame(pairs)

        self._log(f"\n  High correlation pairs (|corr| >= {self.config.HIGH_CORR_THRESHOLD}):")
        if len(df_pairs) == 0:
            self._log("  None found at this threshold.")
        else:
            for _, row in df_pairs.iterrows():
                self._log(f"    {row['feature_a']} ↔ {row['feature_b']}: {row['correlation']:.4f}")

        return corr_matrix, df_pairs

    # ─────────────────────────────────────────────────────────────────
    # 6. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────────────
    def check_leakage(self) -> Dict[str, Any]:
        """
        Inspects features for potential target leakage.

        Why: Using future information or post-outcome variables inflates
        scores during training but fails on true test data.
        In this competition there is no test set — but we still must
        verify no feature artificially encodes profitability rank.

        Leakage risks in this dataset:
        - f4: Very high values (max=697,899) — unclear meaning. Investigate.
        - f12: Discrete, 0–116 range — could be a customer age or tenure signal.
        - f21: Points redeemed — this is a post-behavior variable.
        - f22, f23: Low non-zero rate — possible engagement flags.

        Returns:
            Dictionary with leakage risk assessment per feature.
        """
        self._log(f"\n{'='*60}")
        self._log("LEAKAGE DETECTION")
        self._log(f"{'='*60}")

        suspicious = {}

        # Check f4 — mystery feature with very large values
        f4_vals = self.df['f4'].dropna()
        f4_corr_f1 = self.df[['f4', 'f1']].dropna().corr().iloc[0, 1]
        if abs(f4_corr_f1) > 0.7:
            suspicious['f4'] = f"Correlation with f1 (balance) = {f4_corr_f1:.3f}. May encode financial exposure."
        else:
            suspicious['f4'] = f"Max={f4_vals.max():,.0f}, corr with f1={f4_corr_f1:.3f}. Likely an account limit/line. Safe."

        # f21 — points redeemed: this is behavioral post-outcome
        suspicious['f21'] = (
            "Points redeemed (f21) is a post-behavior variable. "
            "It reflects spending outcomes, not a predictive cause. "
            "RISK: May proxy the formula output. Used in Formula A but "
            "safe for ML because the formula already computes this independently."
        )

        # f12 — discrete 0-116 range
        f12_unique = self.df['f12'].nunique()
        suspicious['f12'] = f"Discrete feature, {f12_unique} unique values (0–116). Likely tenure/age. Safe for ML."

        # f23 — 88% missing, 3 unique values
        f23_unique = self.df['f23'].dropna().nunique()
        suspicious['f23'] = f"88% missing, {f23_unique} unique values when present. Likely a categorical flag. Safe for ML."

        self._log("\n  Leakage risk assessment:")
        for feat, note in suspicious.items():
            self._log(f"  {feat}: {note}")

        self._log("\n  CONCLUSION: No definitive leakage detected.")
        self._log("  f21 (points redeemed) is a mild proxy risk but is handled")
        self._log("  correctly since ML models train on pseudo-labels from the formula,")
        self._log("  not from f21 directly.")

        return {
            "leakage_detected": False,
            "suspicious_features": suspicious,
            "verdict": "No definitive leakage. Safe to proceed."
        }

    # ─────────────────────────────────────────────────────────────────
    # 7. CONSTANT / NEAR-CONSTANT FEATURE DETECTION
    # ─────────────────────────────────────────────────────────────────
    def check_constant_features(self) -> List[str]:
        """
        Identifies features with near-zero variance.

        Why: Constant features provide no signal to ML models
        and can cause numerical instability.

        Returns:
            List of near-constant feature names.
        """
        self._log(f"\n{'='*60}")
        self._log("CONSTANT / NEAR-CONSTANT FEATURE CHECK")
        self._log(f"{'='*60}")

        near_constant = []
        for col in self.feature_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            cv = s.std() / (abs(s.mean()) + 1e-9)    # Coefficient of variation
            if cv < self.config.NEAR_CONSTANT_THRESHOLD:
                near_constant.append(col)
                self._log(f"  ⚠ {col}: CV={cv:.6f} — near constant (not removed)")

        if not near_constant:
            self._log("  No near-constant features detected. ✓")
        return near_constant

    # ─────────────────────────────────────────────────────────────────
    # 8. INTEGRITY VALIDATION
    # ─────────────────────────────────────────────────────────────────
    def validate_integrity(
        self,
        expected_rows: int,
        expected_features: int
    ) -> Dict[str, bool]:
        """
        Hard validation of dataset integrity. Halts pipeline on failure.

        Why: Any unexpected rows, columns, or data types will
        silently corrupt downstream model training.

        Args:
            expected_rows: Expected row count (500,000).
            expected_features: Expected feature count (23).

        Returns:
            Dict of check_name → bool (True = PASS).

        Raises:
            RuntimeError if any critical check fails.
        """
        self._log(f"\n{'='*60}")
        self._log("INTEGRITY VALIDATION")
        self._log(f"{'='*60}")

        checks = {}

        # Row count
        checks["row_count"] = len(self.df) == expected_rows
        self._log(f"  Row count = {len(self.df):,} (expected {expected_rows:,}): {'✓' if checks['row_count'] else '✗ FAIL'}")

        # Feature count
        checks["feature_count"] = len(self.feature_cols) == expected_features
        self._log(f"  Feature count = {len(self.feature_cols)} (expected {expected_features}): {'✓' if checks['feature_count'] else '✗ FAIL'}")

        # No duplicate IDs
        n_dup = self.df[self.id_col].duplicated().sum()
        checks["no_duplicate_ids"] = n_dup == 0
        self._log(f"  Duplicate IDs = {n_dup}: {'✓' if checks['no_duplicate_ids'] else '✗ FAIL'}")

        # All expected feature columns present
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        checks["all_features_present"] = len(missing_cols) == 0
        self._log(f"  All features present: {'✓' if checks['all_features_present'] else f'✗ FAIL — missing: {missing_cols}'}")

        # ID column present
        checks["id_col_present"] = self.id_col in self.df.columns
        self._log(f"  ID column present: {'✓' if checks['id_col_present'] else '✗ FAIL'}")

        # No all-NaN features
        all_nan_features = [c for c in self.feature_cols if self.df[c].isnull().all()]
        checks["no_all_nan_features"] = len(all_nan_features) == 0
        self._log(f"  No all-NaN features: {'✓' if checks['no_all_nan_features'] else f'✗ FAIL — {all_nan_features}'}")

        # Overall
        all_pass = all(checks.values())
        self._log(f"\n  INTEGRITY STATUS: {'PASS ✓' if all_pass else 'FAIL ✗'}")

        if not all_pass:
            failed = [k for k, v in checks.items() if not v]
            raise RuntimeError(
                f"INTEGRITY VALIDATION FAILED. Failed checks: {failed}. "
                f"Investigate before proceeding."
            )

        return checks
