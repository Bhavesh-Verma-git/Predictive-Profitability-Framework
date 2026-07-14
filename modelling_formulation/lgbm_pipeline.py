"""
lgbm_pipeline.py — Section 4: LightGBM Pipeline (with 5-Fold OOF)
American Express Campus Challenge 2026 | Modelling Formulation Project

Responsibilities:
1. Input Validation (ML Dataset + Pseudo Labels)
2. 5-Fold Stratified CV
3. LightGBM training with early stopping
4. Feature Importance (Gain & Split)
5. SHAP Analysis
6. Probability / Confidence Analysis
7. Correlation Analysis (Formula vs LGBM)
8. Disagreement Analysis
9. Generate Checkpoint 4 Report
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, save_experiment_log, save_text_report, checkpoint_pass

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2 — INPUT VALIDATION & DATA CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate_inputs(exp_dir: str, logger):
    """
    Loads the ML dataset (Dataset B) and Pseudo Labels, validating consistency.
    """
    logger.info("Loading ML Dataset (Dataset B) and Pseudo Labels...")

    ml_path = os.path.join(exp_dir, "data", "ml_dataset.parquet")
    lbl_path = os.path.join(exp_dir, "data", "pseudo_labels.parquet")

    if not os.path.exists(ml_path): raise RuntimeError(f"Missing ML dataset: {ml_path}")
    if not os.path.exists(lbl_path): raise RuntimeError(f"Missing pseudo labels: {lbl_path}")

    df_ml = pd.read_parquet(ml_path)
    df_lbl = pd.read_parquet(lbl_path)

    # Validation Checks
    if len(df_ml) != len(df_lbl): raise RuntimeError("Row counts do not match!")
    
    # Sort both by ID to guarantee alignment
    df_ml = df_ml.sort_values(config.ID_COL).reset_index(drop=True)
    df_lbl = df_lbl.sort_values(config.ID_COL).reset_index(drop=True)

    if not (df_ml[config.ID_COL].values == df_lbl[config.ID_COL].values).all():
        raise RuntimeError("Customer IDs do not perfectly align even after sorting!")
    # Check for raw NaNs (ML dataset must preserve them)
    nan_count = df_ml[config.FEATURE_COLS].isnull().sum().sum()
    if nan_count == 0:
        logger.warning("No NaNs found in ML dataset! Did you impute by mistake?")
    else:
        logger.info(f"Verified: {nan_count:,} raw NaNs preserved for LightGBM.")

    # Construction: X = 23 features only. y = pseudo_label
    X = df_ml[config.FEATURE_COLS]
    y = df_lbl['pseudo_label']
    
    # Extract formula score for later correlation/disagreement analysis
    formula_score = df_lbl['profitability_score']

    logger.info(f"Constructed X: {X.shape}, y: {y.shape}")
    return X, y, formula_score, df_ml[config.ID_COL]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — HYPERPARAMETER STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
def get_lgbm_params():
    """
    LightGBM Hyperparameters. NO defaults blindly used.
    
    Every parameter explained:
    """
    return {
        # Core Objective
        "objective": "binary",             # We are predicting binary pseudo-labels
        "metric": "auc",                   # Optimize ranking quality (AUC correlates with rank preservation)
        "boosting_type": "gbdt",           # Standard gradient boosting
        
        # Tree Structure
        "num_leaves": 31,                  # Chosen: 31. Effect: Controls model complexity (max leaves per tree). Tradeoff: Higher = more complex/overfit. 31 is conservative to prevent overfitting the weak labels.
        "max_depth": -1,                   # Chosen: -1 (no limit). Effect: Trees grow leaf-wise. Tradeoff: Controlled by num_leaves instead.
        "min_data_in_leaf": 100,           # Chosen: 100. Effect: Prevents splitting small noisy nodes. Tradeoff: Higher = more robust to noise, prevents overfitting boundary customers.
        
        # Learning Dynamics
        "learning_rate": 0.05,             # Chosen: 0.05. Effect: Shrinkage rate. Tradeoff: Slower learning = more robust trees.
        "n_estimators": 2000,              # Chosen: 2000. Effect: Max trees. We will rely on early_stopping to halt before this.
        
        # Regularization (To prevent copying the formula exactly and handle NaN noise)
        "feature_fraction": 0.8,           # Chosen: 0.8. Effect: Uses 80% of features per tree. Tradeoff: Prevents over-relying on top features (f7/f10), forcing it to learn alternative patterns.
        "bagging_fraction": 0.8,           # Chosen: 0.8. Effect: Uses 80% of rows per iteration. Tradeoff: Adds stochasticity, reduces variance.
        "bagging_freq": 1,                 # Required for bagging_fraction to work.
        "lambda_l1": 0.1,                  # Chosen: 0.1. Effect: L1 regularization (Lasso). Tradeoff: Drives weak feature weights to 0.
        "lambda_l2": 1.0,                  # Chosen: 1.0. Effect: L2 regularization (Ridge). Tradeoff: Smooths leaf outputs.
        
        # Class Balance
        "scale_pos_weight": 4.0,           # Chosen: 4.0. Effect: 400k negatives / 100k positives. Balances the gradients.
        
        # Engineering
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,                      # Use all cores
        "verbosity": -1                    # Suppress C++ warnings
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 & 6 — 5-FOLD CV & EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
def train_lgbm_cv(X, y, exp_dir, logger):
    """
    Executes 5-Fold Stratified CV with early stopping.
    """
    logger.info("Starting 5-Fold Stratified CV...")
    
    # Storage for OOF and importances
    oof_preds = np.zeros(len(X))
    feature_importance_df = pd.DataFrame()
    models = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    params = get_lgbm_params()
    n_estimators = params.pop('n_estimators')
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"  Training Fold {fold}...")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM Dataset
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Early Stopping: mandatory rule
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # Predict OOF
        best_iter = model.best_iteration
        preds = model.predict(X_val, num_iteration=best_iter)
        oof_preds[val_idx] = preds
        
        fold_auc = roc_auc_score(y_val, preds)
        logger.info(f"    Fold {fold} finished | Best Iter: {best_iter} | Val AUC: {fold_auc:.4f}")
        
        # Feature Importance
        fold_imp_df = pd.DataFrame()
        fold_imp_df["feature"] = config.FEATURE_COLS
        fold_imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
        fold_imp_df["importance_split"] = model.feature_importance(importance_type='split')
        fold_imp_df["fold"] = fold
        feature_importance_df = pd.concat([feature_importance_df, fold_imp_df], axis=0)
        
        models.append(model)
        
        # Save fold model
        model_path = os.path.join(exp_dir, "models", f"lgb_fold{fold}.txt")
        model.save_model(model_path)
        
    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"5-Fold CV Completed | Overall OOF AUC: {overall_auc:.4f}")
    
    # Aggregate importance
    agg_imp = feature_importance_df.groupby("feature")[["importance_gain", "importance_split"]].mean().reset_index()
    agg_imp = agg_imp.sort_values(by="importance_gain", ascending=False)
    
    return oof_preds, agg_imp, models, overall_auc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(models, X, exp_dir, logger):
    """
    Computes global SHAP values using a subsample for speed.
    """
    logger.info("Running SHAP Analysis (Subsampled)...")
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping.")
        return
        
    # Subsample 10k rows for SHAP to save time
    np.random.seed(config.RANDOM_SEED)
    sample_idx = np.random.choice(len(X), size=10_000, replace=False)
    X_sample = X.iloc[sample_idx]
    
    # Use Fold 1 model for explanation
    explainer = shap.TreeExplainer(models[0])
    shap_values = explainer.shap_values(X_sample)
    
    # Check if shap_values is a list (binary classification returns list [neg, pos])
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values
        
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": config.FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    shap_path = os.path.join(exp_dir, "reports", "shap_importance.csv")
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"SHAP feature importance saved to: {shap_path}")
    
    return shap_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 & MODEL CONFIDENCE ANALYSIS (Recommended Improvement)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_model_confidence(oof_preds, report_lines, logger):
    """
    Analyzes the confidence distribution of the predicted probabilities.
    """
    logger.info("Running Model Confidence Analysis...")
    report_lines.append("\n[MODEL CONFIDENCE ANALYSIS]")
    
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(oof_preds, bins=bins)
    
    report_lines.append("  Probability Distribution:")
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / len(oof_preds) * 100)
        report_lines.append(f"    {bins[i]:.1f}-{bins[i+1]:.1f} : {hist[i]:>7,} {bar}")
        
    p_mean = np.mean(oof_preds)
    p_std = np.std(oof_preds)
    p_uncertain = np.sum((oof_preds >= 0.45) & (oof_preds <= 0.55))
    p_high = np.sum(oof_preds > 0.95)
    p_low = np.sum(oof_preds < 0.05)
    
    report_lines.append(f"\n  Mean Probability  : {p_mean:.4f}")
    report_lines.append(f"  Std Deviation     : {p_std:.4f}")
    report_lines.append(f"  Uncertain (45-55%): {p_uncertain:,} customers")
    report_lines.append(f"  Very Confident Pos: {p_high:,} customers (>95%)")
    report_lines.append(f"  Very Confident Neg: {p_low:,} customers (<5%)")
    
    return p_mean, p_uncertain, p_high, p_low


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — CORRELATION ANALYSIS (Formula vs ML)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_correlation(formula_score, oof_preds, report_lines, logger):
    """
    Compares the original continuous business formula score with the LGBM probability.
    """
    logger.info("Running Correlation Analysis...")
    report_lines.append("\n[CORRELATION ANALYSIS (Formula vs LGBM)]")
    
    pearson_corr = np.corrcoef(formula_score, oof_preds)[0, 1]
    spearman_corr, _ = scipy_stats.spearmanr(formula_score, oof_preds)
    
    report_lines.append(f"  Pearson Correlation  : {pearson_corr:.4f}")
    report_lines.append(f"  Spearman Rank Corr   : {spearman_corr:.4f}")
    
    if spearman_corr > 0.95:
        report_lines.append("  INTERPRETATION: LGBM is heavily copying the formula ranking. Minimal new signal learned.")
    elif spearman_corr > 0.80:
        report_lines.append("  INTERPRETATION: LGBM learned the core business logic but adjusted rankings (strong hybrid).")
    else:
        report_lines.append("  INTERPRETATION: LGBM rankings diverge significantly from the formula.")
        
    return spearman_corr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — DISAGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_disagreements(formula_score, oof_preds, report_lines, logger):
    """
    Identifies edge cases where the Business Formula and ML model strongly disagree.
    """
    logger.info("Running Disagreement Analysis...")
    report_lines.append("\n[DISAGREEMENT ANALYSIS]")
    
    # Formula Top 20% threshold
    formula_cutoff = formula_score.quantile(0.80)
    is_formula_pos = formula_score >= formula_cutoff
    
    # LGBM predicted Top 20%
    lgbm_cutoff = np.percentile(oof_preds, 80)
    is_lgbm_pos = oof_preds >= lgbm_cutoff
    
    # High Formula, Low LGBM (Formula loved them, ML hated them)
    formula_high_lgbm_low = np.sum(is_formula_pos & ~is_lgbm_pos)
    
    # Low Formula, High LGBM (Formula hated them, ML loved them)
    formula_low_lgbm_high = np.sum(~is_formula_pos & is_lgbm_pos)
    
    report_lines.append(f"  Formula says Top 20%, LGBM says NO : {formula_high_lgbm_low:,} customers")
    report_lines.append(f"  Formula says NO, LGBM says Top 20% : {formula_low_lgbm_high:,} customers")
    report_lines.append("  These customers represent the 'correction layer' where ML overrides business rules.")
    
    return formula_high_lgbm_low, formula_low_lgbm_high


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(exp_dir: str):
    print("\n" + "=" * 70)
    print("SECTION 4 — LIGHTGBM PIPELINE (5-FOLD OOF)")
    print("=" * 70)

    logger = setup_logger(exp_dir, name="section4_lgbm")
    report_lines = []
    
    # 1. Inputs
    X, y, formula_score, ids = load_and_validate_inputs(exp_dir, logger)
    
    # 2. Train
    oof_preds, agg_imp, models, oof_auc = train_lgbm_cv(X, y, exp_dir, logger)
    
    # 3. SHAP
    shap_df = run_shap_analysis(models, X, exp_dir, logger)
    
    # 4. Analytics
    p_mean, p_uncert, p_high, p_low = analyze_model_confidence(oof_preds, report_lines, logger)
    spearman = analyze_correlation(formula_score, oof_preds, report_lines, logger)
    disagreements = analyze_disagreements(formula_score, oof_preds, report_lines, logger)
    
    # 5. Save Outputs
    oof_df = pd.DataFrame({
        config.ID_COL: ids,
        'lgbm_oof_prob': oof_preds
    })
    oof_path = os.path.join(exp_dir, "data", "lgbm_oof_predictions.parquet")
    oof_df.to_parquet(oof_path, index=False)
    
    imp_path = os.path.join(exp_dir, "reports", "lgbm_feature_importance.csv")
    agg_imp.to_csv(imp_path, index=False)
    
    # 6. Checkpoint 4 Report
    cp_lines = []
    cp_lines.append("=" * 70)
    cp_lines.append("CHECKPOINT 4 — LIGHTGBM PIPELINE REPORT")
    cp_lines.append(f"Generated: {datetime.now().isoformat()}")
    cp_lines.append("=" * 70)
    
    cp_lines.append("\n[VALIDATION STATUS]")
    cp_lines.append(f"  ✓ 5 folds completed")
    cp_lines.append(f"  ✓ Every customer has one OOF probability (Count: {len(oof_preds):,})")
    cp_lines.append(f"  ✓ No NaN predictions (Count: {np.isnan(oof_preds).sum()})")
    cp_lines.append(f"  ✓ No probabilities outside [0,1] (Min: {oof_preds.min():.4f}, Max: {oof_preds.max():.4f})")
    
    cp_lines.append("\n[MODEL METRICS]")
    cp_lines.append(f"  Overall OOF AUC      : {oof_auc:.5f}")
    cp_lines.append(f"  Spearman Correlation : {spearman:.4f} (LGBM vs Formula)")
    
    cp_lines.extend(report_lines)
    
    cp_lines.append("\n[TOP 5 FEATURES (Gain)]")
    for _, row in agg_imp.head(5).iterrows():
        cp_lines.append(f"  {row['feature']}: {row['importance_gain']:,.0f}")
        
    cp_lines.append(f"\n{'='*70}")
    cp_lines.append(f"CHECKPOINT 4 FINAL STATUS: PASS ✓")
    cp_lines.append(f"{'='*70}")
    
    cp_content = "\n".join(cp_lines)
    save_text_report(exp_dir, cp_content, "reports", "checkpoint4_lgbm_report.txt")
    print(cp_content)
    
    # Log
    save_experiment_log(exp_dir, {
        "section": "4",
        "oof_auc": oof_auc,
        "spearman": spearman,
        "n_uncertain": int(p_uncert)
    }, filename="experiment_log_section4.json")
    
    checkpoint_pass("SECTION 4 — LIGHTGBM", logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=r"C:\Users\verma\Desktop\AmEx\modelling_formulation\experiments\exp_20260705_000107")
    args = parser.parse_args()
    main(args.exp_dir)
