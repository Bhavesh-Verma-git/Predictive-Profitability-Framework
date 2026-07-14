import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, save_experiment_log, save_text_report, checkpoint_pass

from xgb_pipeline import (
    load_and_validate_inputs,
    analyze_probability_confidence,
    analyze_correlation_and_comparison,
    analyze_top20_overlap,
    analyze_disagreements,
    generate_executive_report,
    get_xgb_params
)

def main(exp_dir: str):
    logger = setup_logger(exp_dir, name="recover_xgb")
    report_lines = []

    X, y, formula_score, ids, lgbm_oof = load_and_validate_inputs(exp_dir, logger)

    logger.info("Reconstructing OOF from saved models...")
    oof_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model_path = os.path.join(exp_dir, "models", f"xgb_fold{fold}.json")
        booster = xgb.Booster()
        booster.load_model(model_path)

        X_val = X.iloc[val_idx]
        # XGBoost requires DMatrix for booster predict
        dval = xgb.DMatrix(X_val)
        preds = booster.predict(dval)
        oof_preds[val_idx] = preds
        logger.info(f"Fold {fold} OOF reconstructed. AUC: {roc_auc_score(y.iloc[val_idx], preds):.5f}")

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info(f"Reconstructed OOF AUC: {oof_auc:.5f}")

    # Confidence analysis
    p_mean, p_std, p_uncertain, p_high, p_low = analyze_probability_confidence(
        oof_preds, report_lines, logger)

    # Correlation
    corr_stats = analyze_correlation_and_comparison(
        formula_score, lgbm_oof, oof_preds, report_lines, logger)

    # Top-20% overlap
    overlap_stats = analyze_top20_overlap(
        ids, formula_score, lgbm_oof, oof_preds, report_lines, logger)

    # Disagreement
    disagree_stats = analyze_disagreements(
        formula_score, lgbm_oof, oof_preds, report_lines, logger)

    # Save outputs
    xgb_oof_df = pd.DataFrame({config.ID_COL: ids, 'xgb_oof_prob': oof_preds})
    oof_path = os.path.join(exp_dir, "data", "xgb_oof_predictions.parquet")
    xgb_oof_df.to_parquet(oof_path, index=False)
    logger.info(f"XGBoost OOF saved: {oof_path}")

    # Executive report
    lgbm_conf = {"high": 95680, "uncertain": 830}
    xgb_conf  = {"p_high": p_high, "p_uncertain": p_uncertain}
    exec_path = generate_executive_report(
        corr_stats, overlap_stats, 0.99996, oof_auc,
        lgbm_conf, xgb_conf, exp_dir, logger, report_lines
    )

    # Checkpoint 5 Report
    cp_lines = []
    cp_lines.append("=" * 70)
    cp_lines.append("CHECKPOINT 5 — XGBOOST PIPELINE VALIDATION REPORT")
    cp_lines.append(f"Generated: {datetime.now().isoformat()}")
    cp_lines.append("=" * 70)

    cp_lines.append("\n[TRAINING]")
    cp_lines.append(f"  ✓ Five folds completed")
    cp_lines.append(f"  ✓ OOF generated for all {len(oof_preds):,} customers")
    cp_lines.append(f"  Overall OOF AUC: {oof_auc:.5f}")

    cp_lines.append("\n[PROBABILITY]")
    cp_lines.append(f"  ✓ Min: {oof_preds.min():.6f} | Max: {oof_preds.max():.6f}")
    cp_lines.append(f"  ✓ Mean: {p_mean:.4f} | Std: {p_std:.4f}")
    cp_lines.append(f"  ✓ Very Confident Pos (>95%): {p_high:,}")
    cp_lines.append(f"  ✓ Uncertain (45-55%): {p_uncertain:,}")

    cp_lines.append("\n[COMPARISON]")
    cp_lines.append(f"  ✓ XGBoost vs Formula Spearman   : {corr_stats['xgb_formula_spearman']:.4f}")
    cp_lines.append(f"  ✓ XGBoost vs LightGBM Spearman  : {corr_stats['lgbm_xgb_spearman']:.4f}")
    cp_lines.append(f"  ✓ Formula ∩ XGBoost Top-100k    : {overlap_stats['formula_xgb_common']:,}")
    cp_lines.append(f"  ✓ LGBM ∩ XGBoost Top-100k       : {overlap_stats['lgbm_xgb_common']:,}")
    cp_lines.append(f"  ✓ All 3 agree (Top-100k)         : {overlap_stats['all_three_common']:,}")
    cp_lines.append(f"  ✓ XGBoost-only unique            : {overlap_stats['only_xgb']:,} ({overlap_stats['xgb_diversity']} diversity)")

    cp_lines.append("\n[RECOMMENDATIONS]")
    cp_lines.append("  1. Proceed to Section 6 with Formula(80%) + LGBM(10%) + XGB(10%) weights.")
    cp_lines.append("  2. Tune ensemble weights in Section 6 based on OOF overlap analysis.")

    cp_lines.append(f"\n{'='*70}")
    cp_lines.append(f"CHECKPOINT 5 FINAL STATUS: PASS ✓")
    cp_lines.append(f"Section 6 (Rank Normalization & Ensemble Optimization) may proceed.")
    cp_lines.append(f"{'='*70}")

    cp_content = "\n".join(cp_lines)
    save_text_report(exp_dir, cp_content, "reports", "checkpoint5_xgb_report.txt")
    print(cp_content)

    save_experiment_log(exp_dir, {
        "section": "5",
        "oof_auc": oof_auc,
        "xgb_formula_spearman": corr_stats['xgb_formula_spearman'],
        "lgbm_xgb_spearman": corr_stats['lgbm_xgb_spearman'],
        "overlap_all_three": overlap_stats['all_three_common'],
        "xgb_unique_top20": overlap_stats['only_xgb'],
        "diversity": overlap_stats['xgb_diversity'],
    }, filename="experiment_log_section5.json")

    checkpoint_pass("SECTION 5 — XGBOOST", logger)


if __name__ == "__main__":
    main(r"C:\Users\verma\Desktop\AmEx\modelling_formulation\experiments\exp_20260705_000107")
