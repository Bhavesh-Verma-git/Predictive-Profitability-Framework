"""
=======================================================================
  American Express Campus Challenge 2026 - Round 1
  PROFITABILITY FRAMEWORK & SCORING ENGINE
=======================================================================

  PROFITABILITY EQUATION (Expert Heuristic Model)
  ─────────────────────────────────────────────────
  Score = R_net - C_net

  Where:
    R_net = Revenue streams (interchange, interest, annual fee proxy, credit line signal)
    C_net = Cost streams   (rewards redemption, benefit utilization, credit loss, retention)

  DETAILED FORMULA:
  ─────────────────
  Revenue (R):
    R1 = Annual Fee Revenue (flat baseline, always positive)
    R2 = Interchange Revenue = (f5 - f6 - f9) * INTERCHANGE_RATE_OTHER
       + f6 * INTERCHANGE_RATE_PREMIUM    ← Airlines (5x, higher merchant fee)
       + f9 * INTERCHANGE_RATE_PREMIUM    ← Lodging  (5x, higher merchant fee)
    R3 = Interest Revenue  = f1 * APR_PROXY
    R4 = Credit Line Signal = f17 * CREDIT_LINE_WEIGHT (high line → profitable/trusted CM)
    R5 = Supplementary Revenue = f19 * SUPP_ACCT_VALUE + f20 * CHARGE_CARD_VALUE

  Cost (C):
    C1 = Rewards Points Redeemed = f21 * POINT_VALUE
    C2 = Points Liability Accrual = pts_earned * POINT_VALUE * ACCRUAL_FRACTION
       where pts_earned = f6*5 + f9*5 + f7*1 + f8*1 + f10*1
    C3 = Lounge Cost = f13 * LOUNGE_UNIT_COST
    C4 = Airline Credit Cost = f14                      (direct dollar credit)
    C5 = Cab Benefit Cost = f15 * CAB_MONTH_COST        (per month of usage)
    C6 = Entertainment Credit Cost = f16                (direct dollar credit)
    C7 = Expected Credit Loss = f1 * f11 * ECL_MULTIPLIER
       + f3 * COLLECTION_CALL_PENALTY                   (collection calls signal severe risk)
    C8 = Retention Cost = f2 * RETENTION_CALL_COST      (each cancel call implies retention spend)

  FINAL SCORE:
    Profitability_Score = R1 + R2 + R3 + R4 + R5 - C1 - C2 - C3 - C4 - C5 - C6 - C7 - C8

  MISSING VALUE STRATEGY:
    - Spend subcategories (f6-f10): fill 0 (no spend in that category)
    - Risk score (f11): fill with 75th percentile (unknown risk is above average)
    - Count features (f13, f15, f2, f3): fill 0 (no usage/calls)
    - Credit features (f14, f16): fill 0 (no credit taken = no cost)
    - Lend lines (f17, f18): fill 0 (unknown)
    - Rewards (f4, f21): fill 0

=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
# Base directory is the folder where this script lives (works on any machine)
BASE          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE, '6a3eb196bc7a3_campus_challenge_r1_data.csv')
TEMPLATE_PATH = os.path.join(BASE, '6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx')
OUT_DIR       = os.path.join(BASE, 'output')
EDA_DIR       = os.path.join(BASE, 'eda_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS (Business-grounded assumptions)
# ══════════════════════════════════════════════════════════════════════

# --- Revenue Parameters ---
ANNUAL_FEE_REVENUE      = 625.0    # Mid-point of $500-$750 annual fee
INTERCHANGE_RATE_PREMIUM= 0.030    # 3% interchange on 5x categories (airlines, hotels)
INTERCHANGE_RATE_OTHER  = 0.020    # 2% interchange on 1x categories (other, entertainment, dining)
APR_PROXY               = 0.22     # ~22% APR on revolving balance (industry standard)
CREDIT_LINE_WEIGHT      = 0.0005   # Small positive signal from higher credit line
SUPP_ACCT_VALUE         = 200.0    # Each supplementary account adds ~$200 in annual spend contribution
CHARGE_CARD_VALUE       = 150.0    # Each active charge card adds ~$150 in revenue proxy

# --- Cost Parameters ---
POINT_VALUE_CENTS       = 0.015    # 1.5 cents per point (midpoint of 1-2 cents)
ACCRUAL_COST_RATE       = 0.40     # Assume 40% of earned points will be redeemed (liability cost)
LOUNGE_UNIT_COST        = 40.0     # $40 per lounge visit (midpoint of $30-$50)
CAB_MONTH_COST          = 15.0     # $15 per month of cab credit used (per problem brief)
ECL_MULTIPLIER          = 1.20     # ECL = Revolve_Balance * Risk_Score * 1.2x multiplier
COLLECTION_CALL_PENALTY = 300.0    # Each collection-related cancel call signals ~$300 expected loss
RETENTION_CALL_COST     = 75.0     # $75 cost per cancel call (retention offers)

print("=" * 70)
print("  AMEX Campus Challenge 2026 — Profitability Framework")
print("=" * 70)
print("\n  Model Parameters:")
print(f"    Annual Fee Revenue:     ${ANNUAL_FEE_REVENUE:,.2f}")
print(f"    Interchange (Premium):  {INTERCHANGE_RATE_PREMIUM*100:.1f}%  (Airlines, Lodging)")
print(f"    Interchange (Other):    {INTERCHANGE_RATE_OTHER*100:.1f}%  (Other, Entertainment, Dining)")
print(f"    APR on Revolving Bal:   {APR_PROXY*100:.1f}%")
print(f"    Point Value:            {POINT_VALUE_CENTS*100:.2f} cents/pt")
print(f"    Lounge Cost:            ${LOUNGE_UNIT_COST:.0f}/visit")
print(f"    Cab Credit Cost:        ${CAB_MONTH_COST:.0f}/month used")
print(f"    ECL Multiplier:         {ECL_MULTIPLIER}x (Balance × RiskScore)")
print(f"    Retention Call Cost:    ${RETENTION_CALL_COST:.0f}/call")
print(f"    Collection Call Pen:    ${COLLECTION_CALL_PENALTY:.0f}/call")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load Data
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 1: Loading Dataset")
print("─" * 70)

df = pd.read_csv(DATA_PATH)
print(f"  Records: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"  IDs are unique: {df['id'].nunique() == len(df)}")
assert df['id'].nunique() == len(df), "Duplicate IDs found!"

original_ids = df['id'].copy()

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Missing Value Imputation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 2: Missing Value Imputation")
print("─" * 70)

# Risk Score: unknown risk → fill with 75th percentile (conservative)
risk_75 = df['f11'].quantile(0.75)
print(f"  f11 (Risk Score) — filling nulls with 75th pct: {risk_75:.6f}")
df['f11'] = df['f11'].fillna(risk_75)

# Spend subcategories → 0 (CM didn't spend in that category)
for f in ['f6', 'f7', 'f8', 'f9', 'f10']:
    n_null = df[f].isnull().sum()
    if n_null > 0:
        print(f"  {f} → filling {n_null:,} nulls with 0")
    df[f] = df[f].fillna(0)

# All other features → 0
for f in ['f1', 'f2', 'f3', 'f4', 'f5', 'f12', 'f13', 'f14', 'f15',
          'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23']:
    n_null = df[f].isnull().sum()
    if n_null > 0:
        print(f"  {f} → filling {n_null:,} nulls with 0")
    df[f] = df[f].fillna(0)

print("  ✓ Imputation complete")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Compute Revenue Components
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 3: Computing Revenue Components")
print("─" * 70)

# R1: Annual Fee (flat for all active cardmembers)
df['R1_annual_fee'] = ANNUAL_FEE_REVENUE

# R2: Interchange Revenue
# KEY INSIGHT FROM EDA: f6-f10 subcategories are the actual card spend variables
# (mean ~$37K reconstructed total vs f5 mean ~$3.4K which is likely industry-level external spend)
# We use f6-f10 for interchange — these are the actual card transactions
# 5x premium categories: Airlines (f6), Lodging (f9) → ~3.0% effective merchant fee rate
# 1x standard categories: Other (f7), Entertainment (f8), Dining (f10) → ~2.0% rate
df['R2_interchange'] = (
    (df['f6'] + df['f9']) * INTERCHANGE_RATE_PREMIUM +
    (df['f7'] + df['f8'] + df['f10']) * INTERCHANGE_RATE_OTHER
)

# R3: Interest Revenue from Revolvers
# Revolve balance × annual interest rate
df['R3_interest'] = df['f1'] * APR_PROXY

# R4: Credit Line Quality Signal
# Higher lend line → lower default risk → slight revenue proxy
df['R4_credit_line'] = df['f17'] * CREDIT_LINE_WEIGHT

# R5: Supplementary & Multi-Card Revenue
df['R5_supplementary'] = (
    df['f19'] * SUPP_ACCT_VALUE +
    df['f20'] * CHARGE_CARD_VALUE
)

# Total Revenue
df['Total_Revenue'] = (
    df['R1_annual_fee'] +
    df['R2_interchange'] +
    df['R3_interest'] +
    df['R4_credit_line'] +
    df['R5_supplementary']
)

print("  Revenue components computed:")
for col in ['R1_annual_fee', 'R2_interchange', 'R3_interest', 'R4_credit_line', 'R5_supplementary', 'Total_Revenue']:
    print(f"    {col:25s}: mean={df[col].mean():>10.2f}, max={df[col].max():>14.2f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Compute Cost Components
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 4: Computing Cost Components")
print("─" * 70)

# C1: Rewards Points Redeemed (actual cash-equivalent outflow)
df['C1_pts_redeemed'] = df['f21'] * POINT_VALUE_CENTS

# C2: Points Liability from Spend (points earned × % likely to redeem)
# 5x: Airlines (f6), Lodging (f9) | 1x: Other (f7), Entertainment (f8), Dining (f10)
df['pts_earned'] = (
    df['f6'] * 5 + df['f9'] * 5 +
    df['f7'] * 1 + df['f8'] * 1 + df['f10'] * 1
)
df['C2_pts_liability'] = df['pts_earned'] * POINT_VALUE_CENTS * ACCRUAL_COST_RATE

# C3: Lounge Access Cost (per visit)
df['C3_lounge'] = df['f13'] * LOUNGE_UNIT_COST

# C4: Airline Credits Utilized (direct reimbursement cost)
df['C4_airline_credits'] = df['f14']

# C5: Cab Benefits Cost (monthly utility)
df['C5_cab_benefits'] = df['f15'] * CAB_MONTH_COST

# C6: Entertainment Credit Cost (direct amount)
df['C6_entertainment'] = df['f16']

# C7: Expected Credit Loss (ECL)
# ECL = Revolve Balance × Risk Score × Multiplier (Probability of Default × Exposure at Default)
# + Collection call penalty (severe risk signal)
df['C7_ecl'] = (
    df['f1'] * df['f11'] * ECL_MULTIPLIER +
    df['f3'] * COLLECTION_CALL_PENALTY
)

# C8: Retention Cost
# Cancel calls indicate risk of churn; Amex likely gave retention offers
df['C8_retention'] = df['f2'] * RETENTION_CALL_COST

# Total Cost
df['Total_Cost'] = (
    df['C1_pts_redeemed'] +
    df['C2_pts_liability'] +
    df['C3_lounge'] +
    df['C4_airline_credits'] +
    df['C5_cab_benefits'] +
    df['C6_entertainment'] +
    df['C7_ecl'] +
    df['C8_retention']
)

print("  Cost components computed:")
for col in ['C1_pts_redeemed', 'C2_pts_liability', 'C3_lounge', 'C4_airline_credits',
            'C5_cab_benefits', 'C6_entertainment', 'C7_ecl', 'C8_retention', 'Total_Cost']:
    print(f"    {col:25s}: mean={df[col].mean():>10.2f}, max={df[col].max():>14.2f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Compute Final Profitability Score
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 5: Computing Final Profitability Score")
print("─" * 70)

# FINAL PROFITABILITY EQUATION:
# Score = R1 + R2 + R3 + R4 + R5 - C1 - C2 - C3 - C4 - C5 - C6 - C7 - C8
df['Profitability_Score'] = df['Total_Revenue'] - df['Total_Cost']

print(f"\n  Profitability Score Statistics:")
print(f"    Min:    ${df['Profitability_Score'].min():,.2f}")
print(f"    P5:     ${df['Profitability_Score'].quantile(0.05):,.2f}")
print(f"    P25:    ${df['Profitability_Score'].quantile(0.25):,.2f}")
print(f"    Median: ${df['Profitability_Score'].median():,.2f}")
print(f"    Mean:   ${df['Profitability_Score'].mean():,.2f}")
print(f"    P75:    ${df['Profitability_Score'].quantile(0.75):,.2f}")
print(f"    P90:    ${df['Profitability_Score'].quantile(0.90):,.2f}")
print(f"    P95:    ${df['Profitability_Score'].quantile(0.95):,.2f}")
print(f"    Max:    ${df['Profitability_Score'].max():,.2f}")

n_positive = (df['Profitability_Score'] > 0).sum()
print(f"\n  Profitable CMs (score > 0): {n_positive:,} ({n_positive/len(df)*100:.1f}%)")
print(f"  Unprofitable CMs (score ≤ 0): {len(df)-n_positive:,} ({(len(df)-n_positive)/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Rank Order & Top 20% Identification
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 6: Ranking & Top 20% Selection")
print("─" * 70)

df = df.sort_values('Profitability_Score', ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1
df['is_top20pct'] = (df['Rank'] <= 100000).astype(int)

TOP20_THRESHOLD = df.loc[99999, 'Profitability_Score']
print(f"  Top 20% cutoff score: ${TOP20_THRESHOLD:,.2f}")
print(f"  Top 100K (Top 20%) Score Range: ${df.loc[99999,'Profitability_Score']:,.2f} to ${df.loc[0,'Profitability_Score']:,.2f}")

# Profile of Top 20% vs Bottom 80%
print("\n  Profile: Top 20% vs Bottom 80%")
profile_cols = ['f1', 'f5', 'f6', 'f9', 'f11', 'f13', 'f21', 'f17', 'f19', 'f20']
profile_names = ['Revolve Bal', 'Total Spend', 'Airlines Spend', 'Lodging Spend', 'Risk Score',
                 'Lounge Visits', 'Pts Redeemed', 'Lend Line', 'Supp Accts', 'Charge Cards']
top = df[df['is_top20pct'] == 1]
bot = df[df['is_top20pct'] == 0]
print(f"\n  {'Feature':<20} {'Top 20% Mean':>15} {'Bot 80% Mean':>15} {'Ratio':>8}")
print(f"  {'-'*60}")
for col, name in zip(profile_cols, profile_names):
    t_mean = top[col].mean()
    b_mean = bot[col].mean()
    ratio = t_mean / b_mean if b_mean != 0 else np.nan
    print(f"  {name:<20} {t_mean:>15,.2f} {b_mean:>15,.2f} {ratio:>8.2f}x")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: Visualization — Score Distribution & Component Attribution
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 7: Generating Framework Visualizations")
print("─" * 70)

revenue_cols = ['R1_annual_fee', 'R2_interchange', 'R3_interest', 'R4_credit_line', 'R5_supplementary']
cost_cols    = ['C1_pts_redeemed', 'C2_pts_liability', 'C3_lounge', 'C4_airline_credits',
                'C5_cab_benefits', 'C6_entertainment', 'C7_ecl', 'C8_retention']
rev_names    = ['Annual Fee', 'Interchange', 'Interest (Revolve)', 'Credit Line', 'Supplementary']
cost_names   = ['Pts Redeemed', 'Pts Liability', 'Lounge Cost', 'Airline Credit',
                'Cab Benefits', 'Entertainment Credit', 'Exp Credit Loss', 'Retention']

fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# 1. Score Distribution
ax1 = fig.add_subplot(gs[0, :2])
score_data = df['Profitability_Score']
ax1.hist(score_data.clip(score_data.quantile(0.01), score_data.quantile(0.99)),
         bins=200, color='#1565C0', alpha=0.7, edgecolor='none')
ax1.axvline(TOP20_THRESHOLD, color='#E53935', linewidth=2.5, linestyle='--',
            label=f'Top 20% Cutoff: ${TOP20_THRESHOLD:,.0f}')
ax1.axvline(0, color='#43A047', linewidth=1.5, linestyle=':', label='Break-even = $0')
ax1.set_xlabel('Profitability Score ($)', fontsize=12)
ax1.set_ylabel('Number of Cardmembers', fontsize=12)
ax1.set_title('Profitability Score Distribution — Full Portfolio', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# 2. Revenue Component Contribution (Portfolio Average)
ax2 = fig.add_subplot(gs[0, 2])
rev_means = [df[c].mean() for c in revenue_cols]
colors_rev = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#90CAF9']
bars = ax2.barh(rev_names[::-1], rev_means[::-1], color=colors_rev[::-1], alpha=0.85)
ax2.set_xlabel('Average $ per CM', fontsize=10)
ax2.set_title('Revenue Components\n(Portfolio Average)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, rev_means[::-1]):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', fontsize=8)

# 3. Cost Component Contribution
ax3 = fig.add_subplot(gs[1, :2])
cost_means = [df[c].mean() for c in cost_cols]
colors_cost = ['#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F44336', '#EF9A9A', '#880E4F', '#6A1B9A']
x_pos = np.arange(len(cost_names))
bars = ax3.bar(x_pos, cost_means, color=colors_cost, alpha=0.85, edgecolor='white')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(cost_names, rotation=25, ha='right', fontsize=9)
ax3.set_ylabel('Average $ per CM', fontsize=10)
ax3.set_title('Cost Components — Portfolio Average', fontsize=12, fontweight='bold')
for bar, val in zip(bars, cost_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'${val:,.0f}', ha='center', fontsize=7.5)

# 4. Top20 vs Bottom80 comparison
ax4 = fig.add_subplot(gs[1, 2])
comp_cols = ['Profitability_Score', 'Total_Revenue', 'Total_Cost']
comp_labels = ['Profit Score', 'Revenue', 'Cost']
top_means = [top[c].mean() for c in comp_cols]
bot_means = [bot[c].mean() for c in comp_cols]
x = np.arange(len(comp_labels))
w = 0.35
ax4.bar(x - w/2, top_means, w, label='Top 20%', color='#1565C0', alpha=0.85)
ax4.bar(x + w/2, bot_means, w, label='Bottom 80%', color='#E53935', alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels(comp_labels, fontsize=9)
ax4.set_ylabel('$ per CM', fontsize=10)
ax4.set_title('Top 20% vs Bottom 80%', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# 5. Rank vs Score curve
ax5 = fig.add_subplot(gs[2, :2])
rank_sample = np.linspace(0, len(df)-1, 2000, dtype=int)
ax5.plot(rank_sample + 1, df.loc[rank_sample, 'Profitability_Score'],
         color='#1565C0', linewidth=1.5, alpha=0.9)
ax5.axvline(100000, color='#E53935', linewidth=2, linestyle='--', label='Top 20% cutoff (100K)')
ax5.axhline(0, color='#43A047', linewidth=1.2, linestyle=':', label='Break-even')
ax5.fill_between(rank_sample+1, df.loc[rank_sample, 'Profitability_Score'],
                 TOP20_THRESHOLD,
                 where=rank_sample < 100000,
                 alpha=0.15, color='#1565C0', label='Top 20% Zone')
ax5.set_xlabel('Rank (sorted by Profitability Score)', fontsize=12)
ax5.set_ylabel('Profitability Score ($)', fontsize=12)
ax5.set_title('Rank-Score Curve — Profitability Waterfall', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# 6. Revenue vs Cost Scatter (with Top20 color)
ax6 = fig.add_subplot(gs[2, 2])
sample_idx = np.random.choice(len(df), 5000, replace=False)
sample = df.iloc[sample_idx]
colors_scatter = ['#1565C0' if x else '#E53935' for x in sample['is_top20pct']]
ax6.scatter(
    sample['Total_Revenue'].clip(0, sample['Total_Revenue'].quantile(0.99)),
    sample['Total_Cost'].clip(0, sample['Total_Cost'].quantile(0.99)),
    c=colors_scatter, alpha=0.3, s=2
)
ax6.set_xlabel('Total Revenue ($)', fontsize=10)
ax6.set_ylabel('Total Cost ($)', fontsize=10)
ax6.set_title('Revenue vs Cost\n(Blue=Top20%, Red=Bottom80%)', fontsize=11, fontweight='bold')

fig.suptitle('American Express Campus Challenge 2026\nProfitability Framework — Score Analysis',
             fontsize=16, fontweight='bold', y=1.01)

plt.savefig(f'{OUT_DIR}/framework_analysis.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Framework visualization saved.")

# ══════════════════════════════════════════════════════════════════════
# STEP 8: Submission File Creation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 8: Preparing Submission File")
print("─" * 70)

submission = df[['id', 'Profitability_Score']].rename(columns={'Profitability_Score': 'prediction'})
submission = submission.sort_values('id').reset_index(drop=True)  # keep original order by id

# Validation checks
print(f"\n  Validation Checks:")
print(f"    ✓ Total rows: {len(submission):,} (expected 500,000)")
assert len(submission) == 500000, f"Row count mismatch: {len(submission)}"

print(f"    ✓ Unique IDs: {submission['id'].nunique():,}")
assert submission['id'].nunique() == 500000, "Duplicate IDs!"

assert submission['prediction'].dtype in [np.float64, np.float32, np.int64], "Non-numeric prediction!"
print(f"    ✓ Prediction dtype: {submission['prediction'].dtype} (continuous numerical)")

print(f"    ✓ Columns: {submission.columns.tolist()} (only id and prediction)")
assert list(submission.columns) == ['id', 'prediction'], "Column name mismatch!"

print(f"    ✓ No nulls in prediction: {submission['prediction'].isnull().sum() == 0}")
assert submission['prediction'].isnull().sum() == 0, "Nulls found in prediction!"

# Save CSV submission
csv_path = f'{OUT_DIR}/submission.csv'
submission.to_csv(csv_path, index=False)
print(f"\n  ✓ CSV Submission saved: {csv_path}")

# Save Excel submission
xlsx_path = f'{OUT_DIR}/submission.xlsx'
submission.to_excel(xlsx_path, index=False)
print(f"  ✓ Excel Submission saved: {xlsx_path}")

# ══════════════════════════════════════════════════════════════════════
# STEP 9: Detailed Score Breakdown (Sample)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 9: Score Breakdown — Sample Cardmembers")
print("─" * 70)

all_score_cols = revenue_cols + cost_cols + ['Total_Revenue', 'Total_Cost', 'Profitability_Score']
breakdown = df.sort_values('Profitability_Score', ascending=False).head(5)[
    ['id', 'Rank'] + all_score_cols
]
print("\nTop 5 Most Profitable CMs:")
print(breakdown.to_string(index=False))

print("\nTop 5 Least Profitable CMs:")
bottom5 = df.sort_values('Profitability_Score').head(5)[['id', 'Rank'] + all_score_cols]
print(bottom5.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# STEP 10: Save Full Scored Dataset
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  STEP 10: Saving Full Scored Dataset")
print("─" * 70)

full_output_cols = ['id', 'Rank', 'is_top20pct', 'Total_Revenue', 'Total_Cost',
                    'Profitability_Score'] + revenue_cols + cost_cols
df[full_output_cols].to_csv(f'{OUT_DIR}/full_scored_data.csv', index=False)
print(f"  ✓ Full scored dataset saved: {OUT_DIR}/full_scored_data.csv")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PROFITABILITY FRAMEWORK — FINAL SUMMARY")
print("=" * 70)

print(f"""
  EQUATION:
  ──────────
  Profitability_Score =
    REVENUE:
      + {ANNUAL_FEE_REVENUE:.0f}                                       (Annual Fee)
      + (f6 + f9) × {INTERCHANGE_RATE_PREMIUM:.3f}                        (Premium Interchange)
      + (f7 + f8 + f10) × {INTERCHANGE_RATE_OTHER:.3f}                    (Other Interchange)
      + f1 × {APR_PROXY:.2f}                                    (Interest on Revolve Balance)
      + f17 × {CREDIT_LINE_WEIGHT:.4f}                               (Credit Line Quality Signal)
      + f19 × {SUPP_ACCT_VALUE:.0f} + f20 × {CHARGE_CARD_VALUE:.0f}             (Supplementary Revenue)

    COSTS:
      - f21 × {POINT_VALUE_CENTS:.4f}                               (Points Redeemed)
      - (5×f6 + 5×f9 + f7 + f8 + f10) × {POINT_VALUE_CENTS:.4f} × {ACCRUAL_COST_RATE:.2f} (Points Liability)
      - f13 × {LOUNGE_UNIT_COST:.0f}                                    (Lounge Cost)
      - f14                                          (Airline Credits Used)
      - f15 × {CAB_MONTH_COST:.0f}                                     (Cab Benefits)
      - f16                                          (Entertainment Credits)
      - f1 × f11 × {ECL_MULTIPLIER:.2f}                             (Expected Credit Loss)
      - f3 × {COLLECTION_CALL_PENALTY:.0f}                                   (Collection Risk Penalty)
      - f2 × {RETENTION_CALL_COST:.0f}                                     (Retention Cost)

  OUTPUTS:
  ─────────
  → submission.csv          : {len(submission):,} rows (id + prediction score)
  → submission.xlsx         : Same as CSV in Excel format
  → full_scored_data.csv    : Detailed breakdown per CM
  → framework_analysis.png  : Visual analysis dashboard

  TOP 20% THRESHOLD: ${TOP20_THRESHOLD:,.2f}
  Total profitable CMs:  {n_positive:,} ({n_positive/len(df)*100:.1f}%)
""")

print("  ✓  FRAMEWORK COMPLETE — Submission files are ready!")
print("=" * 70)
