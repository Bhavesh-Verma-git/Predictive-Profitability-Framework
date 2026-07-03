"""
=======================================================================
  American Express Campus Challenge 2026 - Round 1
  EXPLORATORY DATA ANALYSIS (EDA)
=======================================================================
  Features (fully unmasked via feature_description.csv):
    f1  - Average Revolve Balance in last 12m
    f2  - Cancellation Calls in last 12m
    f3  - Cancellation Calls due to Collection
    f4  - Rewards Points Balance
    f5  - Total Spend in last 12m
    f6  - Airlines Spend in 12m
    f7  - Other Spend in 12m
    f8  - Entertainment Spend in 12m
    f9  - Lodging Spend in 12m
    f10 - Dining Spend in 12m
    f11 - Average Risk Score in 12m
    f12 - Login Counts to website
    f13 - Lounge Access Count
    f14 - Credits used in airlines
    f15 - Cab benefits usage
    f16 - Entertainment Credit Used Amount
    f17 - Total Lend Line Amount
    f18 - Total Consumer Lend Line Amount
    f19 - Number of Supplementary Accounts
    f20 - Count of Active Charge Cards
    f21 - Rewards point redeemed in 12months
    f22 - Emails Open in Last 6 months
    f23 - Emails Clicked in Last 6 months
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
# Base directory is the folder where this script lives (works on any machine)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, '6a3eb196bc7a3_campus_challenge_r1_data.csv')
OUT_DIR = os.path.join(BASE, 'eda_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature Metadata ───────────────────────────────────────────────────
FEATURE_META = {
    'f1':  {'name': 'Avg Revolve Balance (12m)',        'category': 'Balance/Revolve',  'revenue_type': 'revenue'},
    'f2':  {'name': 'Cancellation Calls (12m)',         'category': 'Engagement',       'revenue_type': 'cost'},
    'f3':  {'name': 'Cancellation Calls (Collection)',  'category': 'Risk',             'revenue_type': 'cost'},
    'f4':  {'name': 'Rewards Points Balance',           'category': 'Rewards',          'revenue_type': 'cost'},
    'f5':  {'name': 'Total Spend (12m)',                'category': 'Spend',            'revenue_type': 'revenue'},
    'f6':  {'name': 'Airlines Spend (12m)',             'category': 'Spend',            'revenue_type': 'revenue'},
    'f7':  {'name': 'Other Spend (12m)',                'category': 'Spend',            'revenue_type': 'revenue'},
    'f8':  {'name': 'Entertainment Spend (12m)',        'category': 'Spend',            'revenue_type': 'revenue'},
    'f9':  {'name': 'Lodging Spend (12m)',              'category': 'Spend',            'revenue_type': 'revenue'},
    'f10': {'name': 'Dining Spend (12m)',               'category': 'Spend',            'revenue_type': 'revenue'},
    'f11': {'name': 'Avg Risk Score (12m)',             'category': 'Risk',             'revenue_type': 'cost'},
    'f12': {'name': 'Login Counts (website)',           'category': 'Engagement',       'revenue_type': 'neutral'},
    'f13': {'name': 'Lounge Access Count',              'category': 'Benefits',         'revenue_type': 'cost'},
    'f14': {'name': 'Airline Credits Used',             'category': 'Benefits',         'revenue_type': 'cost'},
    'f15': {'name': 'Cab Benefits Usage',               'category': 'Benefits',         'revenue_type': 'cost'},
    'f16': {'name': 'Entertainment Credit Amount',      'category': 'Benefits',         'revenue_type': 'cost'},
    'f17': {'name': 'Total Lend Line Amount',           'category': 'Credit',           'revenue_type': 'revenue'},
    'f18': {'name': 'Total Consumer Lend Line Amt',     'category': 'Credit',           'revenue_type': 'revenue'},
    'f19': {'name': 'Supplementary Accounts',          'category': 'Engagement',       'revenue_type': 'revenue'},
    'f20': {'name': 'Active Charge Cards Count',        'category': 'Engagement',       'revenue_type': 'revenue'},
    'f21': {'name': 'Rewards Points Redeemed (12m)',   'category': 'Rewards',          'revenue_type': 'cost'},
    'f22': {'name': 'Emails Opened (6m)',               'category': 'Engagement',       'revenue_type': 'neutral'},
    'f23': {'name': 'Emails Clicked (6m)',              'category': 'Engagement',       'revenue_type': 'neutral'},
}

FEATURES = list(FEATURE_META.keys())

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Load Data
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 1: Loading Data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Basic Statistics
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 2: Descriptive Statistics")
print("=" * 70)

stats_df = df[FEATURES].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
stats_df['skewness'] = df[FEATURES].skew()
stats_df['kurtosis'] = df[FEATURES].kurtosis()
stats_df['missing'] = df[FEATURES].isnull().sum()
stats_df['missing_pct'] = (df[FEATURES].isnull().sum() / len(df) * 100).round(2)
stats_df['zeros'] = (df[FEATURES] == 0).sum()
stats_df['zeros_pct'] = (stats_df['zeros'] / len(df) * 100).round(2)
stats_df['feature_name'] = pd.Series({f: FEATURE_META[f]['name'] for f in FEATURES})
stats_df['category'] = pd.Series({f: FEATURE_META[f]['category'] for f in FEATURES})

print("\nDescriptive Statistics:")
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
print(stats_df[['feature_name', 'category', 'mean', 'std', 'min', '50%', 'max',
                 'skewness', 'missing_pct', 'zeros_pct']].to_string())

stats_df.to_csv(f'{OUT_DIR}/eda_statistics.csv')
print(f"\n  → Stats saved to {OUT_DIR}/eda_statistics.csv")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Feature Classification
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 3: Feature Classification by Type")
print("=" * 70)

feature_types = {}
for f in FEATURES:
    col = df[f].dropna()
    n_unique = col.nunique()
    col_min, col_max = col.min(), col.max()
    if set(col.unique()).issubset({0, 1}):
        ftype = 'Binary'
    elif n_unique <= 30 and col.dtype in [np.int64, np.float64] and (col % 1 == 0).all():
        ftype = 'Discrete/Count'
    elif col_max <= 1.0 and col_min >= 0.0 and col.dtype == np.float64:
        ftype = 'Bounded [0,1]'
    elif col.dtype == np.float64 or col.dtype == np.int64:
        ftype = 'Continuous'
    else:
        ftype = 'Other'
    feature_types[f] = ftype

print("\nFeature Type Classification:")
for f, ftype in feature_types.items():
    meta = FEATURE_META[f]
    print(f"  {f:4s}  [{ftype:18s}]  {meta['name']:40s}  Category: {meta['category']}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Distribution Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 4: Distribution Analysis & Skewness")
print("=" * 70)

fig, axes = plt.subplots(6, 4, figsize=(24, 30))
axes = axes.flatten()

color_map = {
    'Spend': '#2196F3',
    'Balance/Revolve': '#4CAF50',
    'Risk': '#F44336',
    'Rewards': '#FF9800',
    'Benefits': '#9C27B0',
    'Credit': '#00BCD4',
    'Engagement': '#795548',
}

for idx, f in enumerate(FEATURES):
    ax = axes[idx]
    col = df[f].dropna()
    cat = FEATURE_META[f]['category']
    color = color_map.get(cat, '#607D8B')

    if feature_types[f] in ['Binary', 'Discrete/Count']:
        vc = col.value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(f"{f}: {FEATURE_META[f]['name']}\n[{feature_types[f]}]", fontsize=7.5, fontweight='bold')
    else:
        # Clip extreme outliers for visualization
        p1, p99 = col.quantile(0.01), col.quantile(0.99)
        clipped = col.clip(p1, p99)
        ax.hist(clipped, bins=60, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(col.median(), color='red', linestyle='--', linewidth=1.2, label=f'Median={col.median():.1f}')
        skew_val = col.skew()
        ax.set_title(f"{f}: {FEATURE_META[f]['name']}\n[Skew={skew_val:.2f}, Missing={df[f].isnull().sum()/len(df)*100:.1f}%]",
                     fontsize=7.5, fontweight='bold')

    ax.set_xlabel(f, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Hide unused axes
for idx in range(len(FEATURES), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('American Express Campus Challenge 2026 — Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_feature_distributions.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Distribution plots saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Missing Value Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 5: Missing Value Deep-Dive")
print("=" * 70)

missing = df[FEATURES].isnull().sum()
missing_pct = missing / len(df) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct}).sort_values('pct', ascending=False)
missing_df = missing_df[missing_df['count'] > 0]
print(missing_df.to_string())

# Patterns: which features tend to be missing together?
print("\nCo-missing pattern analysis (top combos):")
miss_mask = df[FEATURES].isnull()
co_miss = miss_mask.T.dot(miss_mask).astype(int)
print(co_miss.to_string())

if len(missing_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(missing_df.index[::-1],
                   missing_df['pct'][::-1],
                   color='#E53935', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Missing %', fontsize=12)
    ax.set_title('Missing Value Percentage by Feature', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, missing_df['pct'][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/02_missing_values.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → Missing value chart saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Correlation Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 6: Correlation Matrix")
print("=" * 70)

corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, ax=ax,
            xticklabels=[f"{f}\n{FEATURE_META[f]['name'][:15]}" for f in FEATURES],
            yticklabels=[f"{f}: {FEATURE_META[f]['name'][:20]}" for f in FEATURES])
ax.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_correlation_matrix.png', dpi=110, bbox_inches='tight')
plt.close()
print(f"  → Correlation matrix saved.")

# Top correlations
print("\nTop 15 Positive Correlations:")
corr_pairs = corr.unstack()
corr_pairs = corr_pairs[corr_pairs < 0.9999].sort_values(ascending=False)
print(corr_pairs.head(15).to_string())
print("\nTop 10 Negative Correlations:")
print(corr_pairs.tail(10).to_string())

corr.to_csv(f'{OUT_DIR}/correlation_matrix.csv')

# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Spend Decomposition
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 7: Spend Decomposition Analysis")
print("=" * 70)

spend_cols = ['f6', 'f7', 'f8', 'f9', 'f10']
spend_names = ['Airlines', 'Other', 'Entertainment', 'Lodging', 'Dining']

# Compute reconstructed spend
df['spend_reconstructed'] = df[spend_cols].fillna(0).sum(axis=1)
df['spend_diff'] = df['f5'].fillna(0) - df['spend_reconstructed']

print(f"\nf5 (Total Spend) stats:")
print(df['f5'].describe())
print(f"\nReconstructed Spend (f6+f7+f8+f9+f10) stats:")
print(df['spend_reconstructed'].describe())
print(f"\nDifference (f5 - reconstructed):")
print(df['spend_diff'].describe())

# Spend category breakdown
spend_totals = df[spend_cols].fillna(0).sum()
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie chart
wedges, texts, autotexts = axes[0].pie(
    spend_totals,
    labels=spend_names,
    autopct='%1.1f%%',
    startangle=140,
    colors=['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336'],
    pctdistance=0.8
)
axes[0].set_title('Spend Category Breakdown (Total Portfolio)', fontsize=13, fontweight='bold')

# Box plots per category
spend_data_long = df[spend_cols].fillna(0).melt(var_name='Feature', value_name='Spend')
spend_data_long['Category'] = spend_data_long['Feature'].map(dict(zip(spend_cols, spend_names)))
spend_clipped = spend_data_long.copy()
spend_clipped['Spend'] = spend_clipped.groupby('Category')['Spend'].transform(
    lambda x: x.clip(0, x.quantile(0.95))
)
bp = axes[1].boxplot([df[f].fillna(0).clip(0, df[f].quantile(0.95)) for f in spend_cols],
                patch_artist=True,
                boxprops=dict(facecolor='#2196F3', alpha=0.7))
axes[1].set_xticks(range(1, len(spend_names)+1))
axes[1].set_xticklabels(spend_names)
axes[1].set_title('Spend Distribution by Category (clipped at 95th pct)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Spend ($)')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_spend_decomposition.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Spend decomposition chart saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 8: Risk Score Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 8: Risk Score (f11) Analysis")
print("=" * 70)

risk = df['f11'].dropna()
print(f"\nRisk Score Stats:")
print(risk.describe())
print(f"\nPercentile Breakdown:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:2d}: {risk.quantile(p/100):.6f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(risk, bins=80, color='#F44336', alpha=0.8, edgecolor='white')
axes[0].set_title('Risk Score Distribution (f11)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Risk Score')

# Risk vs Revolve Balance
valid = df[['f1', 'f11']].dropna()
axes[1].scatter(valid['f11'], valid['f1'].clip(0, valid['f1'].quantile(0.95)),
                alpha=0.05, s=0.5, color='#F44336')
axes[1].set_xlabel('Risk Score (f11)')
axes[1].set_ylabel('Revolve Balance (f1)')
axes[1].set_title('Risk Score vs Revolve Balance', fontsize=12, fontweight='bold')

# Risk vs Total Spend
valid2 = df[['f5', 'f11']].dropna()
axes[2].scatter(valid2['f11'], valid2['f5'].clip(0, valid2['f5'].quantile(0.95)),
                alpha=0.05, s=0.5, color='#FF9800')
axes[2].set_xlabel('Risk Score (f11)')
axes[2].set_ylabel('Total Spend (f5)')
axes[2].set_title('Risk Score vs Total Spend', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_risk_analysis.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Risk analysis chart saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9: Benefit Usage Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 9: Benefit Usage Analysis")
print("=" * 70)

benefit_cols = {'f13': 'Lounge Access', 'f14': 'Airline Credits', 'f15': 'Cab Benefits', 'f16': 'Entertainment Credit'}
for f, name in benefit_cols.items():
    col = df[f].dropna()
    print(f"\n{f} - {name}:")
    print(f"  Min={col.min():.2f}, Max={col.max():.2f}, Mean={col.mean():.2f}, Median={col.median():.2f}")
    print(f"  % Zero={(col == 0).mean()*100:.1f}%, % Non-zero={(col != 0).mean()*100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = ['#9C27B0', '#673AB7', '#3F51B5', '#2196F3']

for idx, (f, name) in enumerate(benefit_cols.items()):
    col = df[f].dropna()
    ax = axes[idx]
    if col.nunique() <= 50:
        vc = col.value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=colors[idx], alpha=0.8)
    else:
        ax.hist(col.clip(0, col.quantile(0.99)), bins=50, color=colors[idx], alpha=0.8)
    ax.set_title(f'{f}: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel(f)

plt.suptitle('Benefit Usage Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_benefit_usage.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Benefit usage chart saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10: Engagement Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 10: Customer Engagement Analysis")
print("=" * 70)

engagement_cols = {'f2': 'Cancel Calls', 'f3': 'Collection Calls', 'f12': 'Website Logins',
                   'f19': 'Supplementary Accounts', 'f20': 'Active Charge Cards',
                   'f22': 'Emails Opened', 'f23': 'Emails Clicked'}
for f, name in engagement_cols.items():
    col = df[f].dropna()
    print(f"  {f:4s} - {name:30s}: min={col.min():.0f}, max={col.max():.0f}, mean={col.mean():.2f}, "
          f"median={col.median():.0f}, zeros={100*(col==0).mean():.1f}%")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11: Revolve Balance vs Spend Segmentation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 11: Customer Segmentation Insights")
print("=" * 70)

df['is_revolver'] = (df['f1'].fillna(0) > 0).astype(int)
df['spend_tier'] = pd.qcut(df['f5'].fillna(0), q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])

print("\nRevolvers vs Non-Revolvers:")
print(df.groupby('is_revolver')[['f5', 'f1', 'f11', 'f13']].agg(['mean', 'median']).to_string())

print("\nSpend Tier Breakdown:")
print(df.groupby('spend_tier')[['f1', 'f11', 'f13', 'f4', 'f21']].mean().to_string())

# Customer Archetype Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter: Spend vs Revolve Balance, colored by Risk
valid3 = df[['f5', 'f1', 'f11']].dropna()
sc = axes[0].scatter(
    valid3['f5'].clip(0, valid3['f5'].quantile(0.98)),
    valid3['f1'].clip(0, valid3['f1'].quantile(0.98)),
    c=valid3['f11'], cmap='RdYlGn_r', alpha=0.15, s=0.8
)
plt.colorbar(sc, ax=axes[0], label='Risk Score (f11)')
axes[0].set_xlabel('Total Spend - f5 ($)', fontsize=11)
axes[0].set_ylabel('Avg Revolve Balance - f1 ($)', fontsize=11)
axes[0].set_title('Spend vs Revolve Balance\n(colored by Risk Score)', fontsize=12, fontweight='bold')

# Revolver breakdown
rev_counts = df['is_revolver'].value_counts()
axes[1].pie(rev_counts, labels=['Non-Revolver', 'Revolver'], autopct='%1.1f%%',
            colors=['#4CAF50', '#2196F3'], startangle=90)
axes[1].set_title('Revolver vs Non-Revolver Split', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_customer_segmentation.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  → Customer segmentation chart saved.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12: Reward Points Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 12: Rewards Points Analysis")
print("=" * 70)

print("\nf4 (Rewards Points Balance):", df['f4'].describe().to_dict())
print("\nf21 (Points Redeemed 12m):", df['f21'].describe().to_dict())

# Compute expected points generated from spend (5x airlines, 5x lodging, 1x others)
df['pts_generated_estimate'] = (
    df['f6'].fillna(0) * 5 +     # 5x airlines
    df['f9'].fillna(0) * 5 +     # 5x lodging
    df['f7'].fillna(0) * 1 +     # 1x other
    df['f8'].fillna(0) * 1 +     # 1x entertainment
    df['f10'].fillna(0) * 1      # 1x dining
)
print(f"\nEstimated Points Generated from Spend:")
print(df['pts_generated_estimate'].describe())

# Correlation: points generated vs points balance
valid4 = df[['pts_generated_estimate', 'f4', 'f21']].dropna()
print(f"\nCorr (pts_generated, f4): {valid4['pts_generated_estimate'].corr(valid4['f4']):.4f}")
print(f"Corr (pts_generated, f21): {valid4['pts_generated_estimate'].corr(valid4['f21']):.4f}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13: EDA Summary Report
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 13: EDA Summary & Findings")
print("=" * 70)

summary = """
KEY EDA FINDINGS
================

1. REVENUE DRIVERS (Positive Profitability):
   ─ f5 (Total Spend): Highly right-skewed. High-spend CMs generate interchange revenue.
   ─ f6 (Airlines Spend): Key 5x rewards category. High earner for Amex in terms of interchange.
   ─ f9 (Lodging Spend): Another 5x category. Significant revenue potential.
   ─ f1 (Revolve Balance): Revolvers generate ~17-25% APR interest revenue.
   ─ f17/f18 (Lend Lines): Higher credit lines indicate creditworthy, profitable CMs.
   ─ f19 (Supplementary Accounts): More supp accounts = more spend = more revenue.
   ─ f20 (Active Charge Cards): More cards = broader relationship = more revenue.

2. COST DRIVERS (Negative Profitability):
   ─ f4 (Rewards Points Balance): Unspent liability cost. Huge balance = potential redemption cost.
   ─ f21 (Points Redeemed): Direct cash-equivalent redemption cost (1-2 cents/point).
   ─ f13 (Lounge Access): Each visit costs Amex $30-$50 in lounge fees.
   ─ f14 (Airline Credits): Direct dollar credit reimbursement cost ($150-$250/yr budget).
   ─ f15 (Cab Benefits): Monthly cab credit utilization cost ($15/mo = $180/yr).
   ─ f16 (Entertainment Credit): Direct credit cost ($180-$280/yr budget).
   ─ f11 (Risk Score): Higher risk → higher expected credit loss (ECL).
   ─ f2 (Cancellation Calls): Retention effort + possible retention offers given.
   ─ f3 (Collection Calls): Severe risk signal. Calls due to collection = high default risk.

3. ENGAGEMENT SIGNALS (Neutral/Behavioral Indicators):
   ─ f12 (Website Logins): High engagement = better retention, loyalty signal.
   ─ f22/f23 (Email Activity): Open/click rates signal engagement, not direct revenue.

4. MISSING VALUES:
   ─ Spend subcategories (f6-f10) have the most nulls — customers may not spend in all categories.
   ─ Risk score (f11) has some nulls — treat as moderate-high risk when missing.
   ─ Credits (f14-f16) are null when unused — treat as 0 (no cost incurred).

5. DISTRIBUTION INSIGHTS:
   ─ All spend variables are heavily right-skewed (long right tail — super-spenders exist).
   ─ Risk score (f11) appears to be a probability (0-1) — bounded variable.
   ─ Lounge access (f13), cab usage (f15) are count variables (small integers).
   ─ Revolve balance (f1) is zero for a large fraction of customers (transactors).
"""
print(summary)

with open(f'{OUT_DIR}/eda_summary.txt', 'w') as fh:
    fh.write(summary)

print(f"\n  → All EDA outputs saved to: {OUT_DIR}/")
print("\n" + "=" * 70)
print("  EDA COMPLETE ✓")
print("=" * 70)
