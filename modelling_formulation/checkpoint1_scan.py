"""
CHECKPOINT 1 — Dataset & Asset Inspection Script
Modelling Formulation Project | American Express Campus Challenge 2026
"""
import pandas as pd
import os

DATA_PATH = r'C:\Users\verma\Desktop\AmEx\6a3eb196bc7a3_campus_challenge_r1_data.csv'
FORMULA_PATH = r'C:\Users\verma\Desktop\AmEx\amex_campus_challenge_r1-main\amex_campus_challenge_r1-main\pipeline.py'
TEMPLATE_PATH = r'C:\Users\verma\Desktop\AmEx\6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx'

print("=" * 70)
print("CHECKPOINT 1 — PROJECT INITIALIZATION SCAN")
print("=" * 70)

# --- 1. Dataset ---
df = pd.read_csv(DATA_PATH)
print(f"\n[1] DATASET")
print(f"    Shape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Columns       : {list(df.columns)}")
print(f"    ID column     : 'id' | dtype={df['id'].dtype} | unique={df['id'].nunique():,}")

feature_cols = [c for c in df.columns if c != 'id']
print(f"    Feature count : {len(feature_cols)} (f1 through f23)")

print(f"\n[2] DATA TYPES")
print(df.dtypes.to_string())

print(f"\n[3] MISSING VALUES (features only)")
missing = df[feature_cols].isnull().sum()
missing_nonzero = missing[missing > 0]
if len(missing_nonzero) == 0:
    print("    No missing values found.")
else:
    for col, cnt in missing_nonzero.items():
        pct = 100.0 * cnt / len(df)
        print(f"    {col}: {cnt:,} missing ({pct:.2f}%)")

print(f"\n[4] BASIC STATISTICS (numeric features)")
print(df[feature_cols].describe().T[['min', 'mean', 'max', 'std']].to_string())

# --- 2. Assets ---
print(f"\n[5] ASSET VERIFICATION")
print(f"    Formula (pipeline.py) exists : {os.path.exists(FORMULA_PATH)}")
print(f"    Submission template exists   : {os.path.exists(TEMPLATE_PATH)}")

# Submission template sheets
xl = pd.ExcelFile(TEMPLATE_PATH)
print(f"    Template sheets              : {xl.sheet_names}")
df_pred_template = pd.read_excel(TEMPLATE_PATH, sheet_name='Predictions')
df_fw_template = pd.read_excel(TEMPLATE_PATH, sheet_name='Profitability Framework')
print(f"    Predictions columns          : {list(df_pred_template.columns)}")
print(f"    Framework columns            : {list(df_fw_template.columns)}")
print(f"    Framework sections           : {list(df_fw_template['Section'].dropna())}")

print("\n" + "=" * 70)
print("SCAN COMPLETE")
print("=" * 70)
