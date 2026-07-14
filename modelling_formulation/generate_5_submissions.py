import os
import pandas as pd
import numpy as np

# Load dataset
exp_dir = r"C:\Users\verma\Desktop\AmEx\modelling_formulation\experiments\exp_20260705_095922"
df = pd.read_parquet(os.path.join(exp_dir, "data", "formula_dataset.parquet"))

# Load 91.33% Baseline Submission
baseline_submission_path = os.path.join(exp_dir, "submissions", "submission_B_Balanced.csv")
baseline_sub = pd.read_csv(baseline_submission_path)
baseline_top_100k = baseline_sub.nlargest(100_000, 'prediction')
baseline_ids = set(baseline_top_100k['id'])
print(f"Baseline Top 100k Size: {len(baseline_ids)}")
print("-" * 50)

# Constants
IC_TRAVEL = 0.030
IC_OTHER  = 0.020
CPP       = 0.007
URR       = 0.96
INT_RATE  = 0.24
LOUNGE    = 42.0
CAB       = 15.0
LGD       = 1.0
CALL      = 300.0
COLL      = 1000.0
SUPP_FEE  = 100.0
CL_PROXY  = 0.001

f1  = df['f1'].to_numpy(np.float64)
f2  = df['f2'].to_numpy(np.float64)
f3  = df['f3'].to_numpy(np.float64)
f4  = df['f4'].to_numpy(np.float64)
f6  = df['f6'].to_numpy(np.float64)
f7  = np.clip(df['f7'].to_numpy(np.float64), 0, None)
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
f21 = df['f21'].to_numpy(np.float64)

def save_submission(R_cross_sell=None, modify_ecl=False, dynamic_urr=False, exp_name="exp1"):
    R_interchange = (f6 + f9) * IC_TRAVEL + (f7 + f8 + f10) * IC_OTHER
    R_interest = f1 * INT_RATE
    R_supp = f19 * SUPP_FEE + f20 * SUPP_FEE
    R_credit_line = f17 * CL_PROXY

    points_earned = ((f6 + f9) * 2.0) + f7 + f8 + f10
    
    if dynamic_urr:
        urr_val = np.where(f21 > 0, 0.98, 0.75)
        C_points = points_earned * CPP * urr_val
    else:
        C_points = points_earned * CPP * URR

    C_lounge = f13 * LOUNGE
    C_airline = f14
    C_cab = f15 * CAB
    C_ent = f16

    if modify_ecl:
        C_ecl = ((f1 * f11 * LGD) - (f4 * 0.001)) + (f3 * COLL) + (f3 * f1 * 1.0)
    else:
        C_ecl = (f1 * f11 * LGD) + (f3 * COLL) + (f3 * f1 * 1.0)
        
    C_retention = f2 * CALL

    profit = (
        R_interchange + R_interest + R_supp + R_credit_line
        - C_points - C_lounge - C_airline - C_cab - C_ent - C_ecl - C_retention
    )
    if R_cross_sell is not None:
        profit += R_cross_sell

    score = pd.Series(profit, index=df.index, name='score')
    top_100k = score.nlargest(100_000).index
    
    # Create submission file with raw profit score
    sub_df = pd.DataFrame({'id': df['id'], 'prediction': profit})
    
    out_path = os.path.join(exp_dir, "submissions", f"submission_{exp_name}.csv")
    sub_df.to_csv(out_path, index=False)
    
    exp_ids = set(df.loc[top_100k, 'id'])
    return exp_ids

# Exp 1: Add R_cross_sell = f4 * 0.002
print("Generating Exp 1 (Cross-sell: R_cross_sell = f4 * 0.002)...")
exp1_ids = save_submission(R_cross_sell=f4 * 0.002, exp_name="exp1")
overlap_1 = len(exp1_ids.intersection(baseline_ids))
print(f"Exp 1 Overlap with 91.33% Baseline: {overlap_1 / 100000 * 100:.2f}% ({overlap_1} customers)")
print("-" * 50)

# Exp 2: C_ecl = (f1 * f11 * LGD) - (f4 * 0.001)
print("Generating Exp 2 (Risk Mitigator: C_ecl = (f1*f11*LGD) - (f4*0.001))...")
exp2_ids = save_submission(modify_ecl=True, exp_name="exp2")
overlap_2 = len(exp2_ids.intersection(baseline_ids))
print(f"Exp 2 Overlap with 91.33% Baseline: {overlap_2 / 100000 * 100:.2f}% ({overlap_2} customers)")
print("-" * 50)

# Exp 3: dynamic_urr = np.where(f21 > 0, 0.98, 0.75)
print("Generating Exp 3 (Dynamic URR: active=0.98, inactive=0.75)...")
exp3_ids = save_submission(dynamic_urr=True, exp_name="exp3")
overlap_3 = len(exp3_ids.intersection(baseline_ids))
print(f"Exp 3 Overlap with 91.33% Baseline: {overlap_3 / 100000 * 100:.2f}% ({overlap_3} customers)")
print("-" * 50)

# Exp 4: Combo 1 & 3
print("Generating Exp 4 (Combo 1 & 3: Cross-sell + Dynamic URR)...")
exp4_ids = save_submission(R_cross_sell=f4 * 0.002, dynamic_urr=True, exp_name="exp4")
overlap_4 = len(exp4_ids.intersection(baseline_ids))
print(f"Exp 4 Overlap with 91.33% Baseline: {overlap_4 / 100000 * 100:.2f}% ({overlap_4} customers)")
print("-" * 50)

# Exp 5: Combo 2 & 3
print("Generating Exp 5 (Combo 2 & 3: Risk Mitigator + Dynamic URR)...")
exp5_ids = save_submission(modify_ecl=True, dynamic_urr=True, exp_name="exp5")
overlap_5 = len(exp5_ids.intersection(baseline_ids))
print(f"Exp 5 Overlap with 91.33% Baseline: {overlap_5 / 100000 * 100:.2f}% ({overlap_5} customers)")
print("-" * 50)

print("All 5 submission files saved successfully to submissions directory!")
