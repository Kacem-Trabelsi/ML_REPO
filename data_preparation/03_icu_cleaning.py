"""
CRISP-DM: Data Preparation â€” Script 3
Dataset : Kaggle_Sirio_Libanes_ICU_Prediction.xlsx  (ICU Admission Prediction)
Outputs :
  cleaned/icu_cleaned_full.csv      â€” all 5 time-windows, imputed & scaled
  cleaned/icu_cleaned_window0_2.csv â€” first window only (0-2h), for early-admission models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE     = os.path.join(BASE_DIR, "Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
OUTPUT_DIR     = os.path.join(BASE_DIR, "cleaned")
OUTPUT_FULL    = os.path.join(OUTPUT_DIR, "icu_cleaned_full.csv")
OUTPUT_W02     = os.path.join(OUTPUT_DIR, "icu_cleaned_window0_2.csv")
SCALER_FILE    = os.path.join(OUTPUT_DIR, "icu_scaler.pkl")

# â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[1] Loading Excel file (may take a moment)â€¦")
df = pd.read_excel(INPUT_FILE)
print(f"    Loaded  â†’  {df.shape[0]} rows Ã— {df.shape[1]} cols")
print(f"    Unique patients : {df['PATIENT_VISIT_IDENTIFIER'].nunique()}")
print(f"    Windows per patient: {df.groupby('PATIENT_VISIT_IDENTIFIER').size().value_counts().to_dict()}")
print(f"    ICU distribution:\n{df['ICU'].value_counts().to_string()}")

# â”€â”€ 2. Sort by patient + time window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
WINDOW_ORDER = {"0-2": 0, "2-4": 1, "4-6": 2, "6-12": 3, "ABOVE_12": 4}
df["WINDOW_ORDER"] = df["WINDOW"].map(WINDOW_ORDER)
df = df.sort_values(["PATIENT_VISIT_IDENTIFIER", "WINDOW_ORDER"]).reset_index(drop=True)
print("\n[2] Sorted by PATIENT_VISIT_IDENTIFIER â†’ WINDOW order")

# â”€â”€ 3. Ordinal-encode AGE_PERCENTIL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
age_map = {
    "10th": 1, "20th": 2, "30th": 3, "40th": 4, "50th": 5,
    "60th": 6, "70th": 7, "80th": 8, "90th": 9, "Above 90th": 10
}
df["AGE_PERCENTIL"] = df["AGE_PERCENTIL"].map(age_map)
unmapped = df["AGE_PERCENTIL"].isna().sum()
print(f"\n[3] Ordinal-encoded AGE_PERCENTIL (10th=1 â€¦ Above90th=10)  |  unmapped: {unmapped}")

# â”€â”€ 4. Identify column groups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ID_COLS      = ["PATIENT_VISIT_IDENTIFIER", "WINDOW", "WINDOW_ORDER"]
TARGET_COL   = "ICU"
# Columns that are fully present (demographic / disease flags)
DEMO_COLS    = ["AGE_ABOVE65", "AGE_PERCENTIL", "GENDER",
                "DISEASE GROUPING 1", "DISEASE GROUPING 2", "DISEASE GROUPING 3",
                "DISEASE GROUPING 4", "DISEASE GROUPING 5", "DISEASE GROUPING 6",
                "HTN", "IMMUNOCOMPROMISED", "OTHER"]
# All remaining columns (lab / vital statistics per window) are continuous
CONTINUOUS_COLS = [c for c in df.columns
                   if c not in ID_COLS + [TARGET_COL] + DEMO_COLS]

print(f"\n[4] Column groups:")
print(f"    Demo / flag cols : {len(DEMO_COLS)}")
print(f"    Continuous cols  : {len(CONTINUOUS_COLS)}")

# â”€â”€ 5. Missing value analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
miss_before = df[CONTINUOUS_COLS].isnull().sum().sum()
miss_pct    = df[CONTINUOUS_COLS].isnull().mean() * 100
print(f"\n[5] Missing values BEFORE imputation: {miss_before:,}")
print(f"    Continuous cols with >50% missing: {(miss_pct > 50).sum()}")

# â”€â”€ 6. Within-patient forward-fill then backward-fill â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   Clinical vitals tend to be stable across adjacent windows â€” fill from
#   the nearest observed measurement for the same patient.
df[CONTINUOUS_COLS] = (
    df.groupby("PATIENT_VISIT_IDENTIFIER")[CONTINUOUS_COLS]
    .transform(lambda x: x.ffill().bfill())
)
miss_after_ffill = df[CONTINUOUS_COLS].isnull().sum().sum()
print(f"\n[6] After within-patient ffill+bfill: {miss_after_ffill:,} missing remaining")

# â”€â”€ 7. Median imputation for any remaining missing values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
col_medians = df[CONTINUOUS_COLS].median()
df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].fillna(col_medians)
miss_after_median = df[CONTINUOUS_COLS].isnull().sum().sum()
print(f"[7] After global median imputation:    {miss_after_median:,} missing remaining")

# â”€â”€ 8. Outlier capping on continuous columns (1stâ€“99th percentile) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[8] Winsorising continuous columns (1stâ€“99th percentile)â€¦")
total_capped = 0
for col in CONTINUOUS_COLS:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    n  = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    total_capped += n
print(f"    Total values capped across all continuous cols: {total_capped:,}")

# â”€â”€ 9. Feature scaling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])

with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n[9] StandardScaler applied to {len(CONTINUOUS_COLS)} continuous columns.")
print(f"    Scaler saved  â†’  {SCALER_FILE}")

# â”€â”€ 10. Final summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[10] Final shape (full):     {df_scaled.shape[0]} rows Ã— {df_scaled.shape[1]} cols")
print(f"     Missing values: {df_scaled.drop(columns=ID_COLS).isnull().sum().sum()}")
print(f"     ICU distribution:\n{df_scaled[TARGET_COL].value_counts().to_string()}")

# â”€â”€ 11. Save full dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
save_cols = [c for c in df_scaled.columns if c != "WINDOW_ORDER"]
df_scaled[save_cols].to_csv(OUTPUT_FULL, index=False)
print(f"\n[11] Saved full dataset  â†’  {OUTPUT_FULL}")

# â”€â”€ 12. Save first-window (0-2h) dataset for early-admission models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_w02 = df_scaled[df_scaled["WINDOW"] == "0-2"][save_cols].reset_index(drop=True)
df_w02 = df_w02.drop(columns=["PATIENT_VISIT_IDENTIFIER", "WINDOW"])
print(f"\n[12] First-window (0-2h) subset: {df_w02.shape[0]} rows Ã— {df_w02.shape[1]} cols")
df_w02.to_csv(OUTPUT_W02, index=False)
print(f"     Saved  â†’  {OUTPUT_W02}")

